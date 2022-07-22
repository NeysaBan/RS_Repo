import time
import numpy as np
import pandas as pd
import wandb
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

import sys
sys.path.append(r"/data/RS/RS_Repo/")

from src.utils.DeepCoNN.data_utils import getData, trainSet, valtestSet


def date(f='%Y-%m-%d %H:%M:%S'):
    return time.strftime(f, time.localtime())

def train(model, args, word_dict):
    

    reviews_by_user, reviews_by_item = getData(args, word_dict)

    train_df = pd.read_csv(args.csv_path + "train.csv")
    train_dataset = trainSet(train_df, word_dict, args, reviews_by_user, reviews_by_item)
    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True)

    valid_mse = 500
    print(f'{date()} #### Initial validation mse {valid_mse:.6f}')
    start_time = time.perf_counter()

    # opt = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=args.l2_regularization)
    opt = torch.optim.Adam(model.parameters(), args.lr)
    lr_sch = torch.optim.lr_scheduler.ExponentialLR(opt, args.lr_decay)
    ## factor is gamma
    # lr_sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=3, verbose=True, threshold=0.01, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    # lr_sch = torch.optim.lr_scheduler.StepLR(opt, step_size=35, gamma=0.1, last_epoch=-1) ## step_size == 15的意思是在第15个epoch
    
    
    best_mse, best_epoch = valid_mse, 0
    best_latent = None
    early_stopping = 0

    print(f'{date()} #### Start training!')
    for epoch in range(args.epochs):


        model.train()
        total_loss, total_samples = 0, 0

        latent = {}
        latent['user'] = {}
        latent['item'] = {}

        for batch in train_dataloader:
            user_reviews, item_reviews, ratings, user_ids, item_ids = map(lambda x: x.to(args.device), batch)
            predicts, user_latents, item_latents = model(user_reviews, item_reviews)

            ## recording latent
            for idx, (user_id, item_id, user_latent, item_latent) in enumerate(zip(user_ids, item_ids, user_latents, item_latents)):
                latent['user'][user_id.item()] = user_latent
                latent['item'][item_id.item()] = item_latent

            loss_mse = F.mse_loss(predicts, ratings, reduction='sum')
            #loss = torch.sqrt(nn.MSELoss()(predicts, ratings))
            #loss = F.cross_entropy(predicts, (ratings - 1).long().squeeze())
            opt.zero_grad() 
            loss_mse.backward()  
            opt.step() 

            total_loss += loss_mse.item()
            total_samples += predicts.shape[0]

        
        train_loss = total_loss / total_samples
        valid_mse = evaluate(args, model, latent)
        wandb.log({'valid_mse':np.float(valid_mse), 'train_loss':train_loss, 'epoch': epoch})
        print(f"{date()} #### Epoch {epoch:3d}; train loss {train_loss:.6f}; validation mse {valid_mse:.6f}; ")
        print("learning rate is {}".format(opt.state_dict()['param_groups'][0]['lr']))  ## print learning rate

        lr_sch.step()
        # lr_sch.step(valid_mse) #ReduceLROnPlatea
        
        if best_mse > valid_mse:
        # if best_mse - valid_mse >= 0.01:
            best_mse = valid_mse
            best_latent = latent
            best_epoch = epoch
            early_stopping = 0
            torch.save(model.state_dict(), args.model_file)
            wandb.save(args.model_file)
            print("save done!")
        # else: ## (0,0.01)则说明变动不大， (-无穷， 0)说明best_mse<valid_mse，效果降低了
        #     early_stopping += 1
        #     print("--------Early_stopping = {}".format(early_stopping))
        # if early_stopping == 5:
        #     break

    end_time = time.perf_counter()
    print(f'{date()}## End training! The best epoch is {best_epoch}. Time used {end_time - start_time:.0f} seconds.')

    return best_latent

def evaluate(args, model, latent):

    model.eval()

    ## data
    val_dataset = valtestSet(args.csv_path + 'valid.csv', latent, args)
    val_dataloader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle = False)
    ## the network which evaluating needs
    fm = model.fm.to(args.device)

    total_mse, sample_count = 0, 0

    with torch.no_grad():
        for data in val_dataloader:
            user_latents, item_latents, ratings = map(lambda x: x.to(args.device), data)
            concat_latents = torch.cat((user_latents, item_latents), dim=1)
            predicts = fm(concat_latents).squeeze(1)
            total_mse += F.mse_loss(predicts, ratings, reduction='sum').item()
            
            sample_count += len(ratings)


    return total_mse / sample_count


def test(model, args, latent):

    test_dataset = valtestSet(args.csv_path + 'test.csv', latent, args)
    test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False)

    fm = model.fm.to(args.device)

    res = []

    with torch.no_grad():
        for batch in test_dataloader:

            user_latents, item_latents = map(lambda x: x.to(args.device), batch)
            concat_latents = torch.cat((user_latents, item_latents), dim=1)

            prediction = fm(concat_latents)
            prediction = prediction.cpu()
            prediction = prediction.numpy()
            prediction = prediction.flatten()
            prediction = list(prediction)
            res += prediction

    res = np.array(res)

    dataframe = pd.DataFrame({'ratings': res})


    dataframe.index = range(0 , len(dataframe) + 0)  
    index = dataframe.index    
    dataframe.insert(0, 'ID',index)   

    dataframe.loc[dataframe['ratings'] > 5.0, 'ratings']= 5.0
    dataframe.loc[dataframe['ratings'] < 1.0, 'ratings']= 1.0
    dataframe.to_csv(args.result_file, index=False, sep=',')
    wandb.save(args.result_file)
    print("----------------End! success!---------------")