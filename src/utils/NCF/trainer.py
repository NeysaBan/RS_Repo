from datetime import datetime
import torch
import numpy as np
import os
import wandb
import pandas as pd
import sys


sys.path.append(r"/data/RS/RS_Repo/")
from src.utils.metrics import precision

def train(args, train_loader, val_loader, model, loss_function, optimizer):

    best_precision = 0.0
    best_epoch = -1
    record_loss = 0.0

    for epoch in range(args.epochs):

        print(f"{datetime.now()} ######  This is epoch {epoch}")

        model.train()
        train_loss = 0
        for idx ,data in enumerate(train_loader):
            
            user, item, label = data
            
            user = user.cuda()
            item = item.cuda()
            label = label.cuda()
            
            out = model(user, item, label)
            loss = loss_function(out["prediction"].squeeze(1), out["labels"].float()) 
            train_loss = train_loss + loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"{datetime.now()} Train loss is {train_loss}")
        wandb.log({'trainLoss': train_loss ,'epoch': epoch})

        eval_precision, eval_loss = eval(args, epoch, val_loader, model, loss_function)

        if eval_precision > best_precision:
            best_precision, best_epoch, record_loss = eval_precision, epoch, eval_loss
            if not os.path.exists(args.model_path):
                os.mkdir(args.model_path)
            torch.save(model.state_dict(), 
                            '{}{}.pth'.format(args.model_path, args.model_name))

    wandb.save('{}{}.pth'.format(args.model_path, args.model_name))
    print(f"{datetime.now()}****** The best epoch is {best_epoch}, and the precision is {best_precision}, the loss is {record_loss}")



def scatter2int(threshold, predictions):
    res = []
    for prediction in predictions:
        if prediction >= threshold:
            res.append(1)
        else:
            res.append(0)
    return res

def eval(args, epoch, test_loader, model, loss_function):

    precision_list = []
    eval_loss = 0

    model.eval()

    with torch.no_grad(): 
        for i,data in enumerate(test_loader):
            
            user, item, label = data
            
            user = user.cuda()
            item = item.cuda()
            label = label.cuda()
            
            out = model(user, item, label)
            predictions = scatter2int(args.threshold, out['prediction'].squeeze(1))
            labels = out['labels']
            precision_list.append(precision(predictions, labels))
        
            
            loss = loss_function(out["prediction"].squeeze(1), out["labels"].float())
            eval_loss = eval_loss + loss

    final_precision, final_loss = np.float(np.mean(precision_list)), eval_loss
    print(f"{datetime.now()} Evalueate precision is {final_precision}, loss is {final_loss}")
    wandb.log({'valPrecision':final_precision, 'valLoss': final_loss ,'epoch': epoch})

    return final_precision, final_loss

def numpy_around(num):
    return np.around(num, 2)

def test(args, test_loader, model):
    model.cuda()
    model.load_state_dict(torch.load(args.model_path + args.model_name + '.pth'))

    res = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            user, item = data
            user = user.cuda()
            item = item.cuda()
            out = model(user, item, np.zeros(len(user)))
            prediction = out['prediction'].squeeze(1).cpu().numpy().tolist()
            prediction = [round(p, 2) for p in prediction]
            prediction = scatter2int(args.threshold, prediction)
            res.extend(prediction)

    res_df = pd.DataFrame()
    res_df['ratings'] = res
    res_df.to_csv(args.result_path)
    print('End! Successful!')


def valRoc(args, test_loader, model):
    model.cuda()
    model.load_state_dict(torch.load(args.model_path + args.model_name + '.pth'))

    res = []
    target = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            user, item, label = data
            user = user.cuda()
            item = item.cuda()
            out = model(user, item, np.zeros(len(user)))
            prediction = out['prediction'].squeeze(1).cpu().numpy().tolist()
            prediction = [round(p, 2) for p in prediction]
            label = label.numpy().tolist()
            res.extend(prediction)
            target.extend(label)

    res_df = pd.DataFrame()
    res_df['ratings'] = res
    res_df['labels'] = target
    res_df.to_csv(args.result_path, index = False)
    print('End! Successful!')
