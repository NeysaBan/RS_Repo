import time 
import torch
import numpy as np
import os

import sys
sys.path.append(r"/data/RS/RS_Repo/")

from src.utils.metrics import NDCG, Hit

def date(f='%Y-%m-%d %H:%M:%S'):
    return time.strftime(f, time.localtime())

def train(model, args, train_loader, valid_loader, test_loader):
    bce_criterion = torch.nn.BCEWithLogitsLoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, betas=(0.9, 0.98))

    epoch_start_idx = 1

    T = 0.0
    bestNDCG = 0
    bestEpoch = -1
    t0 = time.time()
    for epoch in range(epoch_start_idx, args.epochs):
        total_loss = 0.0
        total_sample= 0.0
        for i,data in enumerate(train_loader):
            model.train()
            u, seq, pos, neg = data
            u, seq, pos, neg = u.long().to(args.device), seq.long().to(args.device), pos.long().to(args.device), neg.long().to(args.device)
            pos_logits, neg_logits = model(u, seq, pos, neg)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device = args.device), torch.zeros(neg_logits.shape, device=args.device)
            indices = np.where(pos.cpu() != 0)
            # loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            adam_optimizer.zero_grad()
            loss.backward()
            adam_optimizer.step()
            total_loss += loss.item()
            total_sample += len(u)
            # f.write("iteration {}: {}".format(i, loss.item())+'\n')

        print(f"{date()}#### Epoch {epoch:3d}; train loss {total_loss / total_sample:.6f}; ")

        if (epoch + 1) % args.evaluate_epochs == 0 :
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            # f.write('*************I\'m Evaluating in this epoch '.format(epoch)+'*****************\n')
            test_ndcg, test_hit = evaluate(model, test_loader, args)
            val_ndcg, val_hit = evaluate(model, valid_loader, args)
            if bestNDCG < test_ndcg:
                bestNDCG = test_ndcg
                bestEpoch = epoch
                torch.save(model.state_dict(), args.model_file)
            print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)'
                    % (epoch, T, val_ndcg, val_hit, test_ndcg,test_hit))

            # f.write('The value of t_valid is'+str(t_valid) + ', and the value of t_test is' + str(t_test) + '\n')
            # f.flush()
    
        # if epoch == args.evaluate_epochs:
        #     folder = args.dataset + '_' + args.train_dir
        #     fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.dev={}.pth'
        #     fname = fname.format(args.evaluate_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen, args.device)
        #     torch.save(model.state_dict(), os.path.join(folder, fname))
    print(f"The best epoch is {bestEpoch}, and the best test NDCG is {bestNDCG} ")
    print('End! Success!')

def evaluate(model, dataloader, args):

    num_user = 0.0
    ndcg = 0.0
    hit = 0.0

    for data in dataloader:
        users, seq, item_idx = data
        users, seq, item_idx = users.long().to(args.device), seq.long().to(args.device),item_idx.long().to(args.device)
        predictions = -model.predict(users, seq, item_idx)
        num_user += len(users)

        ndcg += NDCG(predictions, args.threshold)
        hit += Hit(predictions, args.threshold)

    return ndcg / num_user, hit / num_user