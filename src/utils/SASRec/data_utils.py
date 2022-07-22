import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

class TrainSet(Dataset):
    def __init__(self, user_train, usernum, itemnum, maxlen):
        
        self.users = list(user_train.keys())
        self.seq = np.zeros((usernum, maxlen), dtype=np.int32)
        self.pos = np.zeros((usernum, maxlen), dtype=np.int32)
        self.neg = np.zeros((usernum, maxlen), dtype=np.int32)

        last_item = [value[-1] for key, value in list(user_train.items())]

        u_idx = 0
        for u in self.users:
            ts = set(user_train[u])
            itemEXlast = user_train[u][:-1]
            u = u - 1
            item_nxt = last_item[u]
            idx = maxlen - 1
            for item in reversed(itemEXlast):
                self.seq[u_idx][idx] = item
                self.pos[u_idx][idx] = item_nxt
                ## negative samples
                if item_nxt != 0:
                    self.neg[u_idx][idx] = random_neq(1, itemnum + 1, ts)
                item_nxt = item
                idx -= 1
                if idx == -1:
                    break
            u_idx += 1
    
    def __getitem__(self, index):
        return self.users[index], self.seq[index], self.pos[index], self.neg[index]

    def __len__(self):
        return len(self.users)


class ValTestSet(torch.utils.data.Dataset):
    def __init__(self, dataset, maxlen, isTest = True):
        
        [train, valid, test, usernum, itemnum] = dataset
        if usernum > 10000:
            self.users = np.array(np.random.sample(range(1, usernum + 1), 10000))
        else:
            self.users = np.array(range(1, usernum + 1))

        self.seqs = np.zeros((len(self.users), maxlen), dtype=np.int32)
        self.item_idxs = np.zeros((len(self.users), 101), dtype=np.int32)

        user_index = 0
        for u in self.users:
            if isTest:
                if len(train[u]) < 1 or len(test[u]) < 1 : continue
            else:
                if len(train[u]) < 1 or len(valid[u]) < 1 : continue
            
            #seq = {}
            ## positive samples
            #self.seqs[u] = np.zeros([maxlen], dtype = np.int32)
            idx = maxlen - 1
            if isTest:
                self.seqs[user_index][idx] = valid[u][0]
                idx -= 1
            for i in reversed(train[u]):
                self.seqs[user_index][idx] = i
                idx -= 1
                if idx == -1:break
            #self.seqs.append(seq)

            ## candidates
            #item_idx = {}
            rated = set(train[u])
            rated.add(0)  
            if isTest:
                self.item_idxs[user_index][0] = test[u][0] 
            else:
                self.item_idxs[user_index][0] = valid[u][0] 
            for _ in range(100):
                #if _ == 0: break
                t = np.random.randint(1, itemnum + 1)
                while t in rated: t = np.random.randint(1, itemnum + 1)
                self.item_idxs[user_index][_ + 1] = t
                #item_idx[u].append(t)
            #self.item_idxs.append(item_idx)

            user_index += 1

    def __getitem__(self, index):
        return self.users[index], self.seqs[index], self.item_idxs[index]

    def __len__(self):
        return len(self.users)

def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

# train/val/test data generation
def data_partition(file):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open(file, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    ## User = {user_id: the item browswd by the user}

    for user_id in User:
        nfeedback = len(User[user_id])
        if nfeedback < 3:
            user_train[user_id] = User[user_id]
            user_valid[user_id] = []
            user_test[user_id] = []
        else:
            user_train[user_id] = User[user_id][:-2]
            user_valid[user_id] = []
            user_valid[user_id].append(User[user_id][-2])
            user_test[user_id] = []
            user_test[user_id].append(User[user_id][-1])
    
    ## user_train, user_valid, user_test are all dicts
    return [user_train, user_valid, user_test, usernum, itemnum]

def getData(args):

    dataset= data_partition(args.raw_data)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset

    train = TrainSet(user_train, usernum, itemnum,  maxlen=args.maxlen)
    train_loader = DataLoader(train, batch_size = 128, shuffle = True, drop_last = False, num_workers = 0)
    valid = ValTestSet(dataset, args.maxlen, isTest = False)
    valid_loader = DataLoader(valid, batch_size = 128, shuffle = False, drop_last = False, num_workers = 0)
    test = ValTestSet(dataset, args.maxlen)
    test_loader = DataLoader(test, batch_size = 128, shuffle = False, drop_last = False, num_workers = 0)

    return usernum, itemnum, train_loader, valid_loader, test_loader