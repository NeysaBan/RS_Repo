import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split


class NCFDataset(Dataset):
    def __init__(self, data, labels = None, istrain = 1):
        
        self.data = data
        self.istrain = istrain
        if  istrain == 1:
            self.labels = labels.flatten().astype(float) ## (batch_size, )
    

    def __getitem__(self, index):
        
        user_id = self.data[index, 0]
        item_id = self.data[index, 1]
        if self.istrain == 1:
            label = self.labels[index]
            return user_id, item_id, label
        else:
            return user_id, item_id
    

    def __len__(self):
        return len(self.data)

def getData(args):
    train_uv = pd.read_csv(args.data_path + args.train_csv, usecols = [0, 1])
    user_num = len(train_uv['user_id'].unique())
    item_num = len(train_uv['click_article_id'].unique())

    ## split train csv to train and val, and get dataloaders
    train_data = train_uv.values 
    train_labels = pd.read_csv(args.data_path + args.train_csv, usecols = [2]).values
    trn_x, val_x,trn_y, val_y = train_test_split(train_data, train_labels, test_size = 0.1) ## train:val = 9:1 
    train_dataset = NCFDataset(data = trn_x, labels = trn_y, istrain = 1)
    val_dataset = NCFDataset(data = val_x, labels = val_y, istrain = 1)
    train_loader = DataLoader(train_dataset, batch_size =args.batch_size, shuffle = True, drop_last = False, num_workers = 0)
    val_loader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle = True, drop_last = False, num_workers = 0)

    ## get test dataloader without special operations
    test_data = pd.read_csv(args.data_path + args.test_csv, usecols = [0, 1]).values
    torch_test = NCFDataset(data = test_data, labels = None, istrain = 0)
    test_loader = DataLoader(torch_test, batch_size = args.batch_size, shuffle = False, drop_last = False, num_workers = 0)

    return user_num, item_num, train_loader, val_loader, test_loader