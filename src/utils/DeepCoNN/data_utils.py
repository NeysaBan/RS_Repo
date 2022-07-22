import os
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import WordPunctTokenizer
from sklearn.model_selection import train_test_split

class trainSet(Dataset):
    def __init__(self, train_df, word_dict, args, reviews_by_user, reviews_by_item):
        self.word_dict = word_dict
        self.args = args
        self.PAD_WORD_idx = self.word_dict[args.PAD_WORD]
        self.review_length = args.review_length
        self.review_count = args.review_count
        self.lowest_r_count = args.lowest_review_count  
        
        self.df = train_df
        self.df.columns = ['user_id', 'item_id', 'reviews', 'rating']
        self.sparse_user_idx = set()
        self.sparse_item_idx = set()
        self.sparse_idx = set()
        user_reviews = self.getReviews(self.df, reviews_by_lead=reviews_by_user)
        item_reviews = self.getReviews(self.df, 'item_id', 'user_id', reviews_by_lead=reviews_by_item)
        rating = torch.Tensor(self.df['rating'].tolist()).view(-1, 1)
        
        
        self.user_reviews = user_reviews[[idx for idx in range(user_reviews.shape[0]) if idx not in self.sparse_idx]]
        self.item_reviews = item_reviews[[idx for idx in range(item_reviews.shape[0]) if idx not in self.sparse_idx]]
        self.rating = rating[[idx for idx in range(rating.shape[0]) if idx not in self.sparse_idx]]

    
    def adjust_review_list(self, reviews, r_length, r_count):
        ## regularing the amount of reviews
        reviews = reviews[:r_count] + [[self.PAD_WORD_idx] * r_length] * (r_count - len(reviews))  
        ## regularing the length of a review
        reviews = [r[:r_length] + [0] * (r_length - len(r)) for r in reviews]  
        return reviews


    def getReviews(self, df, lead = 'user_id', costar = 'item_id', reviews_by_lead = None):
        lead_reviews = []
        for idx, (lead_id, costar_id) in enumerate(zip(df[lead], df[costar])):
            df_data = reviews_by_lead[lead_id]
            reviews = df_data['reviews'].tolist()
            if len(reviews) < self.lowest_r_count:
                self.sparse_idx.add(idx) 
            reviews = self.adjust_review_list(reviews, self.review_length, self.review_count)
            lead_reviews.append(reviews)
        
        return torch.LongTensor(lead_reviews)
    
    def __getitem__(self, idx):
        return self.user_reviews[idx], self.item_reviews[idx], self.rating[idx], self.df.loc[idx]['user_id'], self.df.loc[idx]['item_id']


    def __len__(self):
        return self.rating.shape[0]

class valtestSet(Dataset):
    def __init__(self, csv_file, latent, args):

        if "test" in csv_file:
            df = pd.read_csv(csv_file, header=None, names=['user_id', 'item_id'])
            self.val = False
        else:
            df = pd.read_csv(csv_file, header=None, names=['user_id', 'item_id', 'reviews', 'rating'])
            self.val = True

        user_latent = np.empty((df.shape[0], ), dtype = torch.Tensor)
        item_latent = np.empty((df.shape[0], ), dtype = torch.Tensor)

        for idx in df.index:

            user_id = df.loc[idx]['user_id']
            item_id = df.loc[idx]['item_id']

            if user_id not in latent['user'].keys():
                while(user_id not in latent['user'].keys()):
                    user_id = random.randint(1,5541)
            user_latent[idx] = latent['user'][user_id]
            
            if item_id not in latent['item'].keys():
                while(item_id not in latent['item'].keys()):
                    item_id = random.randint(1,3568)
            item_latent[idx] = latent['item'][item_id]

        df.insert(loc=len(df.columns), column='user_latent', value=user_latent) 
        df.insert(loc=len(df.columns), column='item_latent', value=item_latent)
        self.df = df

    def __getitem__(self, idx):
        if self.val:
            return self.df.loc[idx]['user_latent'], self.df.loc[idx]['item_latent'], self.df.loc[idx]['rating']
        else:
            return self.df.loc[idx]['user_latent'], self.df.loc[idx]['item_latent']
    
    def __len__(self):
        return self.df.shape[0]

def load_embedding(word2vec_file):
    with open(word2vec_file, encoding='utf-8') as f:
        word_emb = list()
        word_dict = dict()
        word_emb.append([0])
        word_dict['<UNK>'] = 0
        for line in f.readlines():
            tokens = line.split(' ') 
            word_emb.append([float(i) for i in tokens[1:]])
            if len(word_emb[len(word_emb) - 1]) < 100:
                print(line)
                print(tokens)
            word_dict[tokens[0]] = len(word_dict)
        word_emb[0] = [0] * len(word_emb[1])
    return word_emb, word_dict

def getData(args, word_dict):

    with open(args.stopwords_file) as f:
        stopwords = set(f.read().splitlines())
    with open(args.punctuations_file) as f:  # useless punctuations
        punctuations = set(f.read().splitlines())
    
    def clean_review(review):
        if isinstance(review, float):
            print(review)
        review = review.lower()
        for p in punctuations:
            review = review.replace(p, ' ') 
        review = WordPunctTokenizer().tokenize(review)
        review = [word for word in review if word not in stopwords]
        return ' '.join(review)

    def reviews2id(reviews):
        if not isinstance(reviews, str):
            return [];
        words = []
        for word in reviews.split():
            if word in word_dict:
                words.append(word_dict[word])
            else:
                words.append(word_dict[args.PAD_WORD])
        return words

    raw_train_data = pd.read_csv(args.train_file)
    raw_train_data.columns = ['user_id', 'item_id', 'reviews', 'rating']
    raw_train_data['reviews'] = raw_train_data['reviews'].fillna("")
    raw_train_data['reviews'] = raw_train_data['reviews'].apply(clean_review)
    raw_train_data['reviews'] = raw_train_data['reviews'].apply(reviews2id)
    
    ## Compared with saving csv file first， this way keeps reviews to be list.
    ## Otherwise, reviews will be str.
    reviews_by_user = dict(list(raw_train_data[['item_id', 'reviews']].groupby(raw_train_data['user_id'])))
    reviews_by_item = dict(list(raw_train_data[['user_id', 'reviews']].groupby(raw_train_data['item_id'])))
    
    train_df, val_df = train_test_split(raw_train_data, test_size = 0.1)
    test_df = pd.read_csv(args.test_file)

    train_df.to_csv(os.path.join(args.csv_path, 'train.csv'), index=False, header=False)
    val_df.to_csv(os.path.join(args.csv_path, 'valid.csv'), index=False, header=False)
    test_df.to_csv(os.path.join(args.csv_path, 'test.csv'), index=False, header=False)

    return  reviews_by_user, reviews_by_item                               
    


