import sys
import os
import torch
from torch import nn, optim

from src.model.NCF_model import *

def getModel(args, user_num, item_num):

    if args.model_name == 'NeuMF-pre':
        assert os.path.exists(args.model_path + args.GMF_model_name), 'lack of GMF model'
        assert os.path.exists(args.model_path + args.MLP_model_path), 'lack of MLP model'
        GMF_model = NCF(user_num, item_num, args.embedding_dim, args.dropout, model_name ='GMF')
        GMF_model.load_state_dict(torch.load(args.GMF_model_path))
        MLP_model = NCF(user_num, item_num, args.embedding_dim, args.dropout, model_name ='MLP')
        MLP_model.load_state_dict(torch.load(args.MLP_model_path))
    else:
        GMF_model = None
        MLP_model = None

    model = NCF(user_num, item_num, args.embedding_dim, args.dropout, args.model_name, args.mlp_num_layers, GMF_model, MLP_model)
    model.cuda()

    loss = nn.MSELoss()

    if args.model_name == 'NeuMF-pre':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    return model, loss, optimizer