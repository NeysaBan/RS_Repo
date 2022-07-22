import os
import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

import sys
sys.path.append(r"/data/RS/RS_Repo/")

from src.config.DeepCoNN_config import getParser, recordWandb
from src.model.DeepCoNN_model import DeepCoNN
from src.utils.DeepCoNN.trainer import train, test
from src.utils.DeepCoNN.data_utils import load_embedding




if __name__ == '__main__':

    parser = getParser()

    args = parser.parse_args()
    recordWandb(args)

    word_emb, word_dict = load_embedding(args.w2v_file)

    ## create model
    model = DeepCoNN(args, word_emb).to(args.device)

    ## train and evaluate
    best_latent = train(model, args, word_dict)

    ## test
    test_model = DeepCoNN(args, word_emb).to(args.device)
    test_model.load_state_dict(torch.load(args.model_file))
    test(test_model, args, best_latent)
