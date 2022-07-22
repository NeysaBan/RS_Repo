import torch

import sys
sys.path.append(r"/data/RS/RS_Repo/")

from src.config.SASRec_config import getParser, recordWandb
from src.utils.SASRec.data_utils import getData
from src.model.SASRec_model import SASRec
from src.utils.SASRec.trainer import train

if __name__ == '__main__':
    parser = getParser()
    args = parser.parse_args()

    recordWandb(args)

    user_num, item_num, train_loader, valid_loader, test_loader = getData(args)

    ## create model
    model = SASRec(user_num, item_num, args).to(args.device)
    ## initialize model
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass # just ignore those failed init layers

    train(model, args, train_loader, valid_loader, test_loader)
    



    