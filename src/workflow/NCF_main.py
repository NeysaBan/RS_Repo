import warnings
warnings.filterwarnings('ignore')


import sys
sys.path.append(r"/data/RS/RS_Repo/")

from src.config.NCF_config import getConfig, recordWandb
from src.utils.NCF.data_utils import getData
from src.utils.NCF.model_utils import getModel
from src.utils.NCF.trainer import train, test, valRoc


if __name__=='__main__':
    parser = getConfig()
    args = parser.parse_args()

    recordWandb(args)

    user_num, item_num, train_loader, val_loader, test_loader = getData(args)
    model, loss, optimizer = getModel(args, user_num, item_num)

    train(args, train_loader, val_loader, model, loss, optimizer)

    ## if u just want to get result, use this line
    test(args, test_loader, model)

    ## if u want to find the best threshold，use this line of code, and try trick/roc.py
    # valRoc(args, val_loader, model)
