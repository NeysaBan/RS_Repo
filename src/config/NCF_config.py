import argparse
import wandb

def getConfig():
    parser = argparse.ArgumentParser()

    ## device
    parser.add_argument("--gpu", type=str,default="1",  help="gpu card ID")

    ## path
    parser.add_argument("--data_path", type=str, default="./data/Tianchi_Train_Test/", help="path of dataset")
    parser.add_argument("--train_csv", type=str, default="Tianchi_Train_uid_iid_rt.csv", help="path of training csv")
    parser.add_argument("--test_csv", type=str, default="Tianchi_Test_uid_iid.csv", help="path of testing csv")
    parser.add_argument("--model_path", type=str, default="./temp/NCF/", help="path of model")
    parser.add_argument("--result_path", type=str, default="./temp/NCF/res.csv", help="path of result")

    ## data settings
    parser.add_argument("--batch_size", type=int, default=128, help="batch size for training")

    ## model settings
    parser.add_argument("--model_name",type = str,default = "NeuMF-Pre",help = "the training model")  ## GMF or MLP or NeuMF-Pre 
    parser.add_argument("--epochs", type=int,default=20,  help="training epoches")
    parser.add_argument("--threshold", type=float,default=0.50,  help="Determine the threshold of positive and negative samples")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--dropout", type=float,default=0.3,  help="dropout rate")
    parser.add_argument("--embedding_dim", type=int,default=16, help="predictive factors numbers in the model")
    parser.add_argument("--mlp_num_layers", type=int, default=3, help="net layers of MLP")


    return parser


def recordWandb(args):
    wandb.init(
        project = "NCF", entity="yuej",
        name= args.model_name  + "-e" + str(args.epochs) + "-lr" + str(args.lr),
        config=args
    )