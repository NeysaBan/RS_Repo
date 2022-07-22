import argparse
import wandb

def getParser():
    parser = argparse.ArgumentParser()

    ## device
    parser.add_argument("--device", type=str,default="cuda:1",  help="gpu card ID")

    ## path
    data_path = "./data/ml-1m/"
    parser.add_argument("--data_path", type=str, default=data_path, help="path of dataset")
    parser.add_argument("--raw_data", type=str, default=data_path + "ml-1m.txt", help="path of raw data")
    parser.add_argument("--model_file", type=str, default="./temp/SASRec/SASRec.pth", help="path of model")

    ## data settings
    parser.add_argument("--batch_size", type=int, default=128, help="batch size for training")
    parser.add_argument("--maxlen", type=int, default=200, help="the max length of items sequence the processor can handle for a user")

    ## model settings
    parser.add_argument("--model_name",type = str,default = "SASRec",help = "the name of model")  ## GMF or MLP or NeuMF-Pre 
    parser.add_argument("--epochs", type=int,default=200,  help="training epoches")
    parser.add_argument("--evaluate_epochs", type=int,default=10,  help="evaluating epoch interval")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--dropout", type=float,default=0.2,  help="dropout rate")
    parser.add_argument("-l2_emb", type=float,default=0.0,  help="When calculating the loss, consider the second norms of the model parameters")
    parser.add_argument("--hidden_units", type=int,default=50, help="hidden dimention")
    parser.add_argument("--num_blocks", type=int, default=2, help="blocks num of attention mechnism")
    parser.add_argument("--num_heads", type=int, default=1, help="heads num of multiple attention mechnism")
    parser.add_argument("--threshold", type=int, default=10, help="when the rank of predicted item higher than threshold, ndcg and hit score will increase")


    return parser


def recordWandb(args):
    wandb.init(
        project = "SASRec", entity="yuej",
        name= args.model_name  + "-e" + str(args.epochs) + "-lr" + str(args.lr),
        config=args
    )