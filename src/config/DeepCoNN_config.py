import argparse
import wandb

def getParser():
    parser = argparse.ArgumentParser()

    ## device
    parser.add_argument("--device", type=str,default="cuda:1",  help="gpu card ID")

    ## path
    data_path = "./data/Digital_Train_Test/"
    process_path = data_path + "process/"
    parser.add_argument("--data_path", type=str, default=data_path, help="path of dataset")
    parser.add_argument("--train_file", type=str, default=data_path + "Digital_Train_uid_iid_rev_rt.csv", help="path of train dataset")
    parser.add_argument("--test_file", type=str, default=data_path + "Digital_Test_uid_iid.csv", help="path of train dataset")
    parser.add_argument("--w2v_file", type=str, default=process_path + "glove.6B.100d.txt", help="the file path of word2vector csv")
    parser.add_argument("--stopwords_file", type=str, default=process_path + "stopwords.txt", help="the file path of sdstopwordst")
    parser.add_argument("--punctuations_file", type=str, default=process_path + "punctuations.txt", help="the file path of punctuations")
    parser.add_argument("--csv_path", type=str, default=data_path + "csv/", help="the path of training, evaluating and testing csv")
    parser.add_argument("--model_file", type=str, default="./temp/DeepCoNN/DeepCoNN.pth", help="path of model")
    parser.add_argument("--result_file", type=str, default="./temp/DeepCoNN/res.csv", help="path of result")

    ## data settings
    parser.add_argument("--batch_size", type=int, default=128, help="batch size for training")
    parser.add_argument("--PAD_WORD", type=str, default='<UNK>', help="unknown words")
    parser.add_argument("--review_count", type=int,default=10,  help="The max num of reviews that the program can handle")
    parser.add_argument("--review_length", type=int,default=60,  help="The max length of a review that the program can handle")
    parser.add_argument("--lowest_review_count", type=int,default=0,  help="users/items whose reviews below this number will be deleted")

    ## model settings
    parser.add_argument("--model_name",type = str,default = "DeepCoNN",help = "the model name")
    parser.add_argument("--epochs", type=int,default=64,  help="training epoches")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--lr_decay", type=float,default=0.99,  help="the decaying degree of learning rate")
    parser.add_argument("--l2_regularization", type=float,default=1e-6,  help="dropout rate")
    parser.add_argument("--dropout", type=float,default=0.5,  help="dropout rate")
    parser.add_argument("--embedding_dim", type=int,default=16, help="predictive factors numbers in the model")
    parser.add_argument("--kernel_count", type=int,default=100,  help="amount of cnn kernels")
    parser.add_argument("--kernel_size", type=int,default=3,  help="size of cnn kernel")
    parser.add_argument("--cnn_out_dim", type=int,default=50,  help="output dimension of cnn layer")
    parser.add_argument("--dilation", type=int,default=2,  help="dilation of cnn layer")


    return parser


def recordWandb(args):
    wandb.init(
        project = "DeepCoNN", entity="yuej",
        name= args.model_name  + "-e" + str(args.epochs) + "-lr" + str(args.lr),
        config=args
    )