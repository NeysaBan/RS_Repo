import numpy as np

def precision(preditions, labels):
    TPFP = 0
    TP = 0
    for prediction, label in zip(preditions, labels):
        if int(label.item()) == 1:
            TPFP = TPFP + 1
            if prediction == 1:
                TP = TP + 1
    return TP / TPFP
        
## only 1 label item 
def NDCG(predictions, threshold):

    ndcg = 0.0
    for idx in range(predictions.shape[0]):
            prediction = predictions[idx]
            rank = prediction.argsort().argsort()[0].item()

            if rank < threshold:
                ndcg += 1 / np.log2(rank + 2)
    
    return ndcg

def Hit(predictions, threshold):

    hit = 0.0
    
    for idx in range(predictions.shape[0]):
        prediction = predictions[idx]
        rank = prediction.argsort().argsort()[0].item()

        if rank < threshold:
            hit += 1
    
    return hit
