
def precision(preditions, labels):
    TPFP = 0
    TP = 0
    for prediction, label in zip(preditions, labels):
        if int(label.item()) == 1:
            TPFP = TPFP + 1
            if prediction == 1:
                TP = TP + 1
    return TP / TPFP
        
