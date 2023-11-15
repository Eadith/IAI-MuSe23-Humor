from eval import calc_auc


if __name__ == '__main__':

    preds = []
    labels = []

    score = calc_auc(preds, labels)