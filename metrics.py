from utils import *
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, precision_recall_curve, auc, f1_score, \
    accuracy_score, recall_score, precision_score


def show_metrics(Y_pred):
    lpi = np.loadtxt('adj_total.txt')
    y_true = lpi.flatten()
    y_pred = Y_pred.flatten()
    fpr, tpr, rocth = roc_curve(y_true, y_pred)
    auroc = auc(fpr, tpr)

    precision, recall, prth = precision_recall_curve(y_true, y_pred)
    aupr = auc(recall, precision)
    y_pred_convert = metrix_convert(y_pred)
    f1 = f1_score(y_true, y_pred_convert.astype(np.int64))
    acc = accuracy_score(y_true, y_pred_convert.astype(np.int64))
    pre = average_precision_score(y_true, y_pred.astype(np.int64))
    print('AUROC= %.4f | AUPR= %.4f | ACC= %.4f | F1_score= %.4f | precision= %.4f' % (auroc, aupr, acc, f1, pre))

    return auroc, aupr, acc, f1, pre
