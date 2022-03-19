
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from scipy import interp
import numpy as np
import matplotlib.pyplot as plt
import utility_fuction as utf



def ModelEvaluationMetric(str,trueData,predicted):

    res =0
    if str == "accuracy":
        res=accuracy_score(trueData, predicted)

    return res


def precition(true_values,predicted_values):
    precition=precision_score(true_values,predicted_values,average='macro')
    return precition

def recall(true_values,predicted_values):
    recall = recall_score(true_values,predicted_values,average='macro')
    return recall

def f1Score(true_values,predicted_values):
    f1 = f1_score(true_values,predicted_values, average='macro')
    return f1

def confluence_matrix(true_values,predicted_values):
    confluense_matrix = utf.matrix(5,5,0)

    for i,j in true_values,predicted_values:
        confluense_matrix[i][j]+=1

    return confluense_matrix

def rocAuc(trueData,predicted):
    binaryTrueData = label_binarize(trueData, classes=[0, 1, 2, 3, 4])

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(5):
        fpr[i], tpr[i], threshhold = roc_curve(binaryTrueData[:, i], predicted[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(binaryTrueData.ravel(), predicted.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(5)]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(5):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= 5

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return [fpr,tpr,roc_auc]

def printRoc(roc):

    fpr = roc[0]
    tpr = roc[1]
    roc_auc = roc[2]
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.6f})'
                   ''.format(roc_auc["micro"]),
             linewidth=2)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.6f})'
                   ''.format(roc_auc["macro"]),
             linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()

def printAvergedRocCurve(tprs,base_fpr,auc,name,filename):

    auc = np.array(auc)
    mean_auc = auc.mean(axis=0)

    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)

    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std

    plt.figure()
    plt.plot(base_fpr, mean_tprs,
             label=name + '-average ROC curve (area = {0:0.6f})'
                   ''.format(mean_auc),
             linewidth=2)

    plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig(filename)