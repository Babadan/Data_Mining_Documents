from collections import OrderedDict
import numpy
import operator
from scipy import interp
import metrics as mt
import pandas as pd
from scipy import spatial
from nltk.corpus import stopwords
from sklearn import preprocessing
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score
from sklearn.decomposition import TruncatedSVD


def cosineSimilarity(vector, vector2, len):
    distance = 0
    distance += spatial.distance.cosine(vector , vector2)
    return distance


def getNeighbors(trainSet,predictionVector, k,categoryList):
    distances = []
    length = len(predictionVector) - 1
    for x in range(len(trainSet)):
        dist = cosineSimilarity(predictionVector,trainSet[x], length)
        distances.append((categoryList[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def getResponse(neighbors,classNum):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)

    probabilityVotes=[0 for i in range(classNum)]
    for votes in sortedVotes:
        probabilityVotes[votes[0]] = votes[1]/float(len(neighbors))

    return sortedVotes[0][0],probabilityVotes



def kNearest_neighbors():

    dataTrainPath = '../dataset/train_set.csv'
    dataTestPath = '../dataset/test_set.csv'
    df = pd.read_csv(dataTrainPath, sep='\t')

    X_train = df['Content']
    Y_train = df['Category']
    le = preprocessing.LabelEncoder()
    le.fit(Y_train)
    Y = le.transform(Y_train)
    # Data Preprocessed
    stop = set(stopwords.words('english'))
    stop.add("said")
    stop.add("say")
    stop.add("will")
    stop.add("new")
    stop.add("also")
    stop.add("one")
    stop.add("now")
    stop.add("still")
    stop.add("time")
    stop.add("may")

    count_vect = CountVectorizer(stop_words=stop, min_df=0.03, max_df=0.95)
    X_train = count_vect.fit_transform(X_train)
    # tf
    # count_vect = TfidfVectorizer(stop_words=stop,max_df=0.95,analyzer = 'word')
    # X_train = count_vect.fit_transform(X_train)
    # tf-idf
    tfidf_transformer = TfidfTransformer()
    X_train = tfidf_transformer.fit_transform(X_train)
    # dimensionality reduction
    svd_model = TruncatedSVD(n_components=200, random_state=42)
    X_train = svd_model.fit_transform(X_train)

    K = 11
    classesCount = 5
    k_folds = 10

    crossValidation = StratifiedKFold(Y, n_folds=k_folds )

    sumF1 = 0.0
    sumRec = 0.0
    sumPr = 0.0
    sumAc = 0.0

    base_fprs_macro = numpy.linspace(0, 1, 101)
    base_fprs_micro = numpy.linspace(0, 1, 101)
    tprs_macro = []
    tprs_micro = []
    auc_macro = []
    auc_micro = []

    for i,(train, test) in enumerate(crossValidation):
        predict = []
        predict_prob =[]
        for Xtrain_test in X_train[test]:
            neighbors = getNeighbors(X_train[train], Xtrain_test,K,Y[train])
            result,probability = getResponse(neighbors,classesCount)
            predict.append(result)
            predict_prob.append(probability)

        f1 = f1_score(Y[test], predict, average="macro")
        sumF1 = sumF1 + f1
        prec = precision_score(Y[test], predict, average="macro")
        sumPr = sumPr + prec
        rec = recall_score(Y[test], predict, average="macro")
        sumRec = sumRec + rec
        ac = accuracy_score(Y[test], predict)
        sumAc = sumAc + ac
        # if metric == "auc":
        y_score = numpy.array(predict_prob)

        rl = mt.rocAuc(Y[test], y_score)

        tpr_macro = interp(base_fprs_macro, rl[0]["macro"], rl[1]["macro"])
        tpr_micro = interp(base_fprs_micro, rl[0]["micro"], rl[1]["micro"])

        tpr_macro[0] = 0.0
        tpr_micro[0] = 0.0
        tprs_macro.append(tpr_macro)
        tprs_micro.append(tpr_micro)
        auc_macro.append(rl[2]["macro"])
        auc_micro.append(rl[2]["micro"])

    # if metric == "auc":
    mt.printAvergedRocCurve(tprs_macro, base_fprs_macro, auc_macro, "Macro", "../RocCurve/" + 'KNN_roc_macro.png')
    mt.printAvergedRocCurve(tprs_micro, base_fprs_micro, auc_micro, "Micro", "../RocCurve/" + 'KNN_roc_micro.png')

    totalScore = []
    totalScore.append(float(sumAc) / k_folds)
    totalScore.append(float(sumPr) / k_folds)
    totalScore.append(float(sumRec) / k_folds)
    totalScore.append(float(sumF1) / k_folds)
    auc = numpy.array(auc_macro)

    mean_auc = auc.mean(axis=0)
    totalScore.append(float(mean_auc))

    return totalScore



