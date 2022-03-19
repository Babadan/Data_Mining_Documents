import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn import preprocessing
from sklearn.cross_validation import StratifiedKFold, cross_val_predict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score
from sklearn.decomposition import TruncatedSVD
from sklearn import svm
from scipy import interp
import utility_fuction as ut
import metrics as mt

def support_vector_machine():
    # Data preparation
    dataTrainPath = '../dataset/train_set.csv'
    dataTestPath = '../dataset/test_set.csv'
    df = pd.read_csv(dataTrainPath, sep='\t')

    X_train = df['Content']
    Y_train = df['Category']
    le = preprocessing.LabelEncoder()
    le.fit(df["Category"])
    Y=le.transform(df["Category"])

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

    # counter
    count_vect = CountVectorizer(stop_words=stop,min_df=0.03,max_df=0.95)
    X_train = count_vect.fit_transform(X_train)
    # tf
    # count_vect = TfidfVectorizer(stop_words=stop,max_df=0.95,analyzer = 'word')
    # X_train = count_vect.fit_transform(X_train)
    #tf-idf
    tfidf_transformer = TfidfTransformer()
    X_train = tfidf_transformer.fit_transform(X_train)
    #dimensionality reduction
    svd_model = TruncatedSVD(n_components=200,random_state=42)
    X_train = svd_model.fit_transform(X_train)
    #classifier
    clf = svm.SVC(kernel='rbf', C = 2.0, gamma='auto',probability=True)


    K_Folds = 10
    crossValidation = StratifiedKFold(Y, n_folds=K_Folds)
    sumF1 = 0.0
    sumRec = 0.0
    sumPr = 0.0
    sumAc = 0.0

    base_fprs_macro = np.linspace(0, 1, 101)
    base_fprs_micro = np.linspace(0, 1, 101)
    tprs_macro = []
    tprs_micro = []
    auc_macro = []
    auc_micro = []

    for i, (train, test) in enumerate(crossValidation):
        clf.fit(X_train[train], Y_train[train])
        test_hat = clf.predict(X_train[test])
        f1 = f1_score(Y_train[test], test_hat, average="macro")
        sumF1 = sumF1 + f1
        prec = precision_score(Y_train[test], test_hat, average="macro")
        sumPr = sumPr + prec
        rec = recall_score(Y_train[test], test_hat, average="macro")
        sumRec = sumRec + rec
        ac = accuracy_score(Y_train[test], test_hat)
        sumAc = sumAc + ac

        y_score = clf.predict_proba(X_train[test])
        rl = mt.rocAuc(Y[test], y_score)

        tpr_macro = interp(base_fprs_macro, rl[0]["macro"], rl[1]["macro"])
        tpr_micro = interp(base_fprs_micro, rl[0]["micro"], rl[1]["micro"])

        tpr_macro[0] = 0.0
        tpr_micro[0] = 0.0
        tprs_macro.append(tpr_macro)
        tprs_micro.append(tpr_micro)
        auc_macro.append(rl[2]["macro"])
        auc_micro.append(rl[2]["micro"])


        print  classification_report(Y_train[test], test_hat)


    mt.printAvergedRocCurve(tprs_macro, base_fprs_macro, auc_macro, "Macro", "../RocCurve/" + 'Svc_roc_macro.png')
    mt.printAvergedRocCurve(tprs_micro, base_fprs_micro, auc_micro, "Micro", "../RocCurve/" + 'Svc_roc_micro.png')

    totalScore = []
    totalScore.append(float(sumAc) / K_Folds)
    totalScore.append(float(sumPr) / K_Folds)
    totalScore.append(float(sumRec) / K_Folds)
    totalScore.append(float(sumF1) / K_Folds)


    auc = np.array(auc_macro)
    mean_auc = auc.mean(axis=0)
    totalScore.append(mean_auc)

    return totalScore
