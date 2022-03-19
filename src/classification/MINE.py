import csv
from audioop import avg
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn import preprocessing, svm
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import MultinomialNB

def mine():
    # Data preparation

    dataTrainPath = '../dataset/train_set.csv'
    dataTestPath = '../dataset/test_set.csv'
    df = pd.read_csv(dataTrainPath, sep='\t')
    df1 = pd.read_csv(dataTestPath, sep='\t')


    X_test = df1['Content']
    X_test_test = df1['Title']
    X_train = df['Content']
    X_train_train = df['Title']
    Y_train = df['Category']
    le = preprocessing.LabelEncoder()
    le.fit(df["Category"])
    Y = le.transform(df["Category"])

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
    # count_vect = CountVectorizer(stop_words=stop,min_df=0.03,max_df=0.95,strip_accents='unicode',sublinear_tf=True,use_idf=True,analyzer='word')
    # X_train_vector = count_vect.fit_transform(X_train)
    # X_train_train_vector = count_vect.fit_transform(X_train_train)
    # X_test_vector = count_vect.transform(X_test)
    # X_test_test_vector = count_vect.transform(X_test_test)

    # tf
    count_vect = TfidfVectorizer(stop_words=stop,min_df=0.03,max_df=0.95,strip_accents='unicode',sublinear_tf=True,use_idf=True,analyzer='word')
    #tf-idf
    tfidf_transformer = TfidfTransformer()
    #dimensionality reduction
    svd_model = TruncatedSVD(n_components=50,random_state=42)
    #classifier
    clfNB = MultinomialNB()
    clfSVM = svm.SVC(kernel='rbf', C = 2.0, gamma='auto')

    X_train_vector = count_vect.fit_transform(X_train)
    X_train_transform = tfidf_transformer.fit_transform(X_train_vector)
    X_train_SVM = svd_model.fit_transform(X_train_transform)

    X_test_vector = count_vect.transform(X_test)
    X_test_transform = tfidf_transformer.transform(X_test_vector)
    X_test_SVM = svd_model.transform(X_test_transform)

    X_train_train_vector = count_vect.fit_transform(X_train_train)
    X_train_train_transform = tfidf_transformer.fit_transform(X_train_train_vector)

    X_test_test_vector = count_vect.transform(X_test_test)
    X_test_test_transform = tfidf_transformer.transform(X_test_test_vector)


    # fit && predict
    #NB content
    clfNB.fit(X_train_vector,Y_train)
    test_hat_NB_con = clfNB.predict(X_test_vector)
    #NB title
    clfNB.fit(X_train_train_vector,Y_train)
    test_hat_NB_tit = clfNB.predict(X_test_test_vector)
    #SVM content
    clfSVM.fit(X_train_SVM,Y_train)
    test_hat_SVM_con = clfSVM.predict(X_test_SVM)
    #SVM title
    clfSVM.fit(X_train_train_transform, Y_train)
    test_hat_SVM_tit = clfSVM.predict(X_test_test_transform)

    test_hat = []
    for i in range(0,len(test_hat_NB_con)):
        if test_hat_NB_con[i] == test_hat_SVM_con[i]:
            test_hat.append(test_hat_NB_con[i])
        elif test_hat_SVM_con[i] == test_hat_SVM_tit[i]:
            test_hat.append(test_hat_SVM_con[i])
        elif test_hat_NB_con[i] == test_hat_NB_tit[i]:
            test_hat.append(test_hat_NB_con[i])
        elif  test_hat_SVM_con[i] == test_hat_NB_tit[i] and test_hat_NB_con[i] != test_hat_SVM_tit[i]:
            test_hat.append(test_hat_NB_tit[i])
        elif test_hat_NB_con[i] == test_hat_SVM_tit[i] and test_hat_SVM_con[i] != test_hat_NB_tit[i]:
            test_hat.append(test_hat_SVM_tit[i])
        elif test_hat_NB_con[i] == test_hat_NB_tit[i] and test_hat_SVM_con[i] == test_hat_SVM_tit[i]:
            test_hat.append(test_hat_SVM_tit[i])
        elif test_hat_NB_tit[i] == test_hat_SVM_tit[i]:
            test_hat.append(test_hat_SVM_tit[i])
        else:
            test_hat.append(test_hat_SVM_con[i])

    #write results
    id = df1['Id']
    file_name = r'../outputFiles/testSet_categories.csv'
    newDf = pd.DataFrame({'ID':id,'Predicted Category':test_hat})
    newDf.to_csv(file_name, sep='\t')

    return
