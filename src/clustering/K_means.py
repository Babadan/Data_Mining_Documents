import csv

import nltk
from nltk.cluster.kmeans import KMeansClusterer
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def kMeans():
    # Data preparation
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

    # counter
    count_vect = CountVectorizer(stop_words=stop,min_df=0.03,max_df=0.95)
    X = count_vect.fit_transform(X_train)
    # tf
    # count_vect = TfidfVectorizer(stop_words=stop,max_df=0.95,analyzer = 'word')
    # X_train = count_vect.fit_transform(X_train)
    #tf-idf
    tfidf_transformer = TfidfTransformer()
    X = tfidf_transformer.fit_transform(X)
    #dimensionality reduction
    # svd_model = TruncatedSVD(n_components=20,random_state=42)
    # X_train = svd_model.fit_transform(X_train)

    NUM_CLUSTERS = 5
    data = X.toarray()
    data = np.delete(data,6782,0)
    Y = np.delete(Y,6782,0)

    #kMeans
    kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=20)
    assigned_clusters = kclusterer.cluster(data, assign_clusters=True)

    #evaluation
    kMeansOutput=[['Business','Films','Football','Politics','Technology']]

    evaluationCluster1 = [0, 0, 0, 0, 0]
    business = 0
    film = 0
    football = 0
    politics = 0
    technology = 0
    for i in range(0,data.shape[0]):
        if assigned_clusters[i]==0 :
            if Y[i] == 0:
                business = business + 1
            if Y[i] == 1:
                film = film + 1
            if Y[i] == 2:
                football = football + 1
            if Y[i] == 3:
                politics = politics + 1
            if Y[i] == 4:
                technology = technology + 1
    evaluationCluster1[0] = float(business)/float((sum(Y == 0)))
    evaluationCluster1[1] = float(film) / float((sum(Y == 1)))
    evaluationCluster1[2] = float(football) / float((sum(Y == 2)))
    evaluationCluster1[3] = float(politics) / float((sum(Y == 3)))
    evaluationCluster1[4] = float(technology) / float((sum(Y == 4)))
    kMeansOutput.append(evaluationCluster1)
    print kMeansOutput

    evaluationCluster2 = [0, 0, 0, 0, 0]
    business = 0
    film = 0
    football = 0
    politics = 0
    technology = 0
    for i in range(0,data.shape[0]):
        if assigned_clusters[i]==1 :
            if Y[i] == 0:
                business = business + 1
            if Y[i] == 1:
                film = film + 1
            if Y[i] == 2:
                football = football + 1
            if Y[i] == 3:
                politics = politics + 1
            if Y[i] == 4:
                technology = technology + 1
    evaluationCluster2[0] = float(business)/float((sum(Y == 0)))
    evaluationCluster2[1] = float(film) / float((sum(Y == 1)))
    evaluationCluster2[2] = float(football) / float((sum(Y == 2)))
    evaluationCluster2[3] = float(politics) / float((sum(Y == 3)))
    evaluationCluster2[4] = float(technology) / float((sum(Y == 4)))
    kMeansOutput.append(evaluationCluster2)
    print kMeansOutput

    evaluationCluster3 = [0, 0, 0, 0, 0]
    business = 0
    film = 0
    football = 0
    politics = 0
    technology = 0
    for i in range(0,data.shape[0]):
        if assigned_clusters[i]==2 :
            if Y[i] == 0:
                business = business + 1
            if Y[i] == 1:
                film = film + 1
            if Y[i] == 2:
                football = football + 1
            if Y[i] == 3:
                politics = politics + 1
            if Y[i] == 4:
                technology = technology + 1
    evaluationCluster3[0] = float(business)/float((sum(Y == 0)))
    evaluationCluster3[1] = float(film) / float((sum(Y == 1)))
    evaluationCluster3[2] = float(football) / float((sum(Y == 2)))
    evaluationCluster3[3] = float(politics) / float((sum(Y == 3)))
    evaluationCluster3[4] = float(technology) / float((sum(Y == 4)))
    kMeansOutput.append(evaluationCluster3)
    print kMeansOutput

    evaluationCluster4 = [0, 0, 0, 0, 0]
    business = 0
    film = 0
    football = 0
    politics = 0
    technology = 0
    for i in range(0,data.shape[0]):
        if assigned_clusters[i]==3 :
            if Y[i] == 0:
                business = business + 1
            if Y[i] == 1:
                film = film + 1
            if Y[i] == 2:
                football = football + 1
            if Y[i] == 3:
                politics = politics + 1
            if Y[i] == 4:
                technology = technology + 1
    evaluationCluster4[0] = float(business)/float((sum(Y == 0)))
    evaluationCluster4[1] = float(film) / float((sum(Y == 1)))
    evaluationCluster4[2] = float(football) / float((sum(Y == 2)))
    evaluationCluster4[3] = float(politics) / float((sum(Y == 3)))
    evaluationCluster4[4] = float(technology) / float((sum(Y == 4)))
    kMeansOutput.append(evaluationCluster4)
    print kMeansOutput

    evaluationCluster5 = [0, 0, 0, 0, 0]
    business = 0
    film = 0
    football = 0
    politics = 0
    technology = 0
    for i in range(0,data.shape[0]):
        if assigned_clusters[i]==4 :
            if Y[i] == 0:
                business = business + 1
            if Y[i] == 1:
                film = film + 1
            if Y[i] == 2:
                football = football + 1
            if Y[i] == 3:
                politics = politics + 1
            if Y[i] == 4:
                technology = technology + 1
    evaluationCluster5[0] = float(business)/float((sum(Y == 0)))
    evaluationCluster5[1] = float(film) / float((sum(Y == 1)))
    evaluationCluster5[2] = float(football) / float((sum(Y == 2)))
    evaluationCluster5[3] = float(politics) / float((sum(Y == 3)))
    evaluationCluster5[4] = float(technology) / float((sum(Y == 4)))
    kMeansOutput.append(evaluationCluster5)
    print kMeansOutput

    # write the results in csv file
    with open('../outputFiles/clustering_KMeans.csv', 'wb') as fp:
        a = csv.writer(fp, delimiter=',')
        a.writerows(kMeansOutput)

