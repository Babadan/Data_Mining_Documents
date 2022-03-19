import pandas as pd
import numpy as np


def addNoiseFeatures(tf_idf_data,noiseFeaturesNum = 200):
    if noiseFeaturesNum == 0 :
        return tf_idf_data
    random_state = np.random.RandomState(0)
    n_samples, n_features = tf_idf_data.shape
    tf_idf_data = np.c_[tf_idf_data, random_state.randn(n_samples, noiseFeaturesNum)]
    return tf_idf_data



def matrix(x,y,initial):
    return [[initial for i in range(x)] for j in range(y)]


def loadTrainData(fileName, recordNum='none'):
    df = pd.read_csv(fileName, sep='\t')
    # Coverting String Categories to Numbers
    categories = ["Politics", "Film", "Football", "Business", "Technology"]
    df.loc[df["Category"] == 'Politics', "Category"] = 0
    df.loc[df["Category"] == 'Film', "Category"] = 1
    df.loc[df["Category"] == 'Football', "Category"] = 2
    df.loc[df["Category"] == 'Business', "Category"] = 3
    df.loc[df["Category"] == 'Technology', "Category"] = 4

    if(recordNum=='none'):
        categoryList = np.array(df["Category"], dtype=int)
        contentList = np.array(df["Content"])
    else:
        contentList = np.array(df["Content"].iloc[1:recordNum])
        categoryList = np.array(df["Category"].iloc[1:recordNum], dtype=int)

    return contentList, categoryList

def loadTestData(fileName, recordNum='none'):
    df = pd.read_csv(fileName, sep='\t')
    if (recordNum == 'none'):
        contentList = np.array(df["Content"])
        idList = np.array(df["Id"])
    else:
        contentList = np.array(df["Content"].iloc[1:recordNum])
        idList = np.array(df["Id"].iloc[1:recordNum])

    return contentList,idList
