# -*- coding: utf-8 -*-
__author__ = 'alexjzy'
import numpy as np

np.random.seed(42)
import os

# To plot pretty figures
import matplotlib
import matplotlib.pyplot as plt

# Sklearn part
from sklearn.svm import LinearSVC
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix

# Pandas part
import pandas as pd

# data = pd.read_csv("./dataset/train")
path = os.getcwd()

train_path = "/train.csv"
test_path = "/test.csv"
dataset_path = "/dataset"

data_train = pd.read_csv(path + dataset_path + train_path)

# convert the labels to encode numbers as a new column
def convertClassToEncodeNumber(df, benchmarkCol, newColName):
    labelEncoder = preprocessing.LabelEncoder()  # initial encoder
    labelEncoder.fit(df.loc[:, benchmarkCol])
    encodedData = labelEncoder.transform(df.loc[:, benchmarkCol])
    df[newColName] = encodedData
    return df

def standardScaler(df):
    scaler = preprocessing.StandardScaler().fit(df)
    return scaler

data_train = convertClassToEncodeNumber(data_train, "class", "Y")

# several sets of data
# 1:22 ''
# 22:43 40
# 43:64 60
# 64:85 80
# 85:106 100
# 106:127 120
# 127:148 140
trainX = data_train.iloc[:, 1:22]
trainY = data_train.iloc[:, -1]

trainX_scale = standardScaler(trainX).transform(trainX)

# SVM linear, multi_class = crammer_singer
lin_clf_crammer = LinearSVC(random_state=42, multi_class='crammer_singer')
lin_clf_crammer.fit(trainX_scale, trainY)

# SVM linear, multi_class = ovr
lin_clf = LinearSVC(random_state=42, multi_class='ovr')
lin_clf.fit(trainX_scale, trainY)

trainY_pred = lin_clf.predict(trainX_scale)
accuracy = accuracy_score(trainY, trainY_pred)
print(accuracy)
confusion_matrix(trainY, trainY_pred)
