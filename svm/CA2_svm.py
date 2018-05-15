# -*- coding: utf-8 -*-
__author__ = 'alexjzy'
# codes import part
import numpy as np
import seaborn as sns
np.random.seed(42)
import os

# To plot pretty figures
import matplotlib.pyplot as plt

# Sklearn part
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV

# Pandas part
import pandas as pd

pd.set_option('display.max_columns', 500)

### Load dataset
path = os.getcwd()
train_path = "/train.csv"
test_path = "/test.csv"
dataset_path = "/dataset"

data_train = pd.read_csv(path + dataset_path + train_path)
data_test = pd.read_csv(path + dataset_path + test_path)
data_all = pd.concat([data_train, data_test])

####################################     Functions    ################################################

### SVMs with default parameters
def svmRoughModel(mod, kernel, trainX, trainY, testX, testY):
    mod.fit(trainX, trainY)

    train_pred = model.predict(trainX)
    train_acc = accuracy_score(trainY, train_pred)
    print("%s train accuracy: %f" % (kernel, train_acc))

    test_pred = model.predict(testX)
    test_acc = accuracy_score(testY, test_pred)
    print("%s test accuracy: %f" % (kernel, test_acc))

# convert the labels to encode numbers as a new column
def convertClassToEncodeNumber(df, benchmarkCol, newColName):
    labelEncoder = preprocessing.LabelEncoder()  # initial encoder
    labelEncoder.fit(df.loc[:, benchmarkCol])
    encodedData = labelEncoder.transform(df.loc[:, benchmarkCol])
    df[newColName] = encodedData
    return df

# scaler
def standardScaler(df):
    scaler = preprocessing.StandardScaler().fit(df)
    return scaler

# draw confusion matrix
def drawConfusionMatrix(title, cm):
    labels = sorted(data_train.iloc[:, 0].unique())
    ax= plt.subplot()
    heatmap = sns.heatmap(cm, annot=True, cmap="YlGnBu", ax = ax)
    heatmap.xaxis.set_ticklabels(labels, rotation=45);
    heatmap.yaxis.set_ticklabels(labels, rotation=0);
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(title)

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="b")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="b",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


#################################### Data preparation and baseline model ################################################
### correlation matrics
data_train.corr()
sns.heatmap(data_all.iloc[:, 1:22].corr(), cmap=sns.cubehelix_palette(light=1, as_cmap=True))

### get the train target statistics result
data_train['class'].value_counts()

### variables distribution summary
data_train.iloc[:, 1:22].describe()

### test the rough svm
trainX = data_train.iloc[:, 1:22]
trainY = data_train.iloc[:, 0]

testX = data_test.iloc[:, 1:22]
testY = data_test.iloc[:, 0]

# RBF Baseline model
model = SVC(kernel = 'rbf', random_state = 25, decision_function_shape='ovo')
svmRoughModel(model ,'rbf', trainX, trainY, testX, testY)

############## baseline result ################
#     rbf baseline train accuracy: 1.000000   #
#     rbf baseline test accuracy: 0.154762    #
###############################################


#################################### Proposed model ################################################

# Proposed step by step as follows:
#
# 1. Scaling on train samples, apply the same scaler on test samples.
# 2. Implement  CC  and  γγ  to build soft margins for RBF kerel.
# 3. Split the dataset into train, validation and test dataset.
# 4. Apply grid search with cross validation to confirm  CC  and  γγ .
# 5. Take unbalanced class numbers and excluding outliers into consideration.
# 6. Use the best parameters  CC  and  γγ  to train the whole training dataset.
# 7. Test on test samples.
# 8. Analyze the results.

# data encode
data_train = convertClassToEncodeNumber(data_train, "class", "Y")
data_test = convertClassToEncodeNumber(data_test, "class", "Y")
trainX = data_train.iloc[:, 1:22]
trainY = data_train.iloc[:, -1]

# data scaling
trainXs = standardScaler(trainX).transform(trainX)
trainX_scale = pd.DataFrame(trainXs, columns=trainX.columns)

testX = data_test.iloc[:, 1:22]
testY = data_test.iloc[:, -1]
testX_scale = standardScaler(trainX).transform(testX)

# boxplot of train data
bxplot = pd.DataFrame(trainX_scale).boxplot()
bxplot.xaxis.set_ticklabels(trainX_scale.columns, rotation=60)
bxplot.axhline(y=6, color='r', linestyle='-')

# Outlier module
# Handle outliers from the train dataset, in that the svm is sensitive for the outliers.
outliers = trainX_scale.loc[(trainX_scale > 6).any(1)]
trainX_scale_better = trainX_scale.drop(outliers.index)
trainY_better = trainY.drop(outliers.index)

bxplot = pd.DataFrame(trainX_scale_better).boxplot()
bxplot.xaxis.set_ticklabels(trainX_scale_better.columns, rotation=60)

# SVM RBF grid search with cross validation
# dataset without outlier: trainX_scale_better, trainY_better
# dataset has outlier: trainX_scale, trainY
# cRange = [1, 10, 100, 1000]
# gammaRange = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
# cRange = range(5, 30)
cRange = [22]
gammas= range(1, 100)
gammaRange = 0.001 * np.array(gammas)
parameters = [{'kernel': ['rbf'], 'gamma': gammaRange,
                     'C': cRange}]
clf = GridSearchCV(cv = 3, estimator=SVC(decision_function_shape='ovo', class_weight="balanced"), param_grid = parameters)
clf.fit(trainX_scale_better, trainY_better)
print("Best parameters for the C and gamma: %s" % clf.best_params_)
print("Best parameters accuracy for validation: %s" % clf.best_score_)
clf.grid_scores_

# RBF-SVM grid search parameters results
# Model                C    γ     Training  Valid   Test
# Baseline	           1	auto	1	       -	0.155
# Proposed A	      23	0.010	0.905	0.795	0.863
# Proposed B	      10	0.011	0.864	0.785	0.863
# Proposed C	      29	0.010	0.916	0.798	0.869
# Proposed D	      22	0.010	0.910	0.784	0.839

## predict the train and test results
rbf_svm = SVC(kernel = 'rbf', C=29, gamma=0.01, random_state = 25, decision_function_shape='ovo')
rbf_svm.fit(trainX_scale, trainY)

trainY_pred = rbf_svm.predict(trainX_scale_better)
accuracyTrain = accuracy_score(trainY_better, trainY_pred)
print("Train Accuracy: %f" % accuracyTrain)

testY_pred = rbf_svm.predict(testX_scale)
accuracyTest = accuracy_score(testY, testY_pred)
print("Test Accuracy: %f" % accuracyTest)

############## Proposed C result ##############
#     rbf baseline train accuracy: 0.916000   #
#     rbf baseline test accuracy: 0.869048    #
###############################################

## draw confusion metrics for train
cm = confusion_matrix(trainY_better, trainY_pred)
title = 'RBF-SVM train Confusion Matrix of Model C, \n accuracy = %f' % accuracyTrain
drawConfusionMatrix(title, cm)

## draw confusion metrics for test
cm = confusion_matrix(testY, testY_pred)
title = 'RBF-SVM test Confusion Matrix of Model C, \n accuracy = %f' % accuracyTest
drawConfusionMatrix(title, cm)


## learning curve for the train dataset of the model C
cv = ShuffleSplit(n_splits=3, test_size=1/3, random_state=0)
estimator = rbf_svm
title = "Learning Curves (Proposed RBF-SVM, parameters using Model C)"
plot_learning_curve(estimator, title, trainX_scale_better, trainY_better, ylim=(0, 1.01), cv=cv, n_jobs=4)

## learning curve for the baseline model
estimator = model
title = "Learning Curves (Baseline RBF-SVM)"
plot_learning_curve(estimator, title, trainX, trainY, ylim=(0, 1.01), cv=cv, n_jobs=4)