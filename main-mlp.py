import pandas as pd
import numpy as np
from MultilayerPerceptron import *
from sklearn.model_selection import train_test_split

# Confusion Matrix
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 

# Kfold
from sklearn.model_selection import KFold

data = pd.read_csv("iris.csv")

inputs = data.drop('species', axis = 1)
target = data['species']

# #training
# df = pd.concat([inputs, target], axis=1)
# train, test = train_test_split(df, test_size=0.3)

# trainX = train[['sepal_length','sepal_width','petal_length','petal_width']] # taking the training data features
# trainY = train.species # output of our training data
# testX = test[['sepal_length','sepal_width','petal_length','petal_width']] # taking test data features
# testY = test.species   # output value of test data

# trainX = trainX.reset_index(drop=True)
# trainY = trainY.reset_index(drop=True)
# testX = testX.reset_index(drop=True)
# testY = testY.reset_index(drop=True)
# mlp = myMLP()
# mlp.fit(trainX, trainY)
# prediction = mlp.predict(testX)
# confusion_matrix_results = confusion_matrix(testY.values, prediction)

# print("Confusion Matrix: ")
# print(confusion_matrix_results)
# print("Accuracy Score: ")
# print(accuracy_score(testY.values, prediction))
# print("Classification Report: ")
# print(classification_report(testY.values, prediction))

#Training k-fold
df = pd.concat([inputs, target], axis=1)
train, test = train_test_split(df, test_size=0.2)  # divide 150 data to x and y = 150-x

trainX = train[['sepal_length','sepal_width','petal_length','petal_width']] # taking the training data features
trainY = train.species # output of our training data
testX = test[['sepal_length','sepal_width','petal_length','petal_width']] # taking test data features
testY = test.species   # output value of test data

trainX = trainX.reset_index(drop=True) 
# print("trainX", trainX)
trainY = trainY.reset_index(drop=True)
# print("trainY", trainY)
testX = testX.reset_index(drop=True)
# print("testX", testX)
testY = testY.reset_index(drop=True)
# print("testY", testY)

#k-fold
kf = KFold(n_splits=12)
kf.get_n_splits(trainX)

mlp = myMLP()

for train_index, test_index in kf.split(trainX): 
    print("TEST:", test_index)
    test_dataframe = pd.DataFrame()
    test_results = []
    for n in test_index:
        test_dataframe = test_dataframe.append(trainX.iloc[[n]])
        test_results.append(trainY[n])
    print("test_results = \n", test_results)
    print("test_dataframe = \n", test_dataframe)
    prediction_test_dataframe = mlp.predict(test_dataframe)
    accuracy_of_test = accuracy_score(prediction_test_dataframe, test_results)
    print("accuracy of test = ", accuracy_of_test)
