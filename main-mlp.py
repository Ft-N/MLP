import pandas as pd
import numpy as np
from MultilayerPerceptron import *
from sklearn.model_selection import train_test_split

# Confusion Matrix
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 

data = pd.read_csv("iris.csv")

inputs = data.drop('species', axis = 1)
target = data['species']

# #training
df = pd.concat([inputs, target], axis=1)
train, test = train_test_split(df, test_size=0.3)

trainX = train[['sepal_length','sepal_width','petal_length','petal_width']] # taking the training data features
trainY = train.species # output of our training data
testX = test[['sepal_length','sepal_width','petal_length','petal_width']] # taking test data features
testY = test.species   # output value of test data

trainX = trainX.reset_index(drop=True)
trainY = trainY.reset_index(drop=True)
testX = testX.reset_index(drop=True)
testY = testY.reset_index(drop=True)
mlp = myMLP()
mlp.fit(trainX, trainY)
prediction = mlp.predict(testX)
confusion_matrix_results = confusion_matrix(testY.values, prediction)

print("Confusion Matrix: ")
print(confusion_matrix_results)
print("Accuracy Score: ")
print(accuracy_score(testY.values, prediction))
print("Classification Report: ")
print(classification_report(testY.values, prediction))