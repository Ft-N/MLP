import sklearn
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier # neural network
from sklearn.model_selection import train_test_split
from sklearn import metrics

data = pd.read_csv("iris.csv")
data.sample(5)
data.head(5)
target = data[['species']].replace(['Iris-setosa','Iris-versicolor','Iris-virginica'],[0,1,2])
target.sample(5)
newData = data.drop('species', axis = 1)
df = pd.concat([newData, target], axis=1)
train, test = train_test_split(df, test_size=0.3)
trainX = train[['sepal_length','sepal_width','petal_length','petal_width']] # taking the training data features
trainY=train.species# output of our training data
testX= test[['sepal_length','sepal_width','petal_length','petal_width']] # taking test data features
testY =test.species   #output value of test data

trainX.head(5)
trainY.head(5)

testX.head(5)
testY.head(5)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3, 3), random_state=1)
clf.fit(trainX, trainY)

prediction = clf.predict(testX)
print(prediction)

print(testY.values)

print('The accuracy of the Multi-layer Perceptron is:',metrics.accuracy_score(prediction,testY))