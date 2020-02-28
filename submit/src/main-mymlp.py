import pandas as pd
import numpy as np
from MultilayerPerceptron import *
from sklearn.model_selection import train_test_split

data = pd.read_csv("iris.csv")

mlp = myMLP(max_iter = 10, hidden_layer_sizes=[100])

inputs = data.drop('species', axis = 1)
target = data['species']
mlp.fit(inputs,target)

print("Model (weights):")
mlp.show_model(20)

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

mlp2 = myMLP(max_iter = 200, hidden_layer_sizes=[100])
mlp2.fit(trainX, trainY)
prediction = mlp2.predict(testX)
val = 0
for i in range(len(prediction)):
	if prediction[i] == testY.values[i]:
		val += 1
print("Tested : " + str(len(prediction)))
print("True : " + str(val))
print("Accuracy : " + str(val/len(prediction)))
