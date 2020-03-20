import pandas as pd
import numpy as np
from MultilayerPerceptron import *
from sklearn.model_selection import train_test_split

# Confusion Matrix
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 

data = pd.read_csv("iris.csv")

# mlp = myMLP(max_iter = 1000, hidden_layer_sizes=[10, 5, 2])

inputs = data.drop('species', axis = 1)
target = data['species']
# mlp.fit(inputs,target)

# print("Model (weights):")
# mlp.show_model()
# mlp.save_model_to_file("Output-MLP.txt")
# print("SAVE MODELLL:")
# mlp.load_model_from_file("Output-MLP.txt")
# mlp.fit(inputs, target)
# mlp.save_model_to_file("Output-MLP 2.txt")


# #training
df = pd.concat([inputs, target], axis=1)
train, test = train_test_split(df, test_size=0.3)

trainX = train[['sepal_length','sepal_width','petal_length','petal_width']] # taking the training data features
trainY = train.species # output of our training data
testX = test[['sepal_length','sepal_width','petal_length','petal_width']] # taking test data features
testY = test.species   # output value of test data

trainX = trainX.reset_index(drop=True)
# print("trainX")
# print(trainX)

trainY = trainY.reset_index(drop=True)
# print("trainY")
# print(trainY)

testX = testX.reset_index(drop=True)
# print("testX")
# print(testX)

testY = testY.reset_index(drop=True)
# print("testY")
# print(testY)

mlp2 = myMLP(max_iter = 1000, hidden_layer_sizes=[10, 5, 2])
mlp2.fit(trainX, trainY)
prediction = mlp2.predict(testX)

# val = 0
# for i in range(len(prediction)):
	# if prediction[i] == testY.values[i]:
		# val += 1
# print("Tested : " + str(len(prediction)))
# print("True : " + str(val))
# print("Accuracy : " + str(val/len(prediction)))

print("testY.values")
print(testY.values)
print("prediction")
print(prediction)

confusion_matrix_results = confusion_matrix(testY.values, prediction)

print("Confusion Matrix: ")
print(confusion_matrix_results)
print("Accuracy Score: ")
print(accuracy_score(testY.values, prediction))
print("Classification Report: ")
print(classification_report(testY.values, prediction))