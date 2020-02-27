import pandas as pd
from Perceptron import *
import numpy as np
from sklearn.neural_network import MLPClassifier # neural network
from sklearn.model_selection import train_test_split
from sklearn import metrics

class MultilayerPerceptron:

	def __init__(self, hidden_layer_sizes=[2, 3], learning_rate=0.001, max_iter=200, error_treshold=0.0001, batch_size=32):
		# Attributes
		self.layers = []
		self.hidden_layer_sizes = hidden_layer_sizes
		self.learning_rate = learning_rate
		self.max_iter = max_iter
		self.error_treshold = error_treshold
		self.batch_size = batch_size


	def fit(self, data_inputs, target):
		self.data_inputs = data_inputs
		self.target = target
		self.classes = self.target.unique()

		try:
			# Construct first hidden layer of perceptrons
			number_of_perceptrons_first_hidden_layer = self.hidden_layer_sizes[0]
			number_of_inputs = len(self.data_inputs.columns)
			first_hidden_layer = self.initialize_perceptrons_in_layer(number_of_perceptrons_first_hidden_layer, number_of_inputs)
			self.layers.append(first_hidden_layer)

			# Initialize perceptrons in the hidden layers (from index 1)
			for layer_idx in range(1, len(self.hidden_layer_sizes)):
				# hidden_layer = Array of perceptrons
				number_of_perceptrons_current_layer = self.hidden_layer_sizes[layer_idx]
				number_of_perceptrons_previous_layer = self.hidden_layer_sizes[layer_idx - 1]
				hidden_layer = self.initialize_perceptrons_in_layer(number_of_perceptrons_current_layer, number_of_perceptrons_previous_layer)
				self.layers.append(hidden_layer)

			# Construct last (output) layer of perceptrons
			number_of_perceptrons_last_layer = len(self.target.unique())
			number_of_perceptrons_previous_layer = self.hidden_layer_sizes[-1]

			output_layer = self.initialize_perceptrons_in_layer(number_of_perceptrons_last_layer, number_of_perceptrons_previous_layer)
			self.layers.append(output_layer)

		except Exception as e:
			print(e)
			# Construct last (output) layer of perceptrons
			number_of_perceptrons_last_layer = len(self.target.unique())
			number_of_perceptrons_previous_layer = len(self.data_inputs.columns)
			output_layer = self.initialize_perceptrons_in_layer(number_of_perceptrons_last_layer, number_of_perceptrons_previous_layer)
			self.layers.append(output_layer)

		# Start feed forward and backward prop
		number_of_rows = len(data_inputs)
		for iteration in range(self.max_iter):
			error_total = 0
			for row in range(number_of_rows):
				self.feed_forward(row)

				# Do backward prop then get error
				error = self.backward_prop(row)
				error_total += error
				
				if (row % self.batch_size == 0):
					self.update_all_weights()

			self.update_all_weights()

			if (error_total < self.error_treshold):
				break

	def update_all_weights(self):
		for layer in self.layers:
			for perceptron in layer:
				perceptron.update_weight()

	def calculate_error(self, diff):
		return 0.5 * (diff ** 2)

	def initialize_perceptrons_in_layer (self, number_of_perceptrons, number_of_inputs):
		layer = []
		for idx_perceptron in range(number_of_perceptrons):
			layer.append(Perceptron(self.learning_rate, number_of_inputs+1))
		return layer

	def feed_forward(self, row):
		inputs = []
		outputs = []
		# Initial inputs
		for column in self.data_inputs.columns:
			inputs.append(self.data_inputs[column][row])
		inputs.append(1)

		for layer_idx in range(len(self.layers)):
			outputs.clear()
			for perceptron in self.layers[layer_idx]:
				perceptron.input_data(inputs)
				perceptron.calc_sigmoid()
				outputs.append(perceptron.output)

			inputs.clear()
			for output_data in outputs:
				inputs.append(output_data)

			inputs.append(1)

	def backward_prop(self, row):
		# Last layer
		total_error = 0
		for i in range(len(self.layers[-1])):
			perceptron = self.layers[-1][i]
			# Calculate diff (multiplier):
			if self.classes[i] == self.target[row]:
				result = 1
			else:
				result = 0
			diff = result - perceptron.output
			perceptron.calc_delta(diff)
			perceptron.update_delta_weight()
			total_error += self.calculate_error(diff)

		# Hidden layers
		for layer_idx in range(len(self.layers)-1): #banyaknya layer di layers, kecuali output layer
			layer_size = len(self.layers[-layer_idx-2]) #banyaknya perceptron di layer itu
			for perc_idx in range(layer_size): #untuk setiap perceptron di layer itu
				diff = 0
				for next_perceptron in self.layers[-layer_idx-1]:
					diff += next_perceptron.delta * next_perceptron.weight[perc_idx]
				self.layers[-layer_idx-2][perc_idx].calc_delta(diff)
				self.layers[-layer_idx-2][perc_idx].update_delta_weight()

		return total_error

	def predict(self, data_inputs):
		inputs = []
		outputs = []
		predictions = []
		for row in range(len(data_inputs)):
			inputs.clear()
			outputs.clear()
			# Initial inputs
			for column in data_inputs.columns:
				inputs.append(data_inputs[column][row])
			inputs.append(1)

			for layer_idx in range(len(self.layers)):
				outputs.clear()
				for perceptron in self.layers[layer_idx]:
					perceptron.input_data(inputs)
					perceptron.calc_sigmoid()
					outputs.append(perceptron.output)
				inputs.clear()
				for output in outputs:
					inputs.append(output)
				inputs.append(1)
			idx = outputs.index(max(outputs))
			predictions.append(self.classes[idx])
		return predictions

	def show_model(self):
		for layer_idx in range(len(self.layers)):
			for perceptron_idx in range(len(self.layers[layer_idx])):
				for weight_idx in range(len(self.layers[layer_idx][perceptron_idx].weight)):
					print("Weight ", weight_idx, "-", "[", layer_idx, "][", perceptron_idx, "]: ", self.layers[layer_idx][perceptron_idx].weight[weight_idx])
				


data = pd.read_csv("iris.csv")
for i in range(10):
	mlp = MultilayerPerceptron()
	inputs = data.drop('species', axis = 1)
	target = data[['species']]
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

	mlp.fit(trainX, trainY)
	prediction = mlp.predict(testX)
	val = 0
	for i in range(len(prediction)):
		if prediction[i] == testY.values[i]:
			val += 1
	print("Tested : " + str(len(prediction)))
	print("True : " + str(val))
	print("Accuracy : " + str(val/len(prediction)))

	print("Model (weights):")

	mlp.show_model()
