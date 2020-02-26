import pandas as pd
from Perceptron import *

class MultilayerPerceptron:

	def __init__(self, hidden_layer_sizes=[3], learning_rate=0.01, max_iter=100, error_treshold=0.01, batch_size=32):
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

		# Construct first hidden layer of perceptrons
		try:
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
			for row in range(number_of_rows):
				self.feed_forward(row)
				self.backward_prop(row)

			if (row % self.batch_size == 0):
				# Update all weights
				for layer_idx in range(self.layers):
					for perceptron in self.layers[layer_idx]:
						perceptron.update_weight()

	
	def initialize_perceptrons_in_layer (self, number_of_perceptrons, number_of_inputs):
		layer = []
		for idx_perceptron in range(number_of_perceptrons):
			layer.append(Perceptron(self.learning_rate, number_of_inputs))
		return layer

	def feed_forward(self, row):
		inputs = []
		# Initial inputs
		for column in self.data_inputs.columns:
   	 		inputs.append(self.data_inputs[column][row])

		for layer_idx in range(self.layers):
			outputs = []
			for perceptron in self.layers[layer_idx]:
				perceptron.input_data(inputs)
				outputs.append(perceptron.output)

			inputs = []
			for output_data in outputs:
				inputs.append(output_data)

	def backward_prop(self, row):
		# Last layer
		for perceptron in self.layers[-1]:
			# Calculate error (multiplier):
			error = perceptron.output - self.target[row]
			perceptron.calc_delta(error)
			perceptron.update_delta_weight()

		# Hidden layers
		for layer in range(len(self.le)):
			for index in range(self.layers[layer]):
				error = 0
				for next_perceptron in self.layers[layer + 1]:
					error += next_perceptron.delta * next_perceptron.weight[index]
				self.layers[layer][index].calc_delta(error)
				self.layers[layer][index].update_delta_weight()


data = pd.read_csv("iris.csv")
mlp = MultilayerPerceptron()
inputs = data.drop(['target'])
target = data.target
mlp.fit(inputs, target)