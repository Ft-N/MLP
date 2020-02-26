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
			number_of_perceptrons_last_layer = 1
			number_of_perceptrons_previous_layer = self.hidden_layer_sizes[-1]

			output_layer = self.initialize_perceptrons_in_layer(number_of_perceptrons_last_layer, number_of_perceptrons_previous_layer)
			self.layers.append(output_layer)

		except Exception as e:
			print(e)
			# Construct last (output) layer of perceptrons
			number_of_perceptrons_last_layer = 1
			number_of_perceptrons_previous_layer = len(self.data_inputs.columns)
			output_layer = self.initialize_perceptrons_in_layer(number_of_perceptrons_last_layer, number_of_perceptrons_previous_layer)
			self.layers.append(output_layer)

		# Start feed forward and backward prop
		number_of_rows = len(data_inputs)
		error_total = 0
		
		for iteration in range(self.max_iter):
			for row in range(number_of_rows):
				self.feed_forward(row)

				# Get error and do backward prop
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
		return 0.5 * diff ** 2

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

		for layer_idx in range(self.layers):
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
		for perceptron in self.layers[-1]:
			# Calculate diff (multiplier):
			diff = perceptron.output - self.target[row]
			perceptron.calc_delta(diff)
			perceptron.update_delta_weight()

			error = self.calculate_error(diff)
			total_error += error

		# Hidden layers
		for layer_idx in range(len(self.layers)-1):
			for perc_idx in range(self.layers[-layer_idx-2]):
				diff = 0
				for next_perceptron in self.layers[-layer_idx-1]:
					diff += next_perceptron.delta * next_perceptron.weight[perc_idx]
				self.layers[layer_idx][perc_idx].calc_delta(diff)
				self.layers[layer_idx][perc_idx].update_delta_weight()

		return total_error


data = pd.read_csv("iris.csv")
mlp = MultilayerPerceptron()
# inputs = data.drop(['target'])
# target = data.target
# mlp.fit(inputs, target)