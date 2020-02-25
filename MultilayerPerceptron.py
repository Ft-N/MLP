import pandas as pd
class MultilayerPerceptron:

	def __init__(self, hidden_layers=[3], learning_rate=0.01, max_iter=100, error_treshold=0.01, batch_size=32):
		# Attributes
		self.layers = []
		self.hidden_layers = hidden_layers
		self.learning_rate = learning_rate
		self.max_iter = max_iter
		self.error_treshold = error_treshold
		self.batch_size = 32

	def fit(self, data_inputs, target):
		self.data_inputs = data_inputs
		self.target = target

		# Get how many columns (attributes) in data_inputs:
		input_nums = len(self.data_inputs.columns)
		data_input_nums = len(data_inputs)

		iteration = 1
		while (iteration < self.max_iter):
			self.initial_feed_forward(row)
			self.backward_prop(row)

			for row in range(1, data_input_nums):
				self.backward_prop(row)

			iteration = iteration + 1
	
	def initial_feed_forward(self, row):
		# First layer
		initial_inputs = []
		for column in self.data_inputs.keys():
			initial_input_data = self.data_inputs[column][row]
			initial_inputs.append(initial_input_data)

		perceptron = Perceptron(initial_inputs, self.learning_rate)
		first_hidden_layer = []
		inputs = []

		# Hidden layer
		for layer in range(self.hidden_layers[0]):
			first_hidden_layer.append(perceptron)
			inputs.append(perceptron.output)
			# First hidden layer
			self.layers.append(copy(first_hidden_layer))
			# Second until last hidden_layer
			self.append_perceptrons_from_second_hidden_layer(inputs)

		# Output layer
		output_layer = []

		# Count how many unique value in the target:
		perceptrons_in_output_layer = len(self.target.unique())
		for perceptron_output_layer in range(perceptrons_in_output_layer):
			perceptron = Perceptron(inputs, self.learning_rate)
			output_layer.append(perceptron)
		self.layers.append(copy(output_layer))

	def append_perceptrons_from_second_hidden_layer(self, inputs):
		for layer in range(1, self.hidden_layers):
			hidden_layer = []
			temp = []

			for perceptron_in_hidden_layer in range(self.hidden_layers[layer]):
				perceptron = Perceptron(inputs, self.learning_rate)
				hidden_layer.append(perceptron)
				temp.append(perceptron.output)
			self.layers.append(copy(hidden_layer))

			inputs = []
			for output_data in temp:
				inputs.append(output_data)


	def backward_prop(self, row):
		# Last layer
		last_index = len(self.layers) - 1
		for perceptron in self.layers[last_index]:
			# Calculate error (multiplier):
			error = perceptron.output - self.target[row]
			perceptron.calc_delta(error)
			perceptron.update_delta_weight()

		# Hidden layers
		for layer in range(last_index - 1, 0 - 1):
			for index in range(self.layers[layer]):
				error = 0
				for next_perceptron in self.layers[layer + 1]:
					error += next_perceptron.delta * next_perceptron.weight[index]
				self.layers[layer][index].calc_delta(error)
				self.layers[layer][index].update_delta_weight()


data = pd.read_csv("iris.csv")
# print(data['species'].unique())
# print(data)
# print(data.keys())
# print(data[data.keys()[1]][0])
# for input_index in range(len(data.columns)):
# 	print(data[0])

# perceptron = MultilayerPerceptron(data)

# initial_input.append(self.data_inputs[0][input_index])
# print(data)