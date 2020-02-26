import math
import numpy as np

def sigmoid(x):
	return 1 / (1 + math.exp(-x))

class Perceptron:

	def __init__(self, rate, input_length):
		self.data = []
		self.weight = []
		self.delta_weight = []
		self.rate = rate

		random_matrix = np.random.randn(1, input_length) * np.sqrt(1 / input_length)
		for rand_array in random_matrix:
			for rand_num in rand_array:
				self.weight.append(rand_num)
		
		# print(self.weight)

		for inp in range(input_length):
			self.delta_weight.append(0)
			

	def input_data(self, data):
		self.data = []
		for datum in new_data:
			self.data.append(datum)

	def calc_sigmoid(self):
		jumlah = 0
		for i in range(len(self.data)):
			jumlah += self.data[i] * self.weight[i]
		self.output = sigmoid(jumlah)

	#for backprop
	def calc_delta(self, multiplier):
		self.delta = self.output * (1-self.output) * multiplier

	def update_delta_weight(self):
		for i in range(len(self.delta_weight)):
			self.delta_weight[i] += self.rate * self.delta * self.data[i]
	
	# End of batch-size
	def update_weight(self):
		for i in range(len(self.weight)):
			self.weight[i] += self.delta_weight[i]
			self.delta_weight[i] = 0