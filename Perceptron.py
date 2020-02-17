import math

def sigmoid(x):
	return 1 / (1 + math.exp(-x))

class Perceptron():
	
	self.data = []
	self.weight = []
	#self.rate
	#self.sum
	#self.output
	#self.delta

	def __init__(self, inputs, rate):
		self.rate = rate
		for inp in inputs:
			self.data.append(inp)
			self.weight.append(0)
		self.calcPhi()

	def calcPhi(self):
		jumlah = 0
		for i in range(len(self.data)):
			jumlah += self.data[i] * self.weight[i]
		self.output = sigmoid(jumlah)

	#for backprop
	def delta(self, multiplier):
		self.delta = self.output * (1-self.output) * multiplier

	def update(self)
		for i in range(len(self.weight)):
			self.weight[i] += self.rate * self.delta * self.data[i]