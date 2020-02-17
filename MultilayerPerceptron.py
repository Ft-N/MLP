class MultilayerPerceptron:

	def __init__(self, inputs, max_iter, threshold, layer_count=2, unit_per_layer=1, rate=0.01):
		self.inputs = inputs
		self.layers = []
		data = inputs
		for i in range(layer_count):
			layer = []
			temp = []
			for j in range(unit_per_layer):
				perceptron = Perceptron(data, rate)
				layer.append(perceptron)
				temp.append(perceptron.output)
			self.layers.append(copy(self.layer))
			data = temp