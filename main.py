import pandas as pd
data = pd.read_csv("iris.csv")
# print(data['species'].unique())

print(data.columns)

for column in data.columns:
    print(data[column][0])
# print(data[data.keys()[3]][0])
# print(data.keys())
# print(data[data.keys()[1]][0])
# for input_index in range(len(data.columns)):
# 	print(data[input_index])
# perceptron = MultilayerPerceptron(data)

# initial_input.append(self.data_inputs[0][input_index])
# print(data)