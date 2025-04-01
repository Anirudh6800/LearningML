# Description: This file contains the code for building a neuron, a layer, and a batch of neurons using Numpy.

#Using Numpy build a neuron
import numpy as np
#dot product
inputs = [1.0, 2.0, 3.0, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2.0
output = np.dot(weights, inputs) + bias
print(output)


#Using Numpy build a layer
inputs = [1.0, 2.0, 3.0, 2.5]
weights = [[0.2, 0.8, -0.5, 1.0],
 [0.5, -0.91, 0.26, -0.5],
 [-0.26, -0.27, 0.17, 0.87]]
biases = [2.0, 3.0, 0.5]
output = np.dot(weights, inputs) + biases
print(output)

#Using Numpy build a batch of neurons
inputs = [[1.0, 2.0, 3.0, 2.5],
 [2.0, 5.0, -1.0, 2.0],
 [-1.5, 2.7, 3.3, -0.8]]
weights = [[0.2, 0.8, -0.5, 1.0],
 [0.5, -0.91, 0.26, -0.5],
 [-0.26, -0.27, 0.17, 0.87]]
biases = [2.0, 3.0, 0.5]
output = np.dot(inputs, np.array(weights).T) + biases
print(output)

