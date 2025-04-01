import numpy as np
#Input layer
inputs = [[1.0, 2.0, 3.0, 2.5],
 [2.0, 5.0, -1.0, 2.0],
 [-1.5, 2.7, 3.3, -0.8]]
#Multiple hidden layers
weights1 = [[0.2, 0.8, -0.5, 1.0],
 [0.5, -0.91, 0.26, -0.5],
 [-0.26, -0.27, 0.17, 0.87]]
biases1 = [2.0, 3.0, 0.5]

weights2 = [[0.1, -0.14, 0.5],
    [-0.5, 0.12, -0.33],
    [-0.44, 0.73, -0.13]]
biases2 = [-1.0, 2.0, -0.5]


#Stacking the layers together
weights = [weights1, weights2]
biases = [biases1, biases2]
layer_input = inputs
for i in range(len(weights)):
    output = np.dot(layer_input, np.array(weights[i]).T) + biases[i]
    print(output)
    layer_input = output  # Pass the output of the current layer as input to the next layer
