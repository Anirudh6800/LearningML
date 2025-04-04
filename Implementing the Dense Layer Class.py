import numpy as np
import nnfs
from nnfs.datasets import spiral_data

# Implementing the Dense Layer Class
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        
# Creating the dataset
nnfs.init()
X, y = spiral_data(samples=100, classes=3)

# Creating the Dense Layer
dense1 = Layer_Dense(2, 3)
dense1.forward(X)

print(dense1.output[:5])

        
        
# # Implementing the ReLU Activation Function
# class Activation_ReLU:
#     def forward(self, inputs):
#         self.output = np.maximum(0, inputs)
        
# # Implementing the Softmax Activation Function
# class Activation_Softmax:
#     def forward(self, inputs):
#         exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
#         probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
#         self.output = probabilities
        
# # Implementing the Loss Class
# class Loss:
#     def calculate(self, output, y):
#         sample_losses = self.forward(output, y)
#         data_loss = np.mean(sample_losses)
#         return data_loss
    
