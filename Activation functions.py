# Activation functions
#captures non linearity in neural networks
import numpy as np
import matplotlib.pyplot as plt
import nnfs
#input library for spiral data
from nnfs.datasets import spiral_data

#Dense Layer: A layer in a neural network that is fully connected
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        #initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        #calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    #forward pass
    def forward(self, inputs):
        #calculate output values from inputs
        self.output = np.maximum(0, inputs)
   
class Activation_Softmax:
    #forward pass
    def forward(self, inputs):
        #get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        #normalize them for each example
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        


## 1. Sigmoid
# Sigmoid function is a popular activation function
# f(x) = 1 / (1 + exp(-x))
# It squashes the input into the range [0, 1]

input = ([-10, -5, 0, 5, 10], [-1, -0.5, 0, 0.5, 1]), ([-10, -5, 0, 5, 10], [-1, -0.5, 0, 0.5, 1])
# get unnormalized probabilities
exp_values = np.exp(input - np.max(input, axis=1, keepdims=True))
print(exp_values)
# normalize them for each example
probs = exp_values / np.sum(exp_values, axis=1, keepdims=True)
print(probs)
np.sum(probs, axis=1)


#create dataset
x, y = spiral_data(samples=100, classes=3)
#create dense layer with two inputs and three outputs
dense1 = Layer_Dense(2, 3)
#create ReLU activation (to be used with Dense layer)
activation1 = Activation_ReLU()
#create second Dense layer with three inputs (from previous layer) and one output
dense2 = Layer_Dense(3, 3)
#forward pass of our training data through this layer
dense1.forward(x)
#Create Softmax activation (to be used with Dense layer)
activation2 = Activation_Softmax()
# Make a forward pass through second Dense layer
dense1.forward(x)

activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
print(activation2.output[:5])






# ## 2. ReLU
# # ReLU (Rectified Linear Unit) is one of the most popular activation functions
# # f(x) = max(0, x)
# # It helps solve the vanishing gradient problem and introduces non-linearity
# # while being computationally efficient

# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation

# def relu(x):
#     """
#     ReLU activation function
#     Args:
#         x: Input value or array
#     Returns:
#         max(0, x)
#     """
#     return np.maximum(0, x)

# # Example usage
# x = np.array([-2, -1, 0, 1, 2])
# print("Input:", x)
# print("ReLU output:", relu(x))

# # Visualization of basic ReLU
# x = np.linspace(-5, 5, 100)
# y = relu(x)

# plt.figure(figsize=(8, 6))
# plt.plot(x, y, 'b-', label='ReLU')
# plt.grid(True)
# plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
# plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
# plt.title('ReLU Activation Function')
# plt.xlabel('Input')
# plt.ylabel('Output')
# plt.legend()
# plt.show()

# # Demonstrating how ReLU can create complex shapes
# def complex_shape(x):
#     """
#     Creates a complex shape using multiple ReLU functions
#     This demonstrates how neural networks can approximate complex functions
#     """
#     return 2 * relu(x) - 3 * relu(x - 2) + relu(x - 4)

# # Create a more complex visualization
# x = np.linspace(-2, 6, 200)
# y = complex_shape(x)

# plt.figure(figsize=(10, 6))
# plt.plot(x, y, 'r-', label='Complex Shape', linewidth=2)
# plt.grid(True)
# plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
# plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
# plt.title('Complex Shape Created by Combining ReLU Functions')
# plt.xlabel('Input')
# plt.ylabel('Output')
# plt.legend()
# plt.show()

# # Print some example points to understand the shape
# test_points = np.array([-1, 1, 3, 5])
# print("\nComplex shape at test points:")
# for point in test_points:
#     print(f"f({point}) = {complex_shape(point):.2f}")

# # Approximating complex shape with ReLU functions
# def create_complex_approximation(x, num_relus):
#     """
#     Creates an approximation of the complex shape using multiple ReLU functions
#     Args:
#         x: Input values
#         num_relus: Number of ReLU functions to use
#     Returns:
#         Approximated complex shape
#     """
#     result = np.zeros_like(x)
#     # First component: positive slope
#     if num_relus >= 1:
#         result += 2 * relu(x)
#     # Second component: negative slope
#     if num_relus >= 2:
#         result -= 3 * relu(x - 2)
#     # Third component: positive slope
#     if num_relus >= 3:
#         result += relu(x - 4)
#     return result

# # Create animation of complex shape approximation
# def animate_complex(frame):
#     num_relus = frame + 1
#     approx = create_complex_approximation(x, num_relus)
#     line_approx.set_data(x, approx)
#     ax.set_title(f'Approximating Complex Shape with {num_relus} ReLU Functions')
#     return line_approx, ax

# # Set up complex shape animation
# fig, ax = plt.subplots(figsize=(12, 6))
# x = np.linspace(-2, 6, 200)
# true_shape = complex_shape(x)
# line_true, = ax.plot(x, true_shape, 'b-', label='True Shape', linewidth=2)
# line_approx, = ax.plot(x, np.zeros_like(x), 'r--', label='ReLU Approximation', linewidth=2)

# ax.set_xlim(-2, 6)
# ax.set_ylim(-2, 4)
# ax.grid(True)
# ax.set_title('Approximating Complex Shape with ReLU Functions')
# ax.set_xlabel('Input')
# ax.set_ylabel('Output')
# ax.legend()

# # Create complex shape animation
# anim_complex = animation.FuncAnimation(fig, animate_complex, frames=3, interval=1000, blit=True)
# plt.show()

# # Approximating sine wave with ReLU functions
# def create_sine_approximation(x, num_relus):
#     """
#     Creates an approximation of sine wave using multiple ReLU functions
#     Args:
#         x: Input values
#         num_relus: Number of ReLU functions to use
#     Returns:
#         Approximated sine wave
#     """
#     result = np.zeros_like(x)
#     for i in range(num_relus):
#         # Create shifted and scaled ReLU functions
#         shift = 2 * np.pi * i / num_relus
#         scale = 2 * np.pi / num_relus
#         result += np.sin(shift) * relu(x - shift) * scale
#     return result

# def animate_sine(frame):
#     num_relus = frame + 1
#     approx = create_sine_approximation(x, num_relus)
#     line_approx.set_data(x, approx)
#     ax.set_title(f'Approximating Sine Wave with {num_relus} ReLU Functions')
#     return line_approx, ax

# # Set up sine wave animation
# fig, ax = plt.subplots(figsize=(12, 6))
# x = np.linspace(0, 2*np.pi, 200)
# true_sine = np.sin(x)
# line_true, = ax.plot(x, true_sine, 'b-', label='True Sine Wave', linewidth=2)
# line_approx, = ax.plot(x, np.zeros_like(x), 'r--', label='ReLU Approximation', linewidth=2)

# ax.set_xlim(0, 2*np.pi)
# ax.set_ylim(-1.5, 1.5)
# ax.grid(True)
# ax.set_title('Approximating Sine Wave with ReLU Functions')
# ax.set_xlabel('Input')
# ax.set_ylabel('Output')
# ax.legend()

# # Create sine wave animation
# anim_sine = animation.FuncAnimation(fig, animate_sine, frames=20, interval=200, blit=True)
# plt.show()







## 3. Tanh
