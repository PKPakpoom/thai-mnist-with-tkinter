import numpy as np
import pandas as pd
from PIL import Image

# Activation Functions
def leaky_ReLU(x):
    return np.maximum(x, 0.1 * x)

def leaky_ReLU_derivative(x):
    return np.where(x > 0, 1, 0.1)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Prevent overflow
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# Mean Squared Error Loss
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.shape[0]

def categorical_crossentropy(y, y_pred):
    return np.sum(-np.log(y_pred) * y)

def categorical_crossentropy_derivative(y_true, y_pred):
    return y_pred - y_true


def crop_image(image_path):
    image = Image.open(image_path).convert('L')
    image_array = np.array(image)
    
    binary_image = image_array < 127
    
    coords = np.column_stack(np.where(binary_image))
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    cropped = image.crop((x_min, y_min, x_max + 1, y_max + 1))
    cropped = cropped.resize((28, 28))

    return cropped


class NeuralNetwork:
    def __init__(self, input_layer_size = 784, hidden_layers = [64, 32, 32, 16], output_layer_size = 10):
        np.random.seed(1234)
        self.input_layer_size = input_layer_size  # 28 * 28 pixels
        self.hidden_layers = hidden_layers
        self.output_layer_size = output_layer_size
        self.activation_function = leaky_ReLU
        self.activation_derivative = leaky_ReLU_derivative

        # Initialize weights and biases dynamically
        layer_sizes = [self.input_layer_size] + self.hidden_layers + [self.output_layer_size]
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(1 / layer_sizes[i])
                        for i in range(len(layer_sizes) - 1)]
        self.biases = [np.zeros(layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]  # Initialize biases to zero


    def forward(self, X):
        self.a = [X]  # Store activations for backward pass
        self.z = []  # Store weighted sums for backward pass

        # Hidden layers
        for i in range(len(self.weights) - 1):  
            self.z.append(np.dot(self.a[-1], self.weights[i]) + self.biases[i])
            self.a.append(self.activation_function(self.z[-1]))  # Use activation function here

        # Output layer (no activation here, we apply softmax in loss)
        self.z.append(np.dot(self.a[-1], self.weights[-1]) + self.biases[-1])
        self.a.append(softmax(self.z[-1]))  # Raw output without softmax
        
        # Apply softmax to the output layer
        return self.a[-1]

    def backward(self, y_true, learning_rate):
        # Calculate the loss derivative w.r.t output (using categorical cross-entropy)
        dL_da = categorical_crossentropy_derivative(y_true, self.a[-1])
        
        dW = []
        dB = []
        
        # Backpropagation loop
        for i in reversed(range(len(self.weights))):
            if i == len(self.weights) - 1:
                dL_dz = dL_da  # Output layer gradient (softmax + MSE derivative)
            else:
                dL_dz = np.dot(dL_da, self.weights[i + 1].T) * self.activation_derivative(self.z[i])
            
            dW.insert(0, np.dot(self.a[i].T, dL_dz))
            dB.insert(0, np.sum(dL_dz, axis=0))
            dL_da = dL_dz
        
        # Gradient descent update
        for i in range(len(self.weights)):
            self.weights[i] -= (learning_rate * dW[i]) / y_true.shape[0]
            self.biases[i] -= (learning_rate * dB[i]) / y_true.shape[0]

    def train(self, X, y, epochs = 10, min_learning_rate = 1e-6, max_learning_rate = 1e-2):
        for epoch in range(1, epochs+1):
            current_learning_rate = max_learning_rate - (max_learning_rate - min_learning_rate) * epoch / epochs
            y_pred = self.forward(X)
            loss = categorical_crossentropy(y, y_pred) / y.shape[0]
            self.backward(y, current_learning_rate)

            if epoch % 1 == 0:
                print(f"Epoch {epoch}, learning rate: {current_learning_rate}, Loss: {loss}")