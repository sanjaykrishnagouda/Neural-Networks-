"""
Contains activations used in different layers:
- Sigmoid
- Softmax
- Tanh
- ReLU
"""
import numpy as np

def sigmoid(z):
	return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
	return sigmoid(z) * (1 - sigmoid(z))

def softmax(z):
	return np.exp(z) / np.sum(np.exp(z))

def softmax_prime(z):
	return softmax(z) * (1 - softmax(z))

def tanh(z):
	return np.tanh(z)

def tanh_prime(z):
	return 1 - tanh(z) * tanh(z)

def relu(z):
    for i in range(0,len(z)):
    	for j in range(len(z[i])):
    		if z[i][j] > 0:
    			pass
    		if z[i][j] <= 0:
    			z[i][j] = 0
    return z

def relu_prime(x):
	for i in range(0, len(x)):
		for k in range(len(x[i])):
			if x[i][k] > 0:
				x[i][k] = 1
			else:
				x[i][k] = 0
	return x