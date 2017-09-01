import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from init import load
#from tf_init import main
from activations import sigmoid, sigmoid_prime, softmax, softmax_prime, tanh, tanh_prime,relu, relu_prime
import random, gzip
#train_labels,train_images = load ('train-labels.idx1-ubyte','train-images.idx3-ubyte')  
#test_labels, test_images = load ('t10k-labels.idx1-ubyte','t10k-images.idx3-ubyte')
train_images,train_labels = main()
def layer_sizes(X,Y): # defining the neural network structure
	# returns size of input, hidden and output layer
	n_x = X.shape[1]
	n_h = 5  # size of hidden layer 1
	n_h2 = 5 # size of hidden layer 2
	n_h3 = 5 # size of hidden layer 3
	n_h4 = 5 # size of hidden layer 4
	#Y = Y.reshape((1,Y.shape[0]))
	n_y = Y.shape[0]

	return (n_x, n_h, n_h2, n_h3, n_h4, n_y)
def initialize_prameters(n_x, n_h, n_h2, n_h3, n_h4, n_y):
	#initializing weights and biases for all layers
	sizes = [n_x, n_h, n_h2, n_h3, n_h4, n_y]
	weights = [np.random.randn(a,b)*0.01 for a,b in zip(sizes[1:],sizes[:-1])]
	#weights = [np.random.randn(a,b)/np.sqrt(b) for a,b in zip(sizes[1:],sizes[:-1])]
	biases = [np.random.randn(y,1)*0.01 for y in sizes[1:]]
	parameters = [biases, weights]
	return parameters
def forward_propagation(X,parameters):
	biases = parameters[0]
	weights = parameters[1] 

	Z1 = np.dot(weights[0],X.T) + biases[0]
	A1 = tanh(Z1)
	Z2 = np.dot(weights[1],A1) + biases[1]
	A2 = tanh(Z2)
	Z3 = np.dot(weights[2],A2) + biases[2]
	A3 = tanh(Z3)
	Z4 = np.dot(weights[3],A3) + biases[3]
	A4 = tanh(Z4)
	Z5 = np.dot(weights[4],A4) + biases[4]
	A5 = tanh(Z5)
	"""A = []
	Z = []
	A.append(X.T)
	for  i in range(0,5):
		Z.append(np.dot(weights[i],A[i])+biases[i])
		A.append(sigmoid(Z[i]))
	"""
	cache = {
	'A1' : A1,
	'A2' : A2,
	'A3' : A3,
	'A4' : A4,
	'A5' : A5

	}
	return A5, cache
def compute_cost(A5, Y, parameters):
	Y = Y.reshape(1,Y.shape[0])
	m = Y.shape[1]
	logprobs = np.dot( Y, np.log(A5.T) ) + np.dot( (1-Y), np.log(1-A5.T) ) #cross entropy
	#cost = -(1/m)*(np.sum(logprobs))
	cost = (1/m)*np.sum(np.dot((A5-Y),(A5-Y).T))
	cost = np.squeeze(cost)
	return cost
def backward_propagation(parameters, cache, X, Y):
	# use np.squeeze to make calc values work with fmin_bfgs?
	biases, weights = parameters[0],parameters[1]
	Y = Y.reshape(1,Y.shape[0])
	m = Y.shape[1]
	A1 = cache['A1']
	A2 = cache['A2']
	A3 = cache['A3']
	A4 = cache['A4']
	A5 = cache['A5']

	#for i in cache.keys() : A.append(cache[i]) #a1,a2,a3,a4,a5
	grad_wts = []
	grad_bias = []

	dZ5 = A5 - Y
	dW5 = (1/m)*np.dot(dZ5,A4.T)
	db5 = (1/m)*np.sum(dZ5, axis = 1, keepdims = True)
	grad_bias.append(db5)
	grad_wts.append(dW5)
	
	#for i in range(len(biases)): print(biases[i].shape)
	dZ4 = np.multiply(np.dot(weights[4].T,dZ5),(sigmoid_prime(A4)))
	dW4 = (1/m)*np.dot(dZ4,A3.T)
	grad_wts.append(dW4)
	db4 = (1/m)*np.sum(dZ4, axis = 1, keepdims = True)
	grad_bias.append(db4)

	dZ3 = np.multiply(np.dot(weights[3].T,dZ4),(sigmoid_prime(A3)) )
	dW3 = (1/m)*np.dot(dZ3,A2.T)
	grad_wts.append(dW3)
	db3 = (1/m)*np.sum(dZ3, axis = 1, keepdims = True)
	grad_bias.append(db3)

	dZ2 = np.multiply(np.dot(weights[2].T,dZ3),( sigmoid_prime(A2) ))
	dW2 = (1/m)*np.dot(dZ2,A1.T)
	grad_wts.append(dW2)
	db2 = (1/m)*np.sum(dZ2, axis = 1, keepdims = True)
	grad_bias.append(db2)

	dZ1 = np.multiply(np.dot(weights[1].T,dZ2),(sigmoid_prime(A1)) )
	dW1 = (1/m)*np.dot(dZ1,X)
	grad_wts.append(dW1)
	db1 = (1/m)*np.sum(dZ2, axis = 1, keepdims = True)
	grad_bias.append(db1)

	grads = [grad_bias.reverse(),grad_wts.reverse()]
	return grad_wts,grad_bias
def update_parameters(parameters, grads, learning_rate = 0.5):
	biases = parameters[0]
	weights = parameters[1]
	grad_bias = grads[1]
	grad_wts = grads[0]
	temp_derivs_b = []
	temp_derivs_w = []
	"""for i in range(len(biases)):	
		print('weight derivative shapes', (grad_wts[i].shape))
		print('bias derivative shapes', (grad_bias[i].shape))
	"""
	for i in range(len(biases)):
		temp_derivs_w.append(weights[i]-learning_rate*grad_wts[i])
		temp_derivs_b.append(biases[i]-learning_rate*grad_bias[i])
	
	parameters = [temp_derivs_b,temp_derivs_w]
	return parameters
def nn_model(X, Y, num_iterations = 10000, print_cost = False):
	#np.random.seed(7)
	
	n_x, n_h, n_h2, n_h3, n_h4, n_y = layer_sizes(X,Y)
	parameters = initialize_prameters(n_x, n_h, n_h2, n_h3, n_h4, n_y)

	for i in range(0,num_iterations+1):
		final_activation, cache = forward_propagation(X, parameters)
		cost = compute_cost(final_activation,Y,parameters)
		grads = backward_propagation(parameters,cache,X,Y)
		parameters = update_parameters(parameters,grads, learning_rate = 1.5)
		if np.isnan(cost):
			break
		#print('Running iteration %i/%i'%(i,num_iterations) )
		if print_cost and i%10 == 0 :
			print('iteration %i, current cost is:'%(i),cost)

	return parameters
def predict(parameters, X):
	return predictions
n_x, n_h, n_h2, n_h3, n_h4, n_y = layer_sizes(train_images,train_labels)
print(n_x, n_h, n_h2, n_h3, n_h4, n_y)
#a = nn_model(train_images,train_labels,num_iterations = 30,print_cost = True)
"""for i in range(len(a)):
	for j in range(len(a[i])):
		print(a[i][j].shape)
"""