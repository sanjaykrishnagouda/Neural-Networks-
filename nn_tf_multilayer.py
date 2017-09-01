#python3!

"""
accuracy with 4 layers : ~ 88 %
accuracy with 3 layers : ~87.3%
accuracy with 1 layer : ~94%

"""



import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt
import matplotlib as mpl

RANDOM_SEED = 7
tf.set_random_seed(RANDOM_SEED)
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# neural network architecture


hidden_layer_1 = 512
hidden_layer_2 = 256
hidden_layer_3 = 256
hidden_layer_4 = 128
hidden_layer_5 = 64
input_layer = 784
output_classes = 10

# parameters for gradient descent
# already optimized. Tested a lot of permutations

learning_rate = 0.001
training_epochs = 400
batch_size = 100
display_step = 10

# graph input 
X = tf.placeholder(tf.float32, [None, input_layer])
Y = tf.placeholder(tf.float32, [None, output_classes])

# TF Variables for weights and biases 

weights = {
	'w1': tf.Variable(tf.random_normal([input_layer, hidden_layer_1])),
	'w2': tf.Variable(tf.random_normal([hidden_layer_1, hidden_layer_2])),
	'w3': tf.Variable(tf.random_normal([hidden_layer_2, hidden_layer_3])),
	'w4': tf.Variable(tf.random_normal([hidden_layer_3,hidden_layer_4])),
	'w5': tf.Variable(tf.random_normal([hidden_layer_4,hidden_layer_5])),
	'out': tf.Variable(tf.random_normal([hidden_layer_5, output_classes]))
}

biases = {
	'b1': tf.Variable(tf.random_normal([hidden_layer_1])),
	'b2': tf.Variable(tf.random_normal([hidden_layer_2])),
	'b3': tf.Variable(tf.random_normal([hidden_layer_3])),
	'b4': tf.Variable(tf.random_normal([hidden_layer_4])),
	'b5': tf.Variable(tf.random_normal([hidden_layer_5])),
	'out': tf.Variable(tf.random_normal([output_classes])),

}

# model creation

def nn(X):
	# Z = X*W + b
	layer_1 = tf.add(tf.matmul(X,weights['w1']),biases['b1'])
	layer_2 = tf.add(tf.matmul(layer_1,weights['w2']),biases['b2'])
	layer_3 = tf.add(tf.matmul(layer_2,weights['w3']),biases['b3'])
	layer_4 = tf.add(tf.matmul(layer_3,weights['w4']),biases['b4'])
	layer_5 = tf.add(tf.matmul(layer_4,weights['w5']),biases['b5'])
	output_layer = tf.add(tf.matmul(layer_5,weights['out']),biases['out'])
	return output_layer

# construct 

logits = nn(X)


# cross entropy and optimization function ( not using gradient descent here but feel free to uncomment)
# Gradient descent is slower. Change training_epoch to a lower value and test it urself.
# Can use the following classification functions :
#	softmax
#	sigmoid
#	sparse softmax or 
#	weighted cross entropy

#ce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
ce_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = Y))

#optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(ce_loss)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(ce_loss)

#train = optimizer.minimize(ce_loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)

	for epoch in range(training_epochs):
		avg_cost = 0.
		total_batch = int(mnist.train.num_examples/batch_size)

		for i in range(total_batch):
			batch_xs, batch_ys = mnist.train.next_batch(batch_size)

			el, c = sess.run([train,ce_loss], feed_dict = {X: batch_xs,
				Y: batch_ys})
			avg_cost += c / total_batch
			
		#if epoch%display_step == 0:
			#print("Epoch:", '%04d'%(epoch), "cost = {:.9f}".format(avg_cost))
	print("Optimization with sigmoid fn finished")
	pred = tf.nn.softmax(logits)
	correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(Y,1))

	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	print("Accuracy with sigmoid:",accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))


with tf.Session() as sess_2:
	sess_2.run(init)

	for epoch in range(training_epochs):
		avg_cost = 0.
		total_batch = int(mnist.train.num_examples/batch_size)

		for i in range(total_batch):
			batch_xs, batch_ys = mnist.train.next_batch(batch_size)

			el, c = sess_2.run([train,ce_loss], feed_dict = {X: batch_xs,
				Y: batch_ys})
			avg_cost += c / total_batch
		
	print("Optimization with softmax fn finished")

	pred = tf.nn.softmax(logits)
	correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(Y,1))

	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	print("Accuracy with softmax:",accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))