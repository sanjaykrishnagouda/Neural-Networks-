#p
# Training set of 60,000 letters each of 784 pixels -> 28 X 28
# Test set of 10,000 letters 784 pixels
import os,struct
from array import array
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load(path_lbl, path_img):
	with open(path_lbl,'rb') as file:
		magic, size = struct.unpack(">II", file.read(8))
		labels = array("B",file.read())
	with open(path_img,'rb') as file:
		magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
		image_data = array("B",file.read())
	images = []
	for i in range(size):
		images.append([0]*rows*cols)
	for i in range(size):
		images[i][:] = image_data[i * rows * cols:(i + 1) * rows * cols]
	
	return(np.array(labels), np.array(images))

def show(image):
	reshaped_image=image.reshape(28,28, order = 'C')
	#reshaped_image = image.flatten()

	print(reshaped_image)

train_labels,train_images = load ('train-labels.idx1-ubyte','train-images.idx3-ubyte')

#onehot_vector = np.zeros((train_labels.shape[0],10))
 



#print(onehot_vector)
"""

test_labels, test_images = load ('t10k-labels.idx1-ubyte','t10k-images.idx3-ubyte')
print(test_images[1071].reshape(28,28))
test_labels=test_labels.reshape(test_labels.shape[0],1)
print(test_labels.shape[0])
for i in range(100):
	show(train_images[i])
"""