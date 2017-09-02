# Gaussian Kernel:
# K(x1, x2) = exp(-gamma * abs(x1 - x2)^2)

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
from nn_tf_multilayer import rets
import time
import datetime as dt


ops.reset_default_graph()
mnist = rets()
sess = tf.Session()

# loading the data
X = mnist.train.images
X = X / 255.0
Y = mnist.train.labels

X_cv = mnist.validation.images
Y_cv = mnist.validation.labels

X_test = mnist.test.images
Y_test = mnist.test.labels


# RBF Kernel parameters










