
import numpy as np
from util import *

sigmoid = lambda x: 1.0/(1.0+np.exp(-x))

def fuzzy_or(x, y):
	return 1.0 - (1.0 - x) * (1.0 - y)

def fuzzy_and(x, y):
	return x * y

def fuzzy_not(x):
	return 1.0 - x

def fuzzy_xor(x, y):
	return fuzzy_and(fuzzy_or(x, y), fuzzy_not(fuzzy_and(x, y)))

np.random.seed(11)

input_size = 2
output_size = 1
batch_size = 1
seq_length = 200
layer_size = 30

T = 10.0

W0 = 2.0*(np.random.random((input_size, layer_size))-0.5)
W1 = 2.0*(np.random.random((layer_size, output_size))-0.5)
Z = W1.T

x = np.zeros((seq_length, batch_size, input_size))
y_target = np.zeros((seq_length, batch_size, output_size))

tl = np.asarray([np.linspace(0, T, seq_length)]*input_size).T
x[:,0,:] = np.sin(tl + 10.0*np.random.random((input_size,)))

y_target[:,0,0] = 0.1*fuzzy_xor(x[:,0,0], x[:,0,1])


hh = np.zeros((seq_length, batch_size, layer_size))
ph = np.zeros((seq_length, batch_size, layer_size))
bh = np.zeros((seq_length, batch_size, layer_size))
yh = np.zeros((seq_length, batch_size, output_size))

y = np.zeros((batch_size, output_size))
h = np.zeros((batch_size, layer_size))
p = np.zeros((batch_size, layer_size))
b = np.zeros((batch_size, layer_size))

dL1h = np.zeros((seq_length, batch_size, output_size))
dL0h = np.zeros((seq_length, batch_size, layer_size))

epochs = 1000
yhh = np.zeros((epochs, seq_length, batch_size, output_size))

lrate = 0.005
for epoch in xrange(epochs):
	alphah = np.zeros((seq_length, batch_size, output_size))
	alphah[np.random.random(alphah.shape) > 0.90] = 1.0

	for ti in xrange(seq_length):
		xt, yt_target = x[ti], y_target[ti]
		alpha = alphah[ti]

		h_new = sigmoid(np.dot(xt, W0))
		
		y_new = (1.0 - alpha)*sigmoid(np.dot(h_new, W1)) + alpha * yt_target
		p_new = sigmoid(np.dot(y_new, Z))

		b_new = p_new*h_new
		
		dL1 = alpha * (y_new - y)
		dL0 = alpha * (b_new - b)

		W0 += lrate * np.dot(xt.T, dL0)
		W1 += lrate * np.dot(h_new.T, dL1)

		h = h_new.copy()
		b = b_new.copy()
		p = p_new.copy()
		y = y_new.copy()

		hh[ti] = h.copy()
		yh[ti] = y.copy()
		ph[ti] = p.copy()
		bh[ti] = b.copy()
		dL1h[ti] = dL1.copy()
		dL0h[ti] = dL0.copy()

	# if epoch % 10 == 0:
	# 	shl(y_target, yh)
	yhh[epoch] = yh.copy()

	print "Epoch {}, error: {}".format(epoch, np.sum(np.square(yh - yt_target)))

