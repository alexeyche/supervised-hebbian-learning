

import numpy as np
from poc.opt import *
from datasets import *
from poc.common import *


ds = ToyDataset()
act = Relu()

seed = 5
layer_size = 25
dt = 0.25
omega = 0.5
k = 1.0
num_iters = 25

p, q = 0.09, 0.01
lrate_lat = 0.01
lrate_ff = 0.01
t_data_spike = 5

def phi_capital(C):
	return np.sum(np.square(np.sum(C, axis=0) - p)) * k/2.0

def lateral_cost(L, Ldiag):
	return np.sum(np.square(L - p)) + np.sum(np.square(Ldiag - q))

def correlation_cost(W, syn, y):
	return np.sum(W * np.dot(syn.T, y))

def cmds_cost(x, y):
	x_gram = np.dot(x, x.T)
	y_gram = np.dot(y, y.T)
	return np.sum(np.square(y_gram - x_gram))



np.random.seed(seed)

(batch_size, input_size), (_, output_size) = ds.train_shape

xv, yv = ds.next_train_batch()

W = np.random.random((input_size, layer_size))
W = W/(np.sum(W, 0)/p)

Wo = np.random.random((layer_size, output_size))
Wo = Wo/(np.sum(Wo, 0)/(p*15))

L = np.zeros((layer_size, layer_size))
Ldiag = np.ones((layer_size,))

D = np.ones((layer_size, layer_size)) * p
np.fill_diagonal(D, q)

epochs = 1
metrics = np.zeros((epochs, 7))
for e in xrange(epochs):
	yh = np.zeros((num_iters, batch_size, layer_size))
	lat_h = np.zeros((num_iters, batch_size, layer_size))
	fb_h = np.zeros((num_iters, batch_size, layer_size))
	ff_h = np.zeros((num_iters, batch_size, layer_size))

	y = np.zeros((batch_size, layer_size))
	syn = np.zeros((batch_size, input_size))
	osyn = np.zeros((batch_size, output_size))

	metrics_it = np.zeros((num_iters, 5))

	for t in xrange(num_iters):
		dsyn = np.zeros((batch_size, input_size))
		if t == t_data_spike:
			dsyn += xv
		
		dosyn = np.zeros((batch_size, output_size))
		if t == t_data_spike:
			dosyn += yv

		syn += dt * (dsyn - syn)
		osyn += dt * (dosyn - osyn)


		fb_ap = np.dot(osyn, Wo.T)
		ff = np.dot(syn, W) 
		fb_lat = np.dot(y, L)

		y += dt * (act((ff + fb_ap - fb_lat) / Ldiag) - y)

		dW = np.dot(syn.T, y) - k * (np.sum(W, axis=0) - p)
		dL = np.dot(y.T, y) - p * p
		dLdiag = np.sum(np.square(y), 0) - q * q

		W += lrate_ff * dW
		L += lrate_lat * dL
		Ldiag += lrate_lat * dLdiag

		np.fill_diagonal(L, 0.0)
		W = np.minimum(np.maximum(W, 0.0), omega)
		L = np.maximum(L, 0.0)
		Ldiag = np.maximum(Ldiag, 0.0)

		yh[t] = y.copy()
		lat_h[t] = fb_lat
		ff_h[t] = ff
		fb_h[t] = fb_ap

		metrics_it[t, :] = (
			correlation_cost(W, syn, y),
			phi_capital(W),
			lateral_cost(L, Ldiag),
			np.linalg.norm(ff - y),
			np.linalg.norm(fb_ap - y)
		)

	metrics[e, :5] = np.mean(metrics_it, 0)
	metrics[e, -2] = np.mean(
		np.not_equal(
			np.argmax(np.dot(yh[t_data_spike+1], Wo), 1), 
			np.argmax(yv, 1)
		)
	)
	metrics[e, -1] = cmds_cost(xv, yh[t_data_spike])
	if e % 10 == 0:
		print "Epoch {}, {}".format(
			e,
			", ".join(["{:.4f}".format(m) for m in metrics[e, :]])
		)