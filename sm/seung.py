

import numpy as np
from poc.opt import *
from datasets import *
from poc.common import *
from sm.sm_cost import *

ds = ToyDataset()
act = Relu()

seed = 5
layer_size = 25
dt = 0.25
omega = 0.1
k = 1.0
num_iters = 25

p, q = 0.05, 0.09
lrate_lat = 0.05
lrate_ff = 0.001
t_data_spike = 5



np.random.seed(seed)

(batch_size, input_size), (_, output_size) = ds.train_shape

xv, yv = ds.next_train_batch()

W = np.random.random((input_size, layer_size))
W = W/(np.sum(W, 0)/p)

L = np.zeros((layer_size, layer_size))
Ldiag = np.ones((layer_size,))

D = np.ones((layer_size, layer_size)) * p
np.fill_diagonal(D, q)

epochs = 1000
metrics = np.zeros((epochs, 4))
for e in xrange(epochs):
	yh = np.zeros((num_iters, batch_size, layer_size))
	lat_h = np.zeros((num_iters, batch_size, layer_size))
	ff_h = np.zeros((num_iters, batch_size, layer_size))

	y = np.zeros((batch_size, layer_size))
	syn = np.zeros((batch_size, input_size))

	metrics_it = np.zeros((num_iters, 3))

	for t in xrange(num_iters):
		dsyn = np.zeros((batch_size, input_size))
		dy = np.zeros((batch_size, layer_size))

		if t == t_data_spike:
			dsyn += xv

		syn += dt * (dsyn - syn)

		ff = np.dot(syn, W)
		fb = np.dot(y, L)

		y += dt * (act((ff - fb) / Ldiag) - y)

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
		lat_h[t] = fb
		ff_h[t] = ff

		metrics_it[t, :] = (
			correlation_cost(W, syn, y),
			phi_capital(W, p, k),
			lateral_cost(L, Ldiag, p, q)
		)

	metrics[e, :3] = np.mean(metrics_it, 0)
	metrics[e, 3] = cmds_cost(xv, yh[t_data_spike])
	if e % 100 == 0:
		print "Epoch {}, {}".format(
			e,
			", ".join(["{:.4f}".format(m) for m in metrics[e, :]])
		)