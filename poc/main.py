

import numpy as np
from poc.opt import *
from datasets import *
from poc.common import *
from cost import *


ds = ToyDataset()
act = Relu()

seed = 10
layer_size = 100
dt = 0.25
omega = 1.0
k = 1.0
num_iters = 50

p, q = 0.1, 0.1



np.random.seed(seed)

(batch_size, input_size), (_, output_size) = ds.train_shape

xv, yv = ds.next_train_batch()

yv *= 0.1

W = np.random.random((input_size, layer_size))
W = W/(np.sum(W, 0)/p)

Wo = np.random.random((layer_size, output_size))
Wo = Wo/(np.sum(Wo, 0)/p)

L = np.zeros((layer_size, layer_size))
Ldiag = np.ones((layer_size,))

Lo = np.zeros((output_size, output_size))
Lodiag = np.ones((output_size,))


D = np.ones((layer_size, layer_size)) * p
np.fill_diagonal(D, q)

opt = SGDOpt((
	0.001, 0.1, 0.01, 
	0.0001, 0.1, 0.01
))
opt.init(W, L, Ldiag, Wo, Lo, Lodiag)


epochs = 1500
metrics = np.zeros((epochs, 7))
Ch = np.zeros((epochs, input_size, layer_size))
for e in xrange(epochs):
	yh = np.zeros((num_iters, batch_size, layer_size))
	yoh = np.zeros((num_iters, batch_size, output_size))
	lat_h = np.zeros((num_iters, batch_size, layer_size))
	fb_h = np.zeros((num_iters, batch_size, layer_size))
	ff_h = np.zeros((num_iters, batch_size, layer_size))

	y = np.zeros((batch_size, layer_size))
	yo = np.zeros((batch_size, output_size))
	syn = np.zeros((batch_size, input_size))
	osyn = np.zeros((batch_size, output_size))

	metrics_it = np.zeros((num_iters, 5))

	fb_ap = np.zeros((batch_size, layer_size)) #
	
	ff = np.dot(xv, W) 
	ff = ff/np.linalg.norm(ff)

	fb_ap = np.dot(yv, Wo.T)
	fb_ap = fb_ap/np.linalg.norm(fb_ap)

	for t in xrange(num_iters):	
		y += dt * (act(( (ff + fb_ap)/7.0 - np.dot(y, L)) / Ldiag) - y)
		
		ff_yo = np.dot(y, Wo)
		ff_yo = ff_yo/np.linalg.norm(ff_yo)
		
		fb_yo = yv
		fb_yo = fb_yo/np.linalg.norm(fb_yo)
		
		yo += dt * (act((ff_yo - np.dot(yo, Lo))/ Lodiag) - yo)

		yh[t] = y.copy()
		yoh[t] = yo.copy()


	dW = np.dot(xv.T, y) - k * (np.sum(W, axis=0) - p)
	dL = np.dot(y.T, y) - p * p
	dLdiag = np.sum(np.square(y), 0) - q * q

	dWo = np.dot(y.T, yo) - Wo #k * (np.sum(Wo, axis=0) - p)
	# dWo = np.dot(y.T, yv) - Wo #k * (np.sum(Wo, axis=0) - p)
	dLo = np.dot(yo.T, yo) - np.eye(output_size)
	dLodiag = np.sum(np.square(yv), 0) - q * q

	opt.update(-dW, -dL, -dLdiag, -dWo, -dLo, -dLodiag)

	np.fill_diagonal(L, 0.0)
	W = np.minimum(np.maximum(W, 0.0), omega)
	L = np.maximum(L, 0.0)
	Ldiag = np.maximum(Ldiag, 0.0)


	np.fill_diagonal(Lo, 0.0)
	Wo = np.minimum(np.maximum(Wo, 0.0), omega)
	Lo = np.maximum(Lo, 0.0)
	Lodiag = np.maximum(Lodiag, 0.0)

	opt.init(W, L, Ldiag, Wo, Lo, Lodiag)

	xtv, ytv = ds.next_test_batch()

	yt = np.zeros((xtv.shape[0], layer_size))
	yto = np.zeros((xtv.shape[0], output_size))

	fft = np.dot(xtv, W)
	
	for t in xrange(num_iters):
		yt += dt * (act((fft - np.dot(yt, L)) / Ldiag) - yt)
		yto += dt * (act((np.dot(yt, Wo) - np.dot(yto, Lo))/ Lodiag) - yto)


	metrics[e, :5] = (
		correlation_cost(W, xv, y),
		phi_capital(W, p, k),
		lateral_cost(L, Ldiag, p, q),
		np.linalg.norm(ff - y),
		np.linalg.norm(fb_ap - y)
	)
	metrics[e, -2] = np.mean(
		np.not_equal(
			np.argmax(yto, 1), 
			np.argmax(ytv, 1)
		)
	)
	metrics[e, -1] = cmds_cost(xv, y)
	Ch[e] = np.dot(xv.T, y)
	
	if e % 10 == 0:

		print "Epoch {}, {}".format(
			e,
			", ".join(["{:.4f}".format(m) for m in metrics[e, :]])
		)
