#!/usr/bin/env python

import numpy as np
from util import *
from poc.opt import *
from poc.common import *

np.random.seed(10)

# Setup
batch_size = 1
input_size = 1
layer_size = 30
output_size = 1
dt = 0.5
weight_factor = 1.0
num_iters = 500



xt = np.expand_dims(np.expand_dims(generate_ts(num_iters, params=(0.1, 0.1, 3, 5, 6)), 1), 1)
yt = np.expand_dims(np.expand_dims(generate_ts(num_iters, params=(0.1, 0.1, 6, 2, 1)), 1), 1)





F = weight_factor*np.random.random((input_size, layer_size)) - weight_factor/2.0
Fo = weight_factor*np.random.random((layer_size, output_size)) - weight_factor/2.0


for epoch in xrange(1):
	u = np.zeros((batch_size, layer_size))
	uh = np.zeros((num_iters, batch_size, layer_size))
	
	y_rec_h = np.zeros((num_iters, batch_size, output_size))

	eh = np.zeros((num_iters, 1))

	dF = np.zeros(F.shape)

	for t in xrange(num_iters):
		u += dt * (np.dot(xt[t], F) - u)

		y_rec = np.dot(u, Fo)
		error = yt[t] - y_rec

		dUdE = np.dot(error, Fo.T)
		dF += np.dot(xt[t].T, dUdE) / num_iters

		uh[t] = u.copy()
		eh[t] = np.linalg.norm(error)
		y_rec_h[t] = y_rec.copy()

	F += 0.1 * dF
	if epoch % 100 == 0:
		print "Epoch {}, error {:.4f}, |dF| {:.4f}".format(
			epoch, 
			np.linalg.norm(eh), 
			np.linalg.norm(dF)
		)