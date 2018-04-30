#!/usr/bin/env python

import numpy as np
from util import *
from poc.opt import *
from poc.common import *

np.random.seed(10)

# Setup
batch_size = 10
input_size = 15
layer_size = 30
output_size = 5
dt = 0.5
weight_factor = 0.2
num_iters = 500

c0 = np.sin(np.linspace(0, 20.0, num_iters) + 0.25*np.pi)
c1 = np.sin(np.linspace(0, 20.0, num_iters))

c = np.asarray(((c0,), (c1,)))
c = np.transpose(c, (2, 1, 0))
_, batch_size, input_size = c.shape



F = weight_factor*np.random.random((input_size, layer_size)) - weight_factor/2.0


for epoch in xrange(1000):
	y = np.zeros((batch_size, layer_size))
	yh = np.zeros((num_iters, batch_size, layer_size))
	x_rec_h = np.zeros((num_iters, batch_size, input_size))
	eh = np.zeros((num_iters, 1))

	dF = np.zeros(F.shape)

	for t in xrange(num_iters):
		x = c[t]

		x_rec = np.dot(y, F.T)
		error = x - x_rec
		y += dt * np.dot(error, F)


		dF += np.dot(error.T, y) / num_iters

		yh[t] = y.copy()
		eh[t] = np.linalg.norm(error)
		x_rec_h[t] = x_rec.copy()

	F += 0.1 * dF
	if epoch % 100 == 0:
		print "Epoch {}, error {:.4f}, |dF| {:.4f}".format(
			epoch, 
			np.linalg.norm(eh), 
			np.linalg.norm(dF)
		)