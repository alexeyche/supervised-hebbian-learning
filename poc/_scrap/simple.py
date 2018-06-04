#!/usr/bin/env python

import numpy as np
from util import *
from poc.opt import *
from poc.common import *

np.random.seed(10)

# Setup
batch_size = 10
input_size = 15
layer0_size = 30
layer1_size = 15
output_size = 5
dt = 0.5
weight_factor = 0.2
num_iters = 500

c0 = np.sin(np.linspace(0, 20.0, num_iters) + 0.25*np.pi)
c1 = np.sin(np.linspace(0, 20.0, num_iters))

c = np.asarray(((c0,), (c1,)))
c = np.transpose(c, (2, 1, 0))
_, batch_size, input_size = c.shape



F0 = weight_factor*np.random.random((input_size, layer0_size)) - weight_factor/2.0
F1 = weight_factor*np.random.random((layer0_size, layer1_size)) - weight_factor/2.0
D = weight_factor*np.random.random((layer1_size, input_size)) - weight_factor/2.0


for epoch in xrange(1):
	y0 = np.zeros((batch_size, layer0_size))
	y1 = np.zeros((batch_size, layer1_size))

	y0h = np.zeros((num_iters, batch_size, layer0_size))
	y1h = np.zeros((num_iters, batch_size, layer1_size))

	e0h = np.zeros((num_iters, 1))
	e1h = np.zeros((num_iters, 1))

	x_rec_h = np.zeros((num_iters, batch_size, input_size))

	dF0 = np.zeros(F0.shape)
	dF1 = np.zeros(F1.shape)

	for t in xrange(num_iters):
		x = c[t]

		x_rec = np.dot(y1, D)
		error1_proj = x - x_rec
		error1 = np.dot(error1_proj, D.T)

		error0 = np.dot(y1, F1.T)

		y0 += dt * error0 #np.dot(error0, F0)
		y1 += dt * error1

		# dF1 += np.dot(error1.T, y1) / num_iters

		y0h[t] = y0.copy()
		y1h[t] = y1.copy()
		e0h[t] = np.linalg.norm(error0)
		e1h[t] = np.linalg.norm(error1)
		x_rec_h[t] = x_rec.copy()

	F1 += 0.1 * dF1
	if epoch % 100 == 0:
		print "Epoch {}, error {:.4f}, |dF| {:.4f}".format(
			epoch, 
			np.linalg.norm(e1h), 
			np.linalg.norm(dF1)
		)