#!/usr/bin/env python

import numpy as np
from util import *
from poc.opt import *
from poc.common import *

np.random.seed(17)


# Setup
batch_size = 10
layer_size = 30
output_size = 5
dt = 1.0
weight_factor = 0.5

num_iters = 500

c0 = generate_ts(num_iters, params=(0.1,0.3,10,10,4))
c1 = generate_ts(num_iters, params=(0.01,0.1,1,2,3))

c = np.asarray(((c0,), (c1,)))
c = np.transpose(c, (2, 1, 0))
_, batch_size, input_size = c.shape

lam = 0.1
nu = 0.0

F = weight_factor*np.random.random((input_size, layer_size)) - weight_factor/2.0
Om = weight_factor*np.random.random((layer_size, layer_size)) - weight_factor/2.0

np.fill_diagonal(Om, -1.0)
thr = (-np.diag(Om) + nu)


rh = np.zeros((num_iters, batch_size, layer_size))
oh = np.zeros((num_iters, batch_size, layer_size))
Vh = np.zeros((num_iters, batch_size, layer_size))
x_rec_h = np.zeros((num_iters, batch_size, input_size))


x, r, o, xr, rr = 5*(None,)


def reset_state():
	global x, r, o, xr, rr

	x = np.zeros((batch_size, input_size))
	r = np.zeros((batch_size, layer_size))
	o = np.zeros((batch_size, layer_size))

	xr = np.zeros((input_size, layer_size))
	rr = np.zeros((layer_size, layer_size))

def run_dynamics(t):
	global x, r, xr, rr

	x += dt * ( - lam * x + c[t])

	V  = np.dot(x, F) + np.dot(r, Om)
	o = np.zeros((batch_size, layer_size))
	o[np.where(V >= thr)] = 1.0
	
	r += dt * ( - lam * r + o)

	xr += np.dot(x.T, r)/num_iters
	rr += np.dot(r.T, r)/num_iters 

	rh[t] = r.copy()
	oh[t] = o.copy()
	Vh[t] = V.copy()
	x_rec_h[t] = np.dot(r, F.T)


reset_state()
for t in xrange(num_iters): run_dynamics(t)


D = np.dot(xr, np.linalg.inv(rr + np.eye(layer_size) * 0.001))

F = D
Om = -np.dot(D.T, D)

reset_state()
for t in xrange(num_iters): run_dynamics(t)

shl(Vh[:,0,3], oh[:,0,3])
shl(x_rec_h[:,0,1]/(1.0/lam), c[:,0,1])