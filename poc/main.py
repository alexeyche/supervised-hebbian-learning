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
dt = 0.1

## Act
psi_max = 1.0
k = 0.5
beta = 5.0
thr = 1.0
act = lambda x: psi_max / (1.0 + k * np.exp(beta * (thr - x)))
act_deriv = lambda x, a: a * (psi_max - a)

num_iters = 100

# Const

gL = 2.0
gD = 5.0
Ee = 4.66
Ei = - 1.0/3.0
tau_l = 5.0
tau_s = 2.0

data = np.zeros((num_iters, batch_size, input_size))
for ni in xrange(0, num_iters, 5):
	data[ni, :, (ni/7) % input_size] = 1.0

gI_nudge = np.ones((num_iters, batch_size, output_size)) * 2.0
gE_nudge = np.zeros((num_iters, batch_size, output_size)) 
gE_nudge[(25, 50, 75), :, :] = 1.0
gE_nudge = smooth_batch_matrix(gE_nudge)

class Layer(object):
	def __init__(s, batch_size, input_size, layer_size, weight_factor = 1.0):	
		s.batch_size, s.input_size, s.layer_size = batch_size, input_size, layer_size

		s.W = weight_factor*np.random.random((input_size, layer_size)) - weight_factor/2.0
		s.reset_state()

		s.Uh = np.zeros((num_iters, batch_size, layer_size))
		s.Ah = np.zeros((num_iters, batch_size, layer_size))
		s.Vwh = np.zeros((num_iters, batch_size, layer_size))
		s.Vw_starh = np.zeros((num_iters, batch_size, layer_size))
		s.Isomh = np.zeros((num_iters, batch_size, layer_size))
		s.Eh = np.zeros((num_iters, 1))

	def reset_state(s):
		s.U = np.zeros((s.batch_size, s.layer_size))
		s.A = np.zeros((s.batch_size, s.layer_size))
		s.Vw = np.zeros((s.batch_size, s.layer_size))
		s.dW = np.zeros(s.W.shape)

	def run(s, t, x, gE_t, gI_t):
		s.Vw += dt * ( - s.Vw + np.dot(x, s.W)) / tau_s

		Isom = gE_t * (Ee - s.U) + gI_t * (Ei - s.U)
		
		dU = - gL * s.U + gD * (s.Vw - s.U) + Isom
		s.U += dt * dU

		s.A = act(s.U)

		Vw_star = (gD/(gD + gL)) * s.Vw
		
		E = (s.A - act(Vw_star))
		dAdE = E * act_deriv(s.U, s.A)
		
		s.dW += np.dot(x.T, dAdE)

		s.Eh[t] = np.sum(E)
		s.Uh[t] = s.U.copy()
		s.Ah[t] = s.A.copy()
		s.Vwh[t] = s.Vw.copy()
		s.Isomh[t] = Isom.copy()
		s.Vw_starh[t] = Vw_star.copy()
		# s.Idndh[t] = s.Idnd.copy()


net = [
	Layer(batch_size, input_size, layer_size),
	Layer(batch_size, layer_size, output_size)
]

def net_run(t, data, gE_nudge, gI_nudge):
	for li, l in enumerate(net):
		gE_nudge_l = np.dot(net[li+1].A, net[li+1].W.T) if li < len(net)-1 else gE_nudge
		gI_nudge_l = 2.0
		data_l = data if li == 0 else net[li-1].A
		
		l.run(t, data_l, gE_nudge_l, gI_nudge_l)


for e in xrange(1000):
	for t in xrange(num_iters): net_run(t, data[t], gE_nudge[t], gI_nudge[t])
	
	for l in net:
		l.W += 0.05 * l.dW
		l.reset_state()

	if e % 100 == 0:		
		print "{}, {:.4f}".format(e, np.linalg.norm(net[-1].Eh))


shl(act(net[0].Vw_starh[:,0,0]), net[0].Ah[:,0,0])

# for t in xrange(num_iters):	net_run(t, data[t], 0.0, 0.0)
