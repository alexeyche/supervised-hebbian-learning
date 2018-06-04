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
dt = 0.2

num_iters = 100

tau_syn = 5.0
tau_soma = 1.0

act = Relu()

class Layer(object):
    def __init__(s, batch_size, input_size, layer_size, output_size, weight_factor = 1.0):
        s.batch_size, s.input_size, s.layer_size = batch_size, input_size, layer_size

        s.W = weight_factor*np.random.random((input_size, layer_size)) - weight_factor/2.0
        
        s.Wfb = weight_factor*np.random.random((output_size, layer_size)) - weight_factor/2.0
        
        Morig = weight_factor*np.random.random((layer_size, 2)) - weight_factor/2.0
        
        s.M = np.dot(Morig, Morig.T)

        s.reset_state()

        s.Uh = np.zeros((num_iters, batch_size, layer_size))
        s.Ah = np.zeros((num_iters, batch_size, layer_size))
        s.Eh = np.zeros((num_iters, 3))
    
    def reset_state(s):
        s.U = np.zeros((s.batch_size, s.layer_size))
        s.A = np.zeros((s.batch_size, s.layer_size))
        s.dW = np.zeros(s.W.shape)
        s.dWfb = np.zeros(s.Wfb.shape)
        s.dM = np.zeros(s.M.shape)

    def run(s, t, x, y=None):
        Ix = np.dot(x, s.W)
        Irec = np.dot(s.A, s.M)
        Iy = np.dot(y, s.Wfb)

        dU = (Ix - s.U) + (Iy - s.U) + (Irec - s.U)

        s.U += dt * dU / tau_soma

        s.A = act(s.U)

        s.dW += np.dot(x.T, (s.A - act(Ix)) * act.deriv(s.U))
        s.dWfb += np.dot(y.T, (s.A - act(Iy)) * act.deriv(s.U))
        s.dM += np.dot(s.A.T, (s.A - act(Irec)) * act.deriv(s.U))

        s.Eh[t] = (np.linalg.norm(Ix - s.U), np.linalg.norm(Iy - s.U), np.linalg.norm(Irec - s.U))
        s.Uh[t] = s.U.copy()
        s.Ah[t] = s.A.copy()


net = [
    Layer(batch_size, input_size, layer_size, output_size),
    Layer(batch_size, layer_size, output_size, output_size)
]


l0, l1 = net

l0.Wfb = l1.W.T.copy()

x = np.zeros((num_iters, batch_size, input_size))
for ni in xrange(0, num_iters, 5):
    x[ni, :, (ni/7) % input_size] = 1.0

y = np.zeros((num_iters, batch_size, output_size))
y[(25, 50, 75), :, :] = 1.0

def net_run(t, x, y):
    for li, l in enumerate(net):
        y_l = net[li+1].A if li < len(net)-1 else y # np.zeros((batch_size, output_size))

        x_l = x if li == 0 else net[li-1].A

        l.run(t, x_l, y_l)

for e in xrange(200):
    for t in xrange(num_iters): net_run(t, x[t], y[t])

    for l in net:
        l.W += 0.001*l.dW / num_iters
        l.M += 0.001*l.dM / num_iters
        l.Wfb += 0.001*l.dWfb / num_iters

    # l0.Wfb = l1.W.T.copy()

    l.reset_state()

    print e, np.linalg.norm(l0.Eh), np.linalg.norm(l1.Eh)