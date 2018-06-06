#!/usr/bin/env python

import numpy as np
from util import *
from poc.opt import *
from poc.common import *
from poc.model import *

np.random.seed(12)

# Setup
input_size = 100
layer_size = 50
output_size = 1
batch_size = 1

dt = 0.5
weight_factor = 0.1
num_iters = 100


x = np.zeros((num_iters, batch_size, input_size))

x_isi = 10
for ni in xrange(0, num_iters, x_isi):
    x[ni, 0, ni % input_size] = 1.0
y = np.zeros((num_iters, batch_size, output_size))
y[(25, 50 ,75),0,0] = 1.0

y = smooth_batch_matrix(y, sigma=0.005)
x = smooth_batch_matrix(x, sigma=0.005)

# act = lambda x: np.maximum(1.0/(1.0 + np.exp(-x)) - 0.5, 0.0)
act = lambda x: x

net = Net(
    Layer(
        num_iters,
        batch_size,
        input_size,
        layer_size,
        output_size,
        act=act,
        weight_factor=weight_factor,
        dt=dt
    ),
    Layer(
        num_iters,
        batch_size,
        layer_size,
        output_size,
        output_size,
        act=act,
        weight_factor=weight_factor,
        dt=dt
    )
)



net[0].Wfb = net[1].W.T.copy()
l0, l1 = net.layers

l1.Wfb[0,0] = 0.2

for epoch in xrange(10000):
    for t in xrange(num_iters): net.run(t, x[t], y[t], 1.0)

    # net[1].dW = np.zeros(net[1].dW.shape)

    dW_norm = np.zeros(net.size)

    l0.W += 0.05 * l0.dW
    l1.W += 0.1 * l1.dW

    for li, l in enumerate(net.layers):
        if li < net.size-1:
            net[li].Wfb = net[li+1].W.T.copy()

        dW_norm[li] = np.linalg.norm(l.dW)
        l.reset_state()



    if epoch % 100 == 0:
        # for t in xrange(num_iters): net.run(t, x[t], y[t], 0.0)
        # [l.reset_state() for li, l in enumerate(net.layers)]

        mse = np.linalg.norm(net[-1].ah - y)

        print "Epoch {}, {:.4f}, {}, {}".format(
            epoch,
            mse,
            ",".join(["{:.4f}".format(np.sum(l.eh)) for l in net.layers]),
            ",".join(["{:.4f}".format(dW_norm_l) for dW_norm_l in dW_norm]),
        )

