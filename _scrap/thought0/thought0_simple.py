#!/usr/bin/env python

import numpy as np
from util import *
from poc.opt import *
from poc.common import *
from poc.model import *

np.random.seed(12)

# Setup
input_size = 1
layer_size = 1
output_size = 1
batch_size = 1

dt = 0.5
weight_factor = 0.1
num_iters = 100


# x = np.transpose(np.asarray(((np.sin(np.linspace(0, 20.0, num_iters) + 0.25*np.pi),), )), (2,1,0))
# y = np.transpose(np.asarray(((np.sin(np.linspace(0, 20.0, num_iters)),), )), (2,1,0))

x = 0.25 * np.ones((num_iters, batch_size, input_size))
y = 0.75 * np.ones((num_iters, batch_size, input_size))



# act = lambda x: np.maximum(1.0/(1.0 + np.exp(-x)) - 0.5, 0.0)
act = lambda x: x

net = Net(
    Layer(
        num_iters,
        batch_size,
        input_size,
        output_size,
        output_size,
        act=act,
        weight_factor=weight_factor,
        dt=dt
    ),
)


l = net[0]

l.W[0,0] = 0.2
l.Wfb[0,0] = 1.0

for epoch in xrange(2500):
    for t in xrange(num_iters): net.run(t, x[t], y[t], 1.0)

    # net[1].dW = np.zeros(net[1].dW.shape)

    dW_norm = np.zeros(net.size)

    l.W += 0.1 * l.dW

    for li, l in enumerate(net.layers):
        if li < net.size-1:
            net[li].Wfb = net[li+1].W.T.copy()

        dW_norm[li] = np.linalg.norm(l.dW)
        l.reset_state()



    if epoch % 10 == 0:
        for t in xrange(num_iters): net.run(t, x[t], y[t], 0.0)
        [l.reset_state() for li, l in enumerate(net.layers)]

        mse = np.linalg.norm(net[-1].ah[-1] - y)

        print "Epoch {}, {:.4f}, {}, {}".format(
            epoch,
            mse,
            ",".join(["{:.4f}".format(np.sum(l.eh)) for l in net.layers]),
            ",".join(["{:.4f}".format(dW_norm_l) for dW_norm_l in dW_norm]),
        )

