#!/usr/bin/env python

import numpy as np
from util import *
from poc.opt import *
from poc.common import *
from poc.model import *

np.random.seed(12)

# Setup
input_size = 2
layer_size = 25
output_size = 1
batch_size = 4

dt = 0.5
weight_factor = 0.2
num_iters = 100

act = lambda x: np.maximum(x, 0.0)

x = np.zeros((num_iters, batch_size, input_size))
x[50,0,:] = (0.0, 0.0)
x[50,1,:] = (0.0, 1.0)
x[50,2,:] = (1.0, 0.0)
x[50,3,:] = (1.0, 1.0)


y = np.zeros((num_iters, batch_size, output_size))
y[51,0,:] = (0.0,)
y[51,1,:] = (1.0,)
y[51,2,:] = (1.0,)
y[51,3,:] = (0.0,)

x = smooth_batch_matrix(x)
y = smooth_batch_matrix(y)


net = Net(
    Layer(
        num_iters=num_iters,
        batch_size=batch_size,
        input_size=input_size,
        layer_size=layer_size,
        output_size=output_size,
        act=act,
        weight_factor=weight_factor,
        dt=dt
    ),
    Layer(
        num_iters=num_iters,
        batch_size=batch_size,
        input_size=layer_size,
        layer_size=output_size,
        output_size=output_size,
        act=act,
        weight_factor=weight_factor,
        dt=dt
    ),
)



l0, l1 = net.layers

net[-1].Wfb = np.eye(*net[-1].Wfb.shape)
for li, l in enumerate(net.layers):
    if li < net.size-1:
        net[li].Wfb = net[li+1].W.T.copy()


for epoch in xrange(1):
    for t in xrange(num_iters): net.run(t, x[t], y[t], test=False)

    # net[1].dW = np.zeros(net[1].dW.shape)

    l0.W += 0.1 * l0.dW
    l1.W += 0.1 * l1.dW

    dW_norm = np.zeros(net.size)
    error_mean = np.zeros(net.size)
    for li, l in enumerate(net.layers):
        if li < net.size-1:
            net[li].Wfb = net[li+1].W.T.copy()

        dW_norm[li] = np.linalg.norm(l.dW)
        error_mean[li] = np.sum(l.eh)
        l.reset_state()


    if epoch % 100 == 0:
        l0_ah_train = l0.ah.copy()

        for t in xrange(num_iters): net.run(t, x[t], y[t], test=True)
        [l.reset_state() for li, l in enumerate(net.layers)]

        mse = np.linalg.norm(net[-1].ah - y)

        print "Epoch {}, {:.4f}, {}, {} | {:.4f}".format(
            epoch,
            mse,
            ",".join(["{:.4f}".format(em) for em in error_mean]),
            ",".join(["{:.4f}".format(dW_norm_l) for dW_norm_l in dW_norm]),
            np.linalg.norm(l0_ah_train - l0.ah)
        )


# for t in xrange(num_iters): net.run(t, x[t], y[t], test=True)
# aht = l0.ah.copy()

