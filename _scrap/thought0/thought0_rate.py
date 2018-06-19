#!/usr/bin/env python

import numpy as np
from util import *
from poc.opt import *
from poc.common import *
from poc.model import *

np.random.seed(12)

# Setup
input_size = 25
layer_size = 100
output_size = 10
batch_size = 5

dt = 0.5
weight_factor = 0.2
num_iters = 100

act = lambda x: np.maximum(x, 0.0)

w0 = 0.5*np.random.random((input_size, layer_size)) - 0.25
w1 = 0.5*np.random.random((layer_size, output_size)) - 0.25

x = 0.5*np.random.random((batch_size, input_size)) - 0.25
y = act(np.dot(act(np.dot(x, w0)), w1))



# act = lambda x: np.maximum(1.0/(1.0 + np.exp(-x)) - 0.5, 0.0)

net = Net(
    Layer(
        num_iters=num_iters,
        batch_size=batch_size,
        input_size=input_size,
        layer_size=layer_size,
        output_size=layer_size,
        act=act,
        weight_factor=weight_factor,
        dt=dt
    ),
    Layer(
        num_iters=num_iters,
        batch_size=batch_size,
        input_size=layer_size,
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



l0, l1, l2 = net.layers

net[-1].Wfb = np.eye(*net[-1].Wfb.shape)
for li, l in enumerate(net.layers):
    if li < net.size-1:
        net[li].Wfb = net[li+1].W.T.copy()


for epoch in xrange(1):
    for t in xrange(num_iters): net.run(t, x, y, test=False)

    # net[1].dW = np.zeros(net[1].dW.shape)

    l0.W += 0.1 * l0.dW
    l1.W += 0.1 * l1.dW
    l2.W += 0.1 * l2.dW

    dW_norm = np.zeros(net.size)
    error_mean = np.zeros(net.size)
    for li, l in enumerate(net.layers):
        if li < net.size-1:
            net[li].Wfb = net[li+1].W.T.copy()

        dW_norm[li] = np.linalg.norm(l.dW)
        error_mean[li] = np.sum(l.eh)
        l.reset_state()


    if epoch % 100 == 0:
        for t in xrange(num_iters): net.run(t, x, y, test=True)
        [l.reset_state() for li, l in enumerate(net.layers)]

        mse = np.linalg.norm(net[-1].ah[-1] - y)

        print "Epoch {}, {:.4f}, {}, {}".format(
            epoch,
            mse,
            ",".join(["{:.4f}".format(em) for em in error_mean]),
            ",".join(["{:.4f}".format(dW_norm_l) for dW_norm_l in dW_norm]),
        )


