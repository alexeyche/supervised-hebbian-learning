#!/usr/bin/env python

import numpy as np
from util import *
from poc.opt import *
from poc.common import *
from poc.model import *
import numpy as np
from sklearn.datasets import make_circles

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


np.random.seed(4)

x, y = make_circles(n_samples=200, factor=.3, noise=.05)
y = one_hot_encode(y, 2)

batch_size = x.shape[0]
input_size = x.shape[1]
output_size = y.shape[1]

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


for epoch in xrange(2000):
    for t in xrange(num_iters): net.run(t, x, y, test=False)

    # net[1].dW = np.zeros(net[1].dW.shape)

    l0.W += 0.005 * l0.dW
    l1.W += 0.005 * l1.dW
    l2.W += 0.005 * l2.dW

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

        print "Epoch {}, {:.4f}, {}, {}, {:.4f}".format(
            epoch,
            mse,
            ",".join(["{:.4f}".format(em) for em in error_mean]),
            ",".join(["{:.4f}".format(dW_norm_l) for dW_norm_l in dW_norm]),
            np.mean(np.not_equal(np.argmax(net[-1].ah[-1], 1), np.argmax(y, 1)))
        )


