#!/usr/bin/env python

import numpy as np
from util import *
from datasets import *
from sklearn.metrics import log_loss

np.random.seed(11)

# Setup
layer_size = 10
weight_factor = 0.2
input_size = 10
output_size = 5
batch_size = 100

# act = lambda x: np.maximum(x, 0.0)


act = tanh
# w0 = 0.5*np.random.random((input_size, layer_size)) - 0.25
# w1 = 0.5*np.random.random((layer_size, output_size)) - 0.25
#
# x = 0.5*np.random.random((batch_size, input_size)) - 0.25
# y = act(np.dot(act(np.dot(x, w0)), w1))


# x_v, target_v = get_toy_data_baseline()
# y_v = one_hot_encode(target_v)
#
# test_prop = x_v.shape[0] / 5

x = np.asarray([
    [-1.0, -1.0],
    [-1.0, 1.0],
    [1.0, -1.0],
    [1.0, 1.0]
])

y = np.asarray([
    [-1.0],
    [1.0],
    [1.0],
    [-1.0],
])


xt, yt = x, y

input_size = x.shape[1]
output_size = y.shape[1]

def weights(xsize, ysize):
    return weight_factor * np.random.random((xsize, ysize)) - weight_factor / 2.0


net = [
    [weights(input_size, layer_size), weights(layer_size, layer_size)],
    [weights(layer_size, output_size), np.eye(output_size, output_size)],
]



for li in xrange(len(net)):
    if li < len(net)-1:
        net[li][1] = net[li+1][0].T.copy()

def iterate(x_target, fb0_prev=None, y_target=None):
    (l0_W, l0_Wfb), (l1_W, l1_Wfb) = net[0], net[1]

    ff0 = np.dot(x_target, l0_W)
    a0_pre = act(
        ff0 if fb0_prev is None else (ff0 + fb0_prev)/2.0
    )

    ff1 = np.dot(a0_pre, l1_W)

    if not y_target is None:
        fb1 = np.dot(y_target, l1_Wfb)
        a1 = act((ff1 + fb1)/2.0)
    else:
        fb1 = None
        a1 = act(ff1)

    fb0 = np.dot(a1, l0_Wfb)
    a0 = act((ff0 + fb0)/2.0)

    return a0, a1, ff0, ff1, fb0, fb1



for e in xrange(5000):
    fb0 = None
    for i in xrange(5):
        a0, a1, ff0, ff1, fb0, fb1 = iterate(x, fb0, y_target=y)

    batch_size = x.shape[0]
    net[0][0] += 0.5 * np.dot(x.T, fb0 - ff0) / batch_size
    net[1][0] += 0.5 * np.dot(a0.T, fb1 - ff1) / batch_size

    for li in xrange(len(net)):
        if li < len(net) - 1:
            net[li][1] = net[li + 1][0].T.copy()

    if e % 1000 == 0:
        a1train = a1.copy()
        fb0 = None
        for i in xrange(5):
            a0, a1, ff0, ff1, fb0, fb1 = iterate(xt, fb0, y_target=None)

        print "{}, {:.4f} |\n\t{:.3f} {:.3f} {:.3f} {:.3f}\n\t{:.3f} {:.3f} {:.3f} {:.3f}".format(
            e,
            np.linalg.norm(a1 - yt),
            a1train[0, 0], a1train[1, 0], a1train[2, 0], a1train[3, 0],
            a1[0,0], a1[1,0], a1[2,0], a1[3,0]
        )


