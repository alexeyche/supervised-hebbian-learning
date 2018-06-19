#!/usr/bin/env python

import numpy as np
from util import *
from datasets import *
from sklearn.metrics import log_loss

np.random.seed(12)

# Setup
layer_size = 100
weight_factor = 0.2

act = lambda x: np.maximum(x, 0.0)

# w0 = 0.5*np.random.random((input_size, layer_size)) - 0.25
# w1 = 0.5*np.random.random((layer_size, output_size)) - 0.25
#
# x = 0.5*np.random.random((batch_size, input_size)) - 0.25
# y = act(np.dot(act(np.dot(x, w0)), w1))


x_v, target_v = get_toy_data_baseline()
y_v = one_hot_encode(target_v)

test_prop = x_v.shape[0] / 5

xt = x_v[:test_prop]
yt = y_v[:test_prop]

x = x_v[test_prop:]
y = y_v[test_prop:]

input_size = x.shape[1]
output_size = y.shape[1]

def weights(xsize, ysize):
    return weight_factor * np.random.random((xsize, ysize)) - weight_factor / 2.0


net = [
    [weights(input_size, layer_size), weights(layer_size, layer_size)],
    [weights(layer_size, layer_size), weights(layer_size, layer_size)],
    [weights(layer_size, output_size), np.eye(output_size, output_size)],
]



for li in xrange(len(net)):
    if li < len(net)-1:
        net[li][1] = net[li+1][0].T.copy()

def iterate(x_target, fb0_prev=None, fb1_prev=None, y_target=None):
    (l0_W, l0_Wfb), (l1_W, l1_Wfb), (l2_W, l2_Wfb) = net[0], net[1], net[2]

    ff0 = np.dot(x_target, l0_W)
    a0_pre = act(
        ff0 if fb0_prev is None else (ff0 + fb0_prev)/2.0
    )

    ff1 = np.dot(a0_pre, l1_W)

    a1_pre = act(
        ff1 if fb1_prev is None else (ff1 + fb1_prev)/2.0
    )

    ff2 = np.dot(a1_pre, l2_W)

    if not y_target is None:
        fb2 = np.dot(y_target, l2_Wfb)
        a2 = act((ff2 + fb2)/2.0)
    else:
        fb2 = None
        a2 = act(ff2)

    fb1 = np.dot(a2, l1_Wfb)
    a1 = act((ff1 + fb1)/2.0)

    fb0 = np.dot(a1, l0_Wfb)
    a0 = act((ff0 + fb0)/2.0)

    return a0, a1, a2, ff0, ff1, ff2, fb0, fb1, fb2



for e in xrange(10000):
    fb0, fb1 = None, None
    for i in xrange(5):
        a0, a1, a2, ff0, ff1, ff2, fb0, fb1, fb2 = iterate(x, fb0, fb1, y_target=y)

    batch_size = x.shape[0]
    net[0][0] += 0.05 * np.dot(x.T, fb0 - ff0) / batch_size
    net[1][0] += 0.05 * np.dot(a0.T, fb1 - ff1) / batch_size
    net[2][0] += 0.05 * np.dot(a1.T, fb2 - ff2) / batch_size

    for li in xrange(len(net)):
        if li < len(net) - 1:
            net[li][1] = net[li + 1][0].T.copy()

    if e % 100 == 0:
        fb0, fb1 = None, None
        for i in xrange(5):
            a0, a1, a2, ff0, ff1, ff2, fb0, fb1, fb2 = iterate(xt, fb0, fb1, y_target=None)

        print e, np.linalg.norm(a2 - yt), np.mean(np.not_equal(np.argmax(a2, 1), np.argmax(yt, 1)))


