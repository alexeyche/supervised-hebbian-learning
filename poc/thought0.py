#!/usr/bin/env python

import numpy as np
from util import *
from poc.opt import *
from poc.common import *

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

class Layer(object):
    def __init__(
            s,
            num_iters,
            batch_size,
            input_size,
            layer_size,
            output_size,
            act,
            weight_factor,
            dt
    ):

        s.num_iters = num_iters
        s.layer_size = layer_size
        s.batch_size = batch_size
        s.output_size = output_size
        s.input_size = input_size
        s.dt = dt
        s.act = act

        s.W = weight_factor * np.random.random((s.input_size, s.layer_size)) - weight_factor / 2.0
        s.Wfb = weight_factor * np.random.random((s.output_size, s.layer_size)) - weight_factor / 2.0

        s.uh = np.zeros((s.num_iters, s.batch_size, s.layer_size))
        s.ah = np.zeros((s.num_iters, s.batch_size, s.layer_size))
        s.ffh = np.zeros((s.num_iters, s.batch_size, s.layer_size))
        s.fbh = np.zeros((s.num_iters, s.batch_size, s.layer_size))
        s.eh = np.zeros((s.num_iters, s.batch_size, 1))
        s.reset_state()

    def reset_state(s):
        s.u = np.zeros((s.batch_size, s.layer_size))
        s.a = np.zeros((s.batch_size, s.layer_size))
        s.dW = np.zeros(s.W.shape)
        s.dWfb = np.zeros(s.Wfb.shape)


    def run(s, t, xt, yt):
        ff = np.dot(xt, s.W)
        fb = np.dot(yt, s.Wfb)

        s.u += s.dt * (ff + fb - s.u)
        s.a[:] = s.act(s.u)

        s.dW += np.dot(xt.T, fb - ff)/s.num_iters
        s.dWfb += np.dot(yt.T, fb - ff) / s.num_iters

        s.ah[t, :] = s.a.copy()
        s.uh[t, :] = s.u.copy()
        s.ffh[t, :] = ff
        s.fbh[t, :] = fb
        s.eh[t, :] = np.linalg.norm(fb-ff, axis=1, keepdims=True)


# act = lambda x: np.maximum(1.0/(1.0 + np.exp(-x)) - 0.5, 0.0)
act = lambda x: x

net = [
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
]


def run(t, xt, yt, fb_factor):
    for li, l in enumerate(net):
        x_l = net[li - 1].a if li > 0 else xt
        y_l = net[li + 1].a if li < len(net) - 1 else fb_factor * yt

        l.run(t, x_l, y_l)


net[0].Wfb = net[1].W.T.copy()
l0, l1 = net
l1.Wfb[0,0] = 0.2

for epoch in xrange(10000):
    for t in xrange(num_iters): run(t, x[t], y[t], 1.0)

    # net[1].dW = np.zeros(net[1].dW.shape)

    dW_norm = np.zeros(len(net))

    l0.W += 0.05 * l0.dW
    l1.W += 0.1 * l1.dW

    for li, l in enumerate(net):
        if li < len(net)-1:
            net[li].Wfb = net[li+1].W.T.copy()

        dW_norm[li] = np.linalg.norm(l.dW)
        l.reset_state()



    if epoch % 100 == 0:
        for t in xrange(num_iters): run(t, x[t], y[t], 0.0)
        [l.reset_state() for li, l in enumerate(net)]

        mse = np.linalg.norm(net[-1].ah - y)

        print "Epoch {}, {:.4f}, {}, {}".format(
            epoch,
            mse,
            ",".join(["{:.4f}".format(np.sum(l.eh)) for l in net]),
            ",".join(["{:.4f}".format(dW_norm_l) for dW_norm_l in dW_norm]),
        )

