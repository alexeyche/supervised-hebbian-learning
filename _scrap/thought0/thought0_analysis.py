#!/usr/bin/env python

import numpy as np
from util import *
from sklearn.datasets import make_circles
from datasets import *
np.random.seed(1)

# Setup
layer_size = 20

def tanh(x):
    return (1.0 - np.exp(-2*x))/(1.0 + np.exp(-2*x))

def tanh_derivative(x):
    return (1 + tanh(x))*(1 - tanh(x))

def relu_deriv(x):
    dadx = np.zeros(x.shape)
    dadx[np.where(x > 0.0)] = 1.0
    return dadx

def relu(x):
    return np.maximum(x, 0.0)

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_deriv(x):
    a = sigmoid(x)
    return a * (1.0 - a)


act, dact = relu, relu_deriv
# act, dact = sigmoid, sigmoid_deriv
# act, dact = lambda x: x, lambda x: 1.0

# act = lambda x: 1.0/(1.0 + np.exp(-x))
# dact = lambda x: act(x) * (1.0 - act(x))

# x = np.asarray([
#     [0.0, 0.0],
#     [0.0, 1.0],
#     [1.0, 0.0],
#     [1.0, 1.0]
# ])
#
# y = np.asarray([
#     [0.0],
#     [1.0],
#     [1.0],
#     [0.0],
# ])
#
# # y = np.asarray([
# #     [0.0],
# #     [0.0],
# #     [0.0],
# #     [1.0],
# # ])

# x, y = make_circles(n_samples=200, factor=.3, noise=.05)
# y = one_hot_encode(y, 2)

input_size = 10
batch_size = 200
output_size = 5

w0 = 0.5*np.random.random((input_size, layer_size)) - 0.25
w1 = 0.5*np.random.random((layer_size, output_size)) - 0.25

x = 0.5*np.random.random((batch_size, input_size)) - 0.25
y = act(np.dot(act(np.dot(x, w0)), w1))

xt = x.copy()
yt = y.copy()

input_size = x.shape[1]
output_size = y.shape[1]


def weights(xsize, ysize):
    weight_factor = 0.2
    return weight_factor * np.random.random((xsize, ysize)) - weight_factor / 2.0
    # return 0.1*np.ones((xsize, ysize))


net = [
    [weights(input_size, layer_size), np.zeros(layer_size), weights(layer_size, layer_size)],
    [weights(layer_size, output_size), np.zeros(output_size), np.eye(output_size, output_size)],
]


net[0][2] = net[1][0].T.copy()

def iterate_chl(x_target, fb0_prev=None, y_target=None):
    (l0_W, b0, l0_Wfb), (l1_W, b1, l1_Wfb) = net[0], net[1]

    ff0 = np.dot(x_target, l0_W)

    # ff0 if fb0_prev is None else (ff0 + fb0_prev)/2.0
    u0_pre = ff0 if fb0_prev is None else ff0 + fb0_prev
    a0_pre = act(u0_pre + b0)

    ff1 = np.dot(a0_pre, l1_W)

    if not y_target is None:
        a1 = y_target
        u1 = y_target
        fb1 = 0
    else:
        fb1 = None
        u1 = ff1 + b1
        a1 = act(u1)

    fb0 = np.dot(a1, l0_Wfb)
    # a0 = act((ff0 + fb0)/2.0)
    u0 = ff0 + fb0 + b0
    a0 = act(u0)

    return a0, a1, u0, u1, ff0, ff1, fb0, fb1

def iterate_ml(x_target, fb0_prev=None, y_target=None):
    (l0_W, b0, l0_Wfb), (l1_W, b1, l1_Wfb) = net[0], net[1]

    ff0 = np.dot(x_target, l0_W)
    u0_pre = ff0 if fb0_prev is None else (ff0 + fb0_prev)/2.0
    a0_pre = act(u0_pre + b0)

    ff1 = np.dot(a0_pre, l1_W)

    if not y_target is None:
        fb1 = np.dot(y_target, l1_Wfb)
        u1 = (ff1 + fb1)/2.0
    else:
        fb1 = None
        u1 = ff1

    a1 = act(u1)

    fb0 = np.dot(a1, l0_Wfb)
    u0 = (ff0 + fb0)/2.0
    a0 = act(u0)

    return a0, a1, u0, u1, ff0, ff1, fb0, fb1


def iterate_bp(x_target, y_target):
    (W0, b0, _), (W1, b1, _) = net[0], net[1]

    u0 = np.dot(x_target, W0) + b0
    a0 = act(u0)

    u1 = np.dot(a0, W1) + b1
    a1 = act(u1)

    da1 = (y_target - a1)
    da0 = np.dot(da1, W1.T)

    return a0, a1, u0, u1, da0, da1


# rule = "bp"
rule = "ml"
# rule = "chl"

batch_size = x.shape[0]

for e in xrange(10000):
    if rule == "chl":
        a0_n, a1_n, u0_n, u1_n, ff0_n, ff1_n, fb0_n, fb1_n = \
            iterate_chl(x, None, y_target=None)

        a0, a1, u0, u1, ff0, ff1, fb0, fb1 = \
            iterate_chl(x, None, y_target=y)

        a0_m, a1_m, u0_m, u1_m, ff0_m, ff1_m, fb0_m, fb1_m = \
            iterate_ml(x, None, y_target=y)


        da0 = a0 - a0_n
        da1 = a1 - a1_n

        a0_bp, a1_bp, u0_bp, u1_bp, da0_bp, da1_bp = iterate_bp(x_target=x, y_target=y)
    elif rule == "bp":
        a0, a1, u0, u1, da0, da1 = iterate_bp(x_target=x, y_target=y)
    elif rule == "ml":
        fb0 = None
        for _ in xrange(5):
            a0, a1, u0, u1, ff0, ff1, fb0, fb1 = \
                iterate_ml(x, fb0, y_target=y)

        da0 = act(fb0) - act(ff0)
        da1 = act(fb1) - act(ff1)

        a0_bp, a1_bp, u0_bp, u1_bp, da0_bp, da1_bp = iterate_bp(x_target=x, y_target=y)

    (W0, b0, _), (W1, b1, _) = net[0], net[1]

    lrate = 0.0001

    W0 += lrate * np.dot(x.T, da0) # * dact(u0))
    b0 += lrate * np.mean(da0, 0)
    W1 += lrate * np.dot(a0.T, da1) # * dact(u1))
    b1 += lrate * np.mean(da1, 0)

    net[0][2] = net[1][0].T.copy()

    if (e+1) % 100 == 0:
        if rule == "chl":
            a0t, a1t, u0t, u1t, ff0t, ff1t, fb0t, fb1t = \
                iterate_chl(xt, None, y_target=None)
        elif rule == "ml":
            fb0t = None
            for _ in xrange(5):
                a0t, a1t, u0t, u1t, ff0t, ff1t, fb0t, fb1t = \
                    iterate_ml(xt, fb0t, y_target=None)
        else:
            a0t, a1t, _, _, _, _ = iterate_bp(x_target=xt, y_target=yt)

        print (
            e,
            np.linalg.norm(a1t - yt),
            np.mean(np.not_equal(np.argmax(a1t, 1), np.argmax(yt, 1)))
        )


