#!/usr/bin/env python

import numpy as np
from util import *

np.random.seed(10)

# Setup
layer_size = 10

weight_factor = 1.0


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




# act = lambda x: x # np.maximum(x, 0.0)
# dact = lambda x: 1.0 # np.maximum(x, 0.0)


def weights(xsize, ysize):
    return weight_factor * np.random.random((xsize, ysize)) - weight_factor / 2.0


x = np.asarray([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])

y = np.asarray([
    [0.0],
    [1.0],
    [1.0],
    [0.0],
])

batch_size, input_size = x.shape
_, output_size = y.shape

W0 = weights(input_size, layer_size)
W1 = weights(layer_size, output_size)
b0 = np.zeros(layer_size)
b1 = np.zeros(output_size)

lrate = 0.1

for e in xrange(15000):
    u0 = np.dot(x, W0) + b0
    a0 = act(u0)

    u1 = np.dot(a0, W1) + b1
    a1 = act(u1)

    da1 = (y - a1) * dact(u1)
    da0 = np.dot(da1, W1.T) * dact(u0)

    W0 += lrate * np.dot(x.T, da0)
    b0 += lrate * np.mean(da0, 0)
    W1 += lrate * np.dot(a0.T, da1)
    b1 += lrate * np.mean(da1, 0)


    if e % 100 == 0:
        print "{} {:.4f}".format(e, np.linalg.norm(a1-y))


