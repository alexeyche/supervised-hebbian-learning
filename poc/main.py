
import sys
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
from util import *
import numpy as np

from poc.common import *
from datasets import XorDataset

# np.random.seed(10)

class Layer(object):
    def __init__(self, input_size, layer_size, feedback_size, act):
        self.input_size = input_size
        self.layer_size = layer_size
        self.feedback_size = feedback_size
        self.act = act
        
        self.W = 0.1 - 0.2*np.random.random((input_size, layer_size))
        self.b = np.zeros((layer_size,))


    def run_feedforward(s, I):
        u = np.dot(I, s.W) + s.b
        a = s.act(u)
        return u, a

    def __repr__(self):
        return "Layer({};{};{})".format(self.input_size, self.layer_size, self.feedback_size)


def no_gradient_postproc(dE, a):
    return dE

def ltd(dE, a):
    silent_ap = np.where(dE <= 0.0)
    ltd = np.zeros(a.shape)
    ltd[silent_ap] = a[silent_ap]
    return ltd 

def positive_postproc(dE, a):
    return np.maximum(dE, 0.0) - ltd(dE, a)

def nonlinear_postproc(dE, a):
    return \
        2.0*(sigmoid(100.0*np.maximum(dE, 0.0)) - 0.5) - ltd(dE, a)



gradient_postproc = nonlinear_postproc
ds = XorDataset()
x_shape, y_shape = ds.train_shape

batch_size, input_size = x_shape
batch_size, output_size = y_shape
lrate = 0.1

net_struct = (100, 100, output_size)

net = tuple([
    Layer(inp_size, lsize, fb_size, act) 
    for inp_size, lsize, fb_size, act in 
        zip(
            (input_size,) + net_struct[:-1], 
            net_struct, 
            net_struct[1:] + (None,),
            (Relu(),)*(len(net_struct)-1) + (Sigmoid(),)
        )
])

a_stat = [np.zeros((batch_size, l.layer_size)) for l in net]
u_stat = [np.zeros((batch_size, l.layer_size)) for l in net]
dE_stat = [np.zeros((batch_size, l.layer_size)) for l in net]

epochs = 10000

stat = np.zeros((epochs, len(net_struct)-1, 1))

for epoch in xrange(epochs):
    x_target, y_target = ds.next_train_batch()

    I = x_target
    for l_idx, l in enumerate(net):
        u, a = l.run_feedforward(I)

        a_stat[l_idx] = a.copy()
        u_stat[l_idx] = u.copy()

        I = a

    y = I

    dE = y_target - y
    
    net[-1].W += lrate * np.dot(a_stat[-2].T, dE)
    net[-1].b += lrate * np.mean(dE, 0)
    
    dE_stat[-1] = dE.copy()
    
    for l_idx, (l, l_next, u, a, I) in reversed(
        tuple(
            enumerate(
                zip(net[:-1], net[1:], u_stat[:-1], a_stat[:-1], (x_target,) + tuple(a_stat[:-1]))
            )
        )
    ):
        dE_actual = np.dot(dE, l_next.W.T) * l.act.deriv(u)
        dE = gradient_postproc(dE_actual, a)
        stat[epoch, l_idx] = np.mean(np.sign(dE_actual) == np.sign(dE))
        
        l.W += lrate * np.dot(I.T, dE)
        l.b += lrate * np.mean(dE, 0)

        dE_stat[l_idx] = dE.copy()


    if epoch % 100 == 0:
        print "{}: E^2 train {:.4f}".format(epoch, np.sum(dE ** 2.0))

# shl(dE_actual[1], dE[1], a_stat[0][1], labels=["dE_actual", "dE", "A"])
shl(a_stat[1].T)