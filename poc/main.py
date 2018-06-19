

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
x[20,0,:] = (0.0, 0.0)
x[20,1,:] = (0.0, 1.0)
x[20,2,:] = (1.0, 0.0)
x[20,3,:] = (1.0, 1.0)


y = np.zeros((num_iters, batch_size, output_size))
y[21,0,:] = (0.0,)
y[21,1,:] = (1.0,)
y[21,2,:] = (1.0,)
y[21,3,:] = (0.0,)

x = smooth_batch_matrix(x)
y = smooth_batch_matrix(y)


net = Net(
    Layer(
        num_iters,
        batch_size,
        input_size,
        layer_size,
        output_size,
        act=act,
        weight_factor=weight_factor,
        dt=dt,
        adapt_gain=10.0,
        tau_m=20.0,
        tau_syn=10.0,
    ),
    Layer(
        num_iters,
        batch_size,
        layer_size,
        output_size,
        output_size,
        act=act,
        weight_factor=weight_factor,
        dt=dt,
        adapt_gain=1.0,
        tau_m=20.0,
        tau_syn=10.0,
    )
)

l0, l1 = net.layers

net[0].Wfb = net[1].W.T.copy()
l1.Wfb[0,0] = 1.0



for epoch in xrange(1):
    for t in xrange(num_iters): net.run(t, x[t], y[t], feedback_delay=25, test=False)
    
    l0.W += 0.05 * l0.dW
    l1.W += 0.05 * l1.dW

    net.reset_state()

    if epoch % 100 == 0:
        for t in xrange(num_iters): net.run(t, x[t], y[t], feedback_delay=25, test=True)
        net.reset_state()

        mse = np.linalg.norm(net[-1].ah[21] - y[21])

        print "Epoch {}, {:.4f}".format(
            epoch,
            mse
        )

