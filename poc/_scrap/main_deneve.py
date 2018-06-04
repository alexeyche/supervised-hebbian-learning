#!/usr/bin/env python

import numpy as np
from util import *
from poc.opt import *
from poc.common import *

np.random.seed(13)


# Setup
batch_size = 10
layer_size = 100
output_size = 5
dt = 1.0
weight_factor = 0.5
alpha = 1.0
beta = 1.0
mu = 1.0

num_iters = 500

# c0 = generate_ts(num_iters, params=(0.1,0.01,10,10,4))
# c1 = generate_ts(num_iters, params=(0.01,0.1,1,2,3))

c0 = np.sin(np.linspace(0, 20.0, num_iters) + 0.25*np.pi)
c1 = np.sin(np.linspace(0, 20.0, num_iters))

c = np.asarray(((c0,), (c1,)))
c = np.transpose(c, (2, 1, 0))
_, batch_size, input_size = c.shape



class Layer(object):
    def __init__(
        s, 
        batch_size, 
        input_size, 
        layer_size, 
        output_size, 
        lam,
        threshold, 
        nu, 
        weight_factor = 0.5
    ):
        s.batch_size, s.input_size, s.layer_size = batch_size, input_size, layer_size
        
        s.lam = lam
        s.nu = nu
        s.threshold_scalar = threshold

        s.W = weight_factor*np.random.random((input_size, layer_size)) - weight_factor/2.0
                
        s.Om = weight_factor*np.random.random((layer_size, layer_size)) - weight_factor/2.0
        
        s.reset_state()

        s.Uh = np.zeros((num_iters, batch_size, layer_size))
        s.Ah = np.zeros((num_iters, batch_size, layer_size))
        s.Rh = np.zeros((num_iters, batch_size, layer_size))
        s.X_rec_h = np.zeros((num_iters, batch_size, input_size))
        s.Xh = np.zeros((num_iters, batch_size, input_size))
        s.Eh = np.zeros((num_iters, batch_size, 1))

    def reset_state(s):
        s.X = np.zeros((s.batch_size, s.input_size))
        s.U = np.zeros((s.batch_size, s.layer_size))
        s.A = np.zeros((s.batch_size, s.layer_size))
        s.R = np.zeros((s.batch_size, s.layer_size))
        
        s.dW = np.zeros(s.W.shape)
        s.dOm = np.zeros(s.Om.shape)
        
        s.XR = np.zeros((s.input_size, s.layer_size))
        s.RR = np.zeros((s.layer_size, s.layer_size))
    
        np.fill_diagonal(s.Om, -s.threshold_scalar)
        s.threshold = (-np.diag(s.Om) + s.nu)


    def run(s, t, x, y=None):
        s.X += dt * ( - s.lam * s.X + x)

        ff = np.dot(s.X, s.W)
        s.U = ff + np.dot(s.R, s.Om)

        s.A = np.zeros((batch_size, layer_size))
        s.A[np.where(s.U >= s.threshold)] = 1.0

        s.R += dt * ( - s.lam * s.R + s.A)
        
        s.XR += np.dot(s.X.T, s.R)
        s.RR += np.dot(s.R.T, s.R)

        s.dW += np.dot(alpha * s.X.T, s.A) - s.W
        # s.dOm += np.dot(s.A.T,- (ff + 1.0*s.R)) - s.Om 
        s.dOm += np.dot(s.A.T,- ff) - 0.1*s.Om 

        s.Uh[t] = s.U.copy()
        s.Ah[t] = s.A.copy()
        s.Rh[t] = s.R.copy()
        s.X_rec_h[t] = np.dot(s.R, s.W.T)
        s.Xh[t] = s.X.copy()
        s.Eh[t] = np.linalg.norm(s.X - np.dot(s.R, s.W.T), axis=1, keepdims=True)

    def set_optimal(s):
        D = np.dot(s.XR/num_iters, np.linalg.inv(s.RR/num_iters + np.eye(s.layer_size) * 5.0))

        s.W = D
        s.Om = -np.dot(D.T, D)


net = [
    Layer(
        batch_size, 
        input_size, 
        layer_size, 
        output_size, 
        lam=0.1,
        threshold=0.25, 
        nu=0.0, 
        weight_factor=0.05
    ),
]




l0 = net[0]

opt = SGDOpt((0.00001, 0.00001))
# opt = AdamOpt((0.001, 0.001), 0.9)
opt.init(l0.W, l0.Om)

for e in xrange(200):
    for t in xrange(num_iters): l0.run(t, c[t])

    opt.update(-l0.dW, -l0.dOm)

    l0.reset_state()

    # for l in net:
    #     l.W += 0.00001 * l.dW 
    #     l.Om += 0.00001 * l.dOm

    #     l.reset_state()

    print "Epoch {}, {}".format(e, np.linalg.norm(net[0].Eh))

# l0.set_optimal()

# l0.reset_state()
# for t in xrange(num_iters): l0.run(t, c[t])
