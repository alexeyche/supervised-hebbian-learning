
import sys
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
from util import *
from inspect import currentframe, getframeinfo
import numpy as np

class Act(object):
    def __call__(self, x):
        raise NotImplementedError()

    def deriv(self, x):
        raise NotImplementedError()

class Linear(Act):
    def __call__(self, x):
        return x

    def deriv(self, x):
        if hasattr(x, "shape"):
            return np.ones(x.shape)
        return 1.0

class Sigmoid(Act):
    def __call__(self, x):
        return 1.0/(1.0 + np.exp(-x))

    def deriv(self, x):
        v = self(x)
        return v * (1.0 - v)


class Relu(Act):
    def __call__(self, x):
        return np.maximum(x, 0.0)
        
    def deriv(self, x):
        if isinstance(x, float):
            return 1.0 if x > 0.0 else 0.0
        dadx = np.zeros(x.shape)
        dadx[np.where(x > 0.0)] = 1.0
        return dadx





act_o = Sigmoid()
act = Sigmoid()
sigmoid = Sigmoid()

np.random.seed(35) 


input_size = 2
output_size = 1
batch_size = 4
net_size = 30
num_iters = 100
step = 0.1
fb_factor = 0.0
fb_delay = 1


x = np.asarray([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
])

y = np.asarray([
    [0.0],
    [1.0],
    [1.0],
    [0.0]
])


W0 = 0.1 - 0.2*np.random.random((input_size, net_size))
b0 = np.zeros((net_size,))

# W0 = norm(W0)


W1 = 0.1 - 0.2*np.random.random((net_size, output_size))
b1 = np.zeros((output_size,))

epochs = 1
lrate = 0.1
apical_gain = 1000.0
apical_threshold = 10.0
lrule = "hebb"
# lrule = "bp"

u0h = np.zeros((epochs, batch_size, net_size))
a0h = np.zeros((epochs, batch_size, net_size))
u1h = np.zeros((epochs, batch_size, output_size))
a1h = np.zeros((epochs, batch_size, output_size))
deh = np.zeros((epochs, batch_size, output_size))
aph = np.zeros((epochs, batch_size, net_size))
fbh = np.zeros((epochs, batch_size, net_size))
da0h = np.zeros((epochs, batch_size, net_size))
hb0h = np.zeros((epochs, batch_size, net_size))

for e in xrange(epochs):

    u0 = np.dot(x, W0)        
    a0 = act(u0)

    u1 = np.dot(a0, W1) + b1
    a1 = u1
    
    de = y - a1 
    
    da0 = np.dot(de, W1.T) * act.deriv(u0)
    
    if lrule == "bp":
        

        W0 += lrate * np.dot(x.T, da0)
    
    elif lrule == "hebb":
        fb = np.dot(de, W1.T) * act.deriv(u0)
        ap = 1.0* (
            sigmoid(apical_gain*np.maximum(fb, 0.0) - apical_threshold) - 
            sigmoid(-apical_threshold)
        )
        
        aph[e] = ap.copy()
        fbh[e] = fb.copy()

        hb0 = ap - 0.5
        # hb0 = (a0 + ap) - 0.5
        hb0h[e] = hb0.copy()
        
        W0 += lrate * 0.01 * np.dot(x.T, hb0)
    

    # W1 += lrate * np.dot(a0.T, de)
    # W0 = norm(W0)

    da0h[e] = da0.copy()
    u0h[e] = u0.copy()
    a0h[e] = a0.copy()
    u1h[e] = u1.copy()
    a1h[e] = a1.copy()
    deh[e] = de.copy()

    if e % 20 == 0:
        print "{}: E^2 train {:.4f}, %{:.2f} signs".format(e, np.sum(de ** 2.0), 100.0*np.mean(np.sign(hb0) == np.sign(da0)))


# shm(np.sign(hb0), np.sign(da0))

# shl(0.03*hb0h[10,1], da0h[10,1], labels=["hebb", "da"])
e=-1; shl(0.03*hb0h[e,2], da0h[e,2], np.zeros(net_size), labels=["hebb", "da", "0"], show=False)
e=-1; shl(np.sign(0.03*hb0h[e,2]), np.sign(da0h[e,2]), labels=["hebb", "da"])
# shl(hb0h[:,1,:])
plt.show()