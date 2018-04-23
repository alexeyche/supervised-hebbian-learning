
import sys
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
from util import *
from inspect import currentframe, getframeinfo
import numpy as np

from poc.common import *



act_o = Sigmoid()
act = Relu()
sigmoid = Sigmoid()

# np.random.seed(22) 


input_size = 2
output_size = 1
batch_size = 4
net_size = 30
step = 0.1


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

epochs = 1000
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
    a1 = act_o(u1)
    
    de = y - a1
    
    da0 = np.dot(de, W1.T) * act.deriv(u0)
    
    
    # ap = sigmoid(1000.0*np.maximum(da0, 0.0)) - 0.5
    ap = 10.0*(softplus(10.0*np.maximum(da0, 0.0)) - softplus(0.0))
    # ap = softplus(10.0*np.maximum(da0, 0.0)) - softplus(0.0)
    # ap = (
    #     sigmoid(apical_gain*np.maximum(da0, 0.0) - apical_threshold) - 
    #     sigmoid(-apical_threshold)
    # )
    
    
    # hb0 = sigmoid(np.maximum(fb, 0.0)) - a0
    # hb0 = ap 
    silent_ap = np.where(ap == 0.0)
    ltd = np.zeros(a0.shape)
    ltd[silent_ap] = a0[silent_ap]

    hb0 = ap - ltd

    if lrule == "bp":
        W0 += lrate * np.dot(x.T, da0 * act.deriv(u0))
        b0 += lrate * np.mean(da0 * act.deriv(u0), 0)
        b1 += lrate * np.mean(de, 0)

    elif lrule == "hebb":
        W0 += lrate * np.dot(x.T, hb0 )
        b0 += lrate * np.mean(hb0, 0)
        b1 += lrate * np.mean(de, 0)
    

    W1 += lrate * np.dot(a0.T, de)
    
    # W0 = norm(W0)

    da0h[e] = da0.copy()
    u0h[e] = u0.copy()
    a0h[e] = a0.copy()
    u1h[e] = u1.copy()
    a1h[e] = a1.copy()
    deh[e] = de.copy()

    aph[e] = ap.copy()
    fbh[e] = da0.copy()
    hb0h[e] = hb0.copy()

    if e % 20 == 0:
        print "{}: E^2 train {:.4f}, %{:.2f} signs".format(e, np.sum(de ** 2.0), 100.0*np.mean(np.sign(hb0) == np.sign(da0)))

# shm(np.sign(hb0), da0)

# shl(0.03*hb0h[10,1], da0h[10,1], labels=["hebb", "da"])

b=1
e=-1; shl(
    hb0h[e,b],
    da0h[e,b] * act.deriv(u0h[e,b]), 
    np.zeros(net_size), 
    a0h[e,b], 
    labels=["hebb", "da", "0", "a0"]
)
e=-1; shl(np.sign(hb0h[e,b]), np.sign(da0h[e,b] * act.deriv(u0h[e,b])), labels=["hebb", "da"])
# shl(hb0h[:,1,:])
plt.show()
