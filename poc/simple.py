
import sys
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
from util import *
from inspect import currentframe, getframeinfo
import numpy as np
from poc.common import *

from matplotlib.pyplot import cm
from pylab import rcParams
rcParams['figure.figsize'] = 25, 20


act_o = Sigmoid()
act = Relu()
sigmoid = Sigmoid()

np.random.seed(2) 


input_size = 2
output_size = 1
batch_size = 2
net_size = 1
step = 0.1
fb_factor = 0.0
fb_delay = 1


x = np.asarray([
    [0.56, 0.3],
    [0.86, 0.33],
])


y = np.mean(np.sqrt(x), 1,keepdims=True)


W0 = 0.1*np.random.random((input_size, net_size))
b0 = np.zeros((net_size,)) #np.random.randn(net_size)

# W0 = norm(W0)


W1 = 0.1*np.random.random((net_size, output_size))
b1 = np.zeros((output_size,)) #np.random.randn(output_size)



epochs = 1000
lrate = 0.1 * 1.0
apical_gain = 1000.0
apical_threshold = 10.0
# lrule = "hebb"
lrule = "bp"

resolution = 50
W_0 = np.linspace(0.0, 30.0, resolution)
W_1 = np.linspace(0.0, 30.0, resolution)


dW_0 = np.zeros((resolution, resolution))
dW_1 = np.zeros((resolution, resolution))
W_0r = np.zeros((resolution, resolution))
W_1r = np.zeros((resolution, resolution))
E = np.zeros((resolution, resolution))
o = np.zeros((resolution, resolution, batch_size))
apr = np.zeros((resolution, resolution, batch_size))
hb0r = np.zeros((resolution, resolution, batch_size))
a0r = np.zeros((resolution, resolution, batch_size))

id0, id1 = (0,0), (1,0)
for ww0_id, ww0 in enumerate(W_0):
    for ww1_id, ww1 in enumerate(W_1):
        for epoch in xrange(1):
            W0w, b0w, W1w, b1w = W0.copy(), b0.copy(), W1.copy(), b1.copy()
            
            W0w[id0[0], id0[1]] = ww0
            W0w[id1[0], id1[1]] = ww1

            u0 = np.dot(x, W0w) + b0w 
            a0 = act(u0)

            u1 = np.dot(a0, W1w) + b1w
            a1 = act_o(u1)
            
            de = y - a1 
            
            da0 = np.dot(de, W1w.T)
            
            # ap = 2.0*(sigmoid(10.0*np.maximum(da0, 0.0)) - 0.5)
            # ap = softplus(10.0*np.maximum(da0, 0.0)) - softplus(0.0)
            # ap = softplus(10.0*np.maximum(da0, 0.0)) - softplus(0.0)
            ap = 1.0*(softplus(10.0*np.maximum(da0, 0.0)) - softplus(0.0))

            # ap = (
            #     sigmoid(apical_gain*np.maximum(da0, 0.0) - apical_threshold) - 
            #     sigmoid(-apical_threshold)
            # )
        
            silent_ap = np.where(ap == 0.0)
            ltd = np.zeros(a0.shape)
            ltd[silent_ap] = a0[silent_ap]

            
            # hb0 = sigmoid(np.maximum(fb, 0.0)) - a0
            hb0 = ap - 0.01*ltd
            # hb0 = 0.1 * (ap - a0) #- 0.01
            # hb0 = (a0-0.5) + ap - 0.05
            

            if lrule == "bp":
                dW0 = np.dot(x.T, da0 * act.deriv(u0))
            elif lrule == "hebb":
                dW0 = np.dot(x.T, hb0)
            
            dW1 = np.dot(a0.T, de)

            dW_0[ww0_id, ww1_id] = dW0[id0[0], id0[1]]
            dW_1[ww0_id, ww1_id] = dW0[id1[0], id1[1]]
            W_0r[ww0_id, ww1_id] = ww0
            W_1r[ww0_id, ww1_id] = ww1

            E[ww0_id, ww1_id] = np.sum(np.square(de))
            o[ww0_id, ww1_id, :] = a1[:,0].copy()
            apr[ww0_id, ww1_id, :] = ap[:,0].copy()
            hb0r[ww0_id, ww1_id, :] = hb0[:,0].copy()
            a0r[ww0_id, ww1_id, :] = a0[:,0].copy()

            # W0 += 0.5 * dW0
            # W1 += 0.5 * dW1

            # print epoch, np.sum(np.square(de))

M = (np.hypot(dW_0, dW_1))
M[M == 0] = 1.      
dW_0 /= M             
dW_1 /= M                                  


plt.quiver(
    W_0r, W_1r, dW_0, dW_1, E,
    cmap=cm.seismic,
    headlength=7, headwidth=5.0
)
plt.colorbar()
plt.show()