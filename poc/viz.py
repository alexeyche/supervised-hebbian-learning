
import sys
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
from util import *
import numpy as np

from sklearn.decomposition import RandomizedPCA
from mpl_toolkits.mplot3d import Axes3D

from poc.common import *
from datasets import *

np.random.seed(9)

def read_vec(net):
    return np.concatenate([np.concatenate((l.W.reshape(-1), l.b.reshape(-1))) for l in net])

def read_vec_list(list_of_params):
    return np.concatenate([np.concatenate((W.reshape(-1), b.reshape(-1))) for (W,b) in list_of_params])


def set_vec(net, vec):
    left_b = 0
    for l in net:
        d = l.W.shape[0]*l.W.shape[1]
    
        l.W = vec[left_b:(left_b+d)].reshape(l.W.shape)
        
        left_b += d
        d = l.b.shape[0]
        
        l.b = vec[left_b:(left_b+d)]
        left_b += d
        

def run_net(net, params, ds, gradient_postproc, learn=False):
    set_vec(net, params)

    derivatives, a_stat, u_stat, m  = run_feedforward(
        net, 
        ds, 
        is_train_phase=True, 
        gradient_postproc=gradient_postproc
    )
    if learn:
        for l, (dW, db) in zip(net, derivatives):
            l.W += lrate * dW
            l.b += lrate * db


    se, ll, er = m
    return ll


gradient_postproc = no_gradient_postproc
# ds = XorDataset()
ds = ToyDataset()
# ds = MNISTDataset()
x_shape, y_shape = ds.train_shape

_, input_size = x_shape
_, output_size = y_shape


lrate = 0.001

net = build_network(input_size, (300, output_size))

epochs = 1000
params = np.zeros((epochs, read_vec(net).shape[0]))
errors = np.zeros(epochs)
for epoch in xrange(epochs):
    sigma = read_vec(net)

    params[epoch] = sigma
    errors[epoch] = run_net(net, sigma, ds, positive_postproc, learn=True)
    if epoch % 100 == 0:
        print epoch


pca = RandomizedPCA(n_components=2, whiten=False)

params_to_fit = params[:-1].copy()
for epoch in xrange(epochs-1):
    params_to_fit[epoch] = params_to_fit[epoch] - params[-1]

pca.fit(params_to_fit)
# pca.fit(params)
path = pca.transform(params)

resolution = 50

alpha_X, beta_Y = np.meshgrid(
    np.linspace(np.min(path[:,0])-1.0, np.max(path[:,0])+1.0, resolution), 
    np.linspace(np.min(path[:,1])-1.0, np.max(path[:,1])+1.0, resolution)
)

Esurface = np.zeros((resolution, resolution))

for alpha_idx in xrange(resolution):
    for beta_idx in xrange(resolution):
        alpha, beta = alpha_X[alpha_idx, beta_idx], beta_Y[alpha_idx, beta_idx]
        sigma = pca.inverse_transform(np.asarray((alpha, beta)))
        Esurface[alpha_idx, beta_idx] = run_net(net, sigma, ds, gradient_postproc)


errors = [
    run_net(net, pca.inverse_transform(path[epoch]), ds, gradient_postproc, learn=False)
    for epoch in xrange(epochs)
]

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(
    alpha_X, 
    beta_Y, 
    Esurface, 
    cmap=cm.coolwarm,
    linewidth=0, 
    antialiased=False
)
ax.plot(path[:,0], path[:,1], errors)

plt.show()
