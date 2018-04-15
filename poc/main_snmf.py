
import numpy as np
from poc.opt import *
from datasets import *
from poc.common import *
from cost import *


ds = ToyDataset()
act = Relu()

seed = 10
layer_size = 300
dt = 0.25
omega = 1.0
k = 1.0
num_iters = 50

p, q = 0.1, 0.1



np.random.seed(seed)

(batch_size, input_size), (_, output_size) = ds.train_shape

x, y = ds.next_train_batch()

threshold = 0.1

W = np.random.random((input_size, layer_size)) * 0.1
M = np.random.random((layer_size, layer_size)) * 0.1

def run_cd():
	a = np.zeros((batch_size, layer_size))

	n_idx = np.arange(layer_size)
	for ni in xrange(layer_size):
		not_ni = np.where(n_idx != ni)[0]
		
		c = np.dot(x, W[:, ni]) - np.dot(a[:, not_ni], M[not_ni, ni])
		a[:, ni] = act(c - threshold)

	return a


def run_online(num_iters, dt):
	a = np.zeros((batch_size, layer_size))
	ah = np.zeros((num_iters, batch_size, layer_size))

	for i in xrange(num_iters):
		c = np.dot(x, W) - np.dot(a, M)
		a += dt * act(c - threshold)
		
		ah[i] = a.copy()
	return a, ah

def run_online_spike(num_iters, dt, spike_gain):
	a = np.zeros((batch_size, layer_size))
	ah = np.zeros((num_iters, batch_size, layer_size))
	a_sp = np.zeros(a.shape)

	for i in xrange(num_iters):
		c = np.dot(x, W) - np.dot(a_sp, M)
		a += dt * act(c - threshold)
		
		a_sp[np.where(a * spike_gain * dt > np.random.random(a.shape))] += 1.0
		a_sp += - 10.0*dt * a_sp

		ah[i] = a_sp.copy()
	return a_sp, ah


a = run_cd()
ao, aoh = run_online(100, 0.01)
aos, aosh = run_online_spike(100, 0.01, 10.0)

