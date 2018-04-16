
import numpy as np
from poc.opt import *
from datasets import *
from poc.common import *
from cost import *

def run_cd():
	a = np.zeros((batch_size, layer_size))

	n_idx = np.arange(layer_size)
	for ni in xrange(layer_size):
		not_ni = np.where(n_idx != ni)[0]
		
		c = np.dot(x, W[:, ni]) - np.dot(a[:, not_ni], M[not_ni, ni])
		a[:, ni] = act(c - threshold[ni])

	return a, None


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


act = Relu()

seed = 10
layer_size = 100
dt = 0.25
omega = 1.0
k = 1.0
num_iters = 50

np.random.seed(seed)


# run_dynamic = lambda: run_online(100, 0.01)
# run_dynamic = lambda: run_online_spike(100, 0.01, 1.0)
run_dynamic = lambda: run_cd()

batch_size = 1000

ds = FacesDataset(batch_size=batch_size)

(_, input_size), (_, output_size) = ds.train_shape



threshold = np.repeat(0.0,layer_size)

W = np.random.random((input_size, layer_size)) * 0.1
M = np.random.random((layer_size, layer_size)) * 0.1
np.fill_diagonal(M, 0.0)


reg = 1.0
p = 0.5

W = W/(np.sum(np.abs(W), 0)/p)


opt = SGDOpt((0.01, 0.001))
opt.init(W, M)

tau_m = 100.0

am = np.zeros((layer_size, ))
for e in xrange(100):
	dW = np.zeros(W.shape)
	dM = np.zeros(M.shape)

	for bi in xrange(ds.train_batches_num):
		x, y = ds.next_train_batch()

		a, ah = run_dynamic()

		dW += np.dot((x - np.dot(a, W.T)).T, a) - reg * W
		dM += np.dot((a - np.dot(a, M.T)).T, a) - reg * M	

		am += (np.mean(a * a, 0) - am) / tau_m
		threshold = am

	opt.update(-dW, -dM)
	np.fill_diagonal(M, 0.0)
	
	
	print e, cmds_cost(x, a)

# shm(
# 	W,
# 	W[:,0].reshape((20, 20)),
# 	W[:,10].reshape((20, 20)),
# 	W[:,25].reshape((20, 20)),
# 	W[:,-5].reshape((20, 20)),
# 	W[:,-1].reshape((20, 20))
# )

shm(
	W,
	W[:,0].reshape((20, 20)),
	W[:,1].reshape((20, 20)),
	W[:,2].reshape((20, 20)),
	W[:,3].reshape((20, 20)),
	W[:,4].reshape((20, 20))
)