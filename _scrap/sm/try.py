

import autograd.numpy as np
from autograd import grad
from util import shl, shm, shs

from datasets import ToyDataset

np.random.seed(10)

ds = ToyDataset()

xv, yv = ds.next_train_batch()

batch_size = xv.shape[0]

def cost(y):
	x_gram = np.dot(xv, xv.T)
	y_gram = np.dot(y, y.T)
	return np.sum(np.square(y_gram - x_gram))

grad_cost = grad(cost)


n_comp = 2

yit = np.random.random((batch_size, n_comp))


for it in xrange(200):
	cost_val = cost(yit)

	dy = grad_cost(yit)

	yit -= 0.0003 * dy


	print "It {}, cost {:.4f}".format(it, cost_val)

