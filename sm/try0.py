
import numpy as np
from util import shl, shm, shs
from datasets import ToyDataset
from sklearn.datasets import make_classification, make_circles
import sklearn.decomposition as dec
from poc.opt import *

def xavier_init(fan_in, fan_out, const=1.0):
    low = -const * np.sqrt(6.0 / (fan_in + fan_out))
    high = const * np.sqrt(6.0 / (fan_in + fan_out))
    return (low + np.random.random((fan_in, fan_out)) * (high - low)).astype(np.float32)


def cost(x, y):
	x_gram = np.dot(x, x.T)
	y_gram = np.dot(y, y.T)
	return np.sum(np.square(y_gram - x_gram))

def eigvec_cost(W, M, eigvec):
	F = np.dot(W, np.eye(M.shape[0]) + np.linalg.inv(M))

	_, eigvecF = np.linalg.eig(np.dot(F, F.T))
	return np.sum(np.square(eigvecF - eigvec))

def orthonormal_cost(W, M):
	F = np.dot(W, np.eye(M.shape[0]) + np.linalg.inv(M))

	return np.linalg.norm(np.dot(F.T, F) - np.eye(M.shape[0]))
	

seed = 5
np.random.seed(seed)

batch_size = 300
input_size = 2
layer_size = 10
dt = 0.1

# x, classes = make_circles(n_samples=300, factor=0.5, noise=0.05)

# x = np.random.random((batch_size, 2))

x = np.concatenate([
	np.asarray((-0.0985, -0.3379)) + 0.04*np.random.randn(batch_size/3, input_size),
	np.asarray((-0.6325, 0.9322)) + 0.04*np.random.randn(batch_size/3, input_size),
	np.asarray((1.1078, 1.0856)) + 0.04*np.random.randn(batch_size/3, input_size)
])
classes = [0]*(batch_size/3) + [1]*(batch_size/3) + [2]*(batch_size/3)

C = np.cov(x.T)
eigval, eigvec = np.linalg.eig(C)


W = xavier_init(input_size, layer_size)
Mbase = xavier_init(1, layer_size)
M = np.dot(Mbase.T, Mbase)
np.fill_diagonal(M, 0.0)

num_iters = 50

ysq = np.zeros((batch_size, layer_size))

# opt = AdamOpt((0.00001, 0.000001))
opt = SGDOpt((0.000001, 0.0000001))
# opt = SGDOpt((0.00001, 0.00001))
opt.init(W, M)


for e in xrange(2500):
	yh = np.zeros((num_iters, batch_size, layer_size))

	y = np.zeros((batch_size, layer_size))
	
	for t in xrange(num_iters):		
		dy = np.dot(x, W) - np.dot(y, M)
		# dy = np.maximum(np.dot(x, W) - np.dot(y, M), 0.0)
		y += dt * dy
		ysq += dt * np.square(y)

		yh[t] = y.copy()

	dW = np.dot((x - np.dot(y, W.T)).T, y) #/ysq)
	dM = np.dot((y - np.dot(y, M.T)).T, y) #/ysq)

	opt.update(-dW, -dM)
	np.fill_diagonal(M, 0.0)

	if e % 10 == 0:
		print "E {}, Cost(0) {:.4f}, Cost(eig): {:.4f}, Cost(orth): {:.4f}".format(
			e, 
			cost(x, y), 
			eigvec_cost(W, M, eigvec),
			orthonormal_cost(W, M),
		)







	# shs(y, labels=([str(c) for c in classes],))

# shs(x, labels=(np.argmax(y,1),), show=True)