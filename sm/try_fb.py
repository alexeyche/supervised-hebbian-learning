
import numpy as np
from util import shl, shm, shs
from datasets import ToyDataset
from sklearn.datasets import make_classification, make_circles
import sklearn.decomposition as dec
from poc.opt import *
from sm.sm_cost import *

def xavier_init(fan_in, fan_out, const=1.0):
    low = -const * np.sqrt(6.0 / (fan_in + fan_out))
    high = const * np.sqrt(6.0 / (fan_in + fan_out))
    return (low + np.random.random((fan_in, fan_out)) * (high - low)).astype(np.float32)



seed = 5
np.random.seed(seed)

batch_size = 300
input_size = 2
layer_size = 5
dt = 0.1

# x, classes = make_circles(n_samples=300, factor=0.5, noise=0.05)

# x = np.random.random((batch_size, 2))

# x = np.concatenate([
# 	np.asarray((-0.0985, -0.3379)) + 0.04*np.random.randn(batch_size/3, input_size),
# 	np.asarray((-0.6325, 0.9322)) + 0.04*np.random.randn(batch_size/3, input_size),
# 	np.asarray((1.1078, 1.0856)) + 0.04*np.random.randn(batch_size/3, input_size)
# ])

# z = np.concatenate([
# 	np.asarray((-0.2314, -0.0978)) + 0.04*np.random.randn(batch_size/3, input_size),
# 	np.asarray((-0.1492, 0.9862)) + 0.04*np.random.randn(batch_size/3, input_size),
# 	np.asarray((-0.2422,-0.6053)) + 0.04*np.random.randn(batch_size/3, input_size)
# ])

x = np.random.multivariate_normal(
	np.asarray([0.5, 0.5]),
	[
		[0.5, 0.45], 
		[0.45, 0.5]
	],
	size=batch_size,
)

Cx = np.cov(x.T)
eigvalX, eigvecX = np.linalg.eig(Cx)

z = np.random.multivariate_normal(
	np.asarray([0.5, 0.5]),
	[
		[0.5, 0.45], 
		[0.45, 0.5]
	],
	size=batch_size,
)

Cz = np.cov(z.T)
eigvalZ, eigvecZ = np.linalg.eig(Cz)

# pca = dec.PCA(2)
# a = pca.fit(x).transform(x)


classes = [0]*(batch_size/3) + [1]*(batch_size/3) + [2]*(batch_size/3)



W = xavier_init(input_size, layer_size)
Wz = xavier_init(input_size, layer_size)

Mbase = xavier_init(5, layer_size)
M = np.dot(Mbase.T, Mbase)
np.fill_diagonal(M, 0.0)

num_iters = 50

ysq = np.zeros((batch_size, layer_size))

opt = SGDOpt((0.000001, 0.000001, 0.0000001))
opt.init(W, Wz, M)

F = np.dot(W + Wz, np.eye(layer_size) + np.linalg.inv(M))

for e in xrange(5000):
	yh = np.zeros((num_iters, batch_size, layer_size))

	y = np.zeros((batch_size, layer_size))
	
	for t in xrange(num_iters):	
		dy = np.dot(x, W) + np.dot(z, Wz) - np.dot(y, M)
		# dy = np.maximum(np.dot(x, W) + np.dot(z, Wz) - np.dot(y, M), 0.0)
		y += dt * dy
		ysq += dt * np.square(y)

		yh[t] = y.copy()

	# dW = np.dot(x.T, y) - W
	# dM = np.dot(y.T, y) - M
	dW = np.dot((x - np.dot(y, W.T)).T, y)
	dWz = np.dot((z - np.dot(y, Wz.T)).T, y)
	dM = np.dot((y - np.dot(y, M.T)).T, y)

	opt.update(-dW, -dWz, -dM)
	np.fill_diagonal(M, 0.0)

	if e % 10 == 0:
		print "E {}, Cost0 {:.4f}, Cost1 {:.4f}".format(e, cmds_cost(x, y), cmds_cost(z, y))


F = np.dot(W+Wz, np.eye(M.shape[0]) + np.linalg.inv(M))
_, eigvecF = np.linalg.eig(np.dot(F, F.T))

eigval, eigvec =np.linalg.eig(np.cov(y.T))
np.real(eigvec[:,np.argsort(np.real(eigval))])

	# shs(y, labels=([str(c) for c in classes],))

# shs(x, labels=(np.argmax(y,1),), show=False)
# shs(z, labels=(np.argmax(y,1),))