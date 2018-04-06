
import numpy as np
from util import shl, shm, shs
from datasets import ToyDataset
from sklearn.datasets import make_classification, make_circles
import sklearn.decomposition as dec
from poc.opt import *
from datasets import *
from poc.common import *

def xavier_init(fan_in, fan_out, const=1.0):
    low = -const * np.sqrt(6.0 / (fan_in + fan_out))
    high = const * np.sqrt(6.0 / (fan_in + fan_out))
    return (low + np.random.random((fan_in, fan_out)) * (high - low)).astype(np.float32)


def cost(x, y):
	x_gram = np.dot(x, x.T)
	y_gram = np.dot(y, y.T)
	return np.sum(np.square(y_gram - x_gram))



seed = 5
np.random.seed(seed)

input_size = 2
layer_size = 10
output_size = 1
dt = 0.1


# pca = dec.PCA(2)
# a = pca.fit(x).transform(x)

ds = XorDataset()
x, y_t = ds.train_data
batch_size = x.shape[0]


act_o = Sigmoid()


W = xavier_init(input_size, layer_size)
Wo = xavier_init(layer_size, output_size)

Mbase = xavier_init(5, layer_size)
M = np.dot(Mbase.T, Mbase)
np.fill_diagonal(M, 0.0)

num_iters = 50

opt = SGDOpt((0.05, 0.005))
opt.init(W, M)

for e in xrange(5000):
	yh = np.zeros((num_iters, batch_size, layer_size))
	zh = np.zeros((num_iters, batch_size, output_size))

	y = np.zeros((batch_size, layer_size))
	z = np.zeros((batch_size, output_size))

	for t in xrange(num_iters):	
		if t == 5:
			dy = np.dot(x, W) + np.dot(z, Wo.T)
		else:
			dy = np.dot(z, Wo.T) 
		
		dy += - np.dot(y, M) 
		
		dy = np.maximum(dy, 0.0) - 2.0*y
		
		y += dt * dy

		z = np.dot(y, Wo)
		
		zh[t] = z.copy()	
		yh[t] = y.copy()

		if t == 5:
			z = y_t
		else:
			z = np.zeros((batch_size, output_size))		


	# dW = np.dot(x.T, y) - W
	# dM = np.dot(y.T, y) - M

	y = yh[6]

	dW = np.dot((x - np.dot(y, W.T)).T, y)
	dM = np.dot((y - np.dot(y, M.T)).T, y)
	dWo = np.dot(y.T, z)

	opt.update(-dW, -dM)
	np.fill_diagonal(M, 0.0)



	yth = np.zeros((num_iters, batch_size, layer_size))
	oth = np.zeros((num_iters, batch_size, output_size))

	yt = np.zeros((batch_size, layer_size))
	zt = np.zeros((batch_size, output_size))

	for t in xrange(num_iters):	
		dy = np.dot(x, W) 
		if t == 5:
			dy = np.dot(x, W) + np.dot(zt, Wo.T)
		else:
			dy = np.dot(zt, Wo.T) 

		dy += - np.dot(yt, M)

		dy = np.maximum(dy, 0.0) - 2.0*yt		

		# dy = np.maximum(np.dot(x, W) + np.dot(zt, Wo.T) - np.dot(yt, M), 0.0)
		yt += dt * dy

		ot = np.dot(yt, Wo)

		if t == 5:
			zt = ot
		else:
			zt = np.zeros((batch_size, output_size))

		oth[t] = ot.copy()	
		yth[t] = yt.copy()



	if e % 10 == 0:
		print "E {}, Cost0 {:.4f}, Cost1 {:.4f}, Class.error: {:.4f}, Class.error, train: {:.4f}".format(
			e, 
			cost(x, y), 
			cost(z, y),
			np.linalg.norm(oth[6] - y_t),
			np.linalg.norm(zh[6] - y_t)
		)

