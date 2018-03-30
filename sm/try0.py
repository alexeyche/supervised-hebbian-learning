
import numpy as np
from util import shl, shm, shs
from datasets import ToyDataset
from sklearn.datasets import make_classification
import sklearn.decomposition as dec

def xavier_init(fan_in, fan_out, const=1.0):
    low = -const * np.sqrt(6.0 / (fan_in + fan_out))
    high = const * np.sqrt(6.0 / (fan_in + fan_out))
    return (low + np.random.random((fan_in, fan_out)) * (high - low)).astype(np.float32)


def cost(x, y):
	x_gram = np.dot(x, x.T)
	y_gram = np.dot(y, y.T)
	return np.sum(np.square(y_gram - x_gram))


seed = 10
np.random.seed(seed)

batch_size = 200
input_size = 20
layer_size = 100
output_size = 2
dt = 0.1

x, classes = make_classification(
    n_samples=batch_size,
    n_features=input_size,
    n_classes=3,
    n_clusters_per_class=3*9,
    n_informative=9,
    random_state=seed
)

pca = dec.PCA(2)
x_pca = pca.fit(x).transform(x)
# shs(x_values_pca, labels=[classes]) 

W = xavier_init(input_size, layer_size)
Mbase = xavier_init(10, layer_size)
M = np.dot(Mbase.T, Mbase)



num_iters = 50

ysq = np.zeros((batch_size, layer_size))

for e in xrange(100):
	yh = np.zeros((num_iters, batch_size, layer_size))

	y = np.zeros((batch_size, layer_size))
	
	for t in xrange(num_iters):
		xt = x if t == 10 else np.zeros((batch_size, input_size))
		
		dy = np.maximum(np.dot(xt, W) - np.dot(y, M), 0.0) - y
		
		dW = np.dot((xt - np.dot(y, W.T)).T, y)
		dM = np.dot((y - np.dot(y, M.T)).T, y)
		
		ysq += dt * np.square(y)
		y += dt * dy

		W += (dt/1.0) * dW
		M += (dt/100.0) * dM

		yh[t] = y.copy()

	if e % 1 == 0:
		print "E {}, Cost {:.4f}".format(e, cost(x, yh[10]))


	# shs(y, labels=([str(c) for c in classes],))