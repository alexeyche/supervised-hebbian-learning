
import numpy as np
from poc.opt import *
from datasets import *
from poc.common import *
from sm.sm_cost import *

def xavier_init(fan_in, fan_out, const=1.0):
    low = -const * np.sqrt(6.0 / (fan_in + fan_out))
    high = const * np.sqrt(6.0 / (fan_in + fan_out))
    return (low + np.random.random((fan_in, fan_out)) * (high - low)).astype(np.float32)

ds = ToyDataset()
act = Relu()



seed = 1
np.random.seed(seed)

(batch_size, input_size), (_, output_size) = ds.train_shape

xv, yv = ds.next_train_batch()
x = xv

Gx = np.dot(xv, xv.T) / input_size / batch_size
eigvalGx, eigvecGx = np.linalg.eig(Gx)
eigvecGx = np.real(eigvecGx[:, :input_size])
assert np.abs(np.real(eigvalGx[input_size])) < 1e-10, "Bad results of eigen decomposition"


layer_size = 100

lrate = 0.01
p = 0.09
num_iters = 25
dt = 1.0

W = xavier_init(input_size, layer_size, const=0.01)
Mbase = xavier_init(4, layer_size, const=0.1)
M = np.dot(Mbase.T, Mbase)
np.fill_diagonal(M, 0.0)


Wo = xavier_init(layer_size, output_size, const=0.01)
Mobase = xavier_init(4, output_size, const=0.1)
Mo = np.dot(Mobase.T, Mobase)
np.fill_diagonal(Mo, 0.0)

# W = xavier_init(input_size, layer_size)


opt = SGDOpt((0.0005,0.0001,0.0001)) #,0.001))
opt.init(W, M, Wo)

epochs = 3000
metrics = np.zeros((epochs, 5))

C = np.cov(x.T)
eigval, eigvec = np.linalg.eig(C)


for e in xrange(epochs):
	yh = np.zeros((num_iters, batch_size, layer_size))
	y = np.zeros((batch_size, layer_size))

	for it in xrange(num_iters):
		y += dt * (act(np.dot(x, W) - np.dot(y, M)) - y)
		
		yh[it] = y.copy()

	# y = np.dot(x, np.dot(np.linalg.inv(M), W.T).T)

	# y = np.dot(x, np.dot(np.linalg.inv(M), W.T).T)
	yo = np.dot(y, Wo)
	
	dW = np.dot(x.T, y)/batch_size - W
	dM = np.dot(y.T, y)/batch_size - M # np.eye(layer_size)	
	dWo = np.dot(y.T, yv)/batch_size - Wo
	
	opt.update(-dW, -dM, -dWo)

	xtv, ytv = ds.next_test_batch()
	yt = np.zeros((xtv.shape[0], layer_size))

	for it in xrange(num_iters):
		yt += dt * (np.dot(xtv, W) - np.dot(yt, M))

	yto = np.dot(yt, Wo)

	metrics[e, 0] = np.linalg.norm(yto - ytv)
	metrics[e, 1] = np.mean(
		np.not_equal(
			np.argmax(yto, 1), 
			np.argmax(ytv, 1)
		)
	)
	metrics[e, 2] = cmds_cost(xtv, yt)
	metrics[e, 3] = eigvec_cost(W, M, eigvec)
	

	# if metrics[e, 3] < 1e-04:
	# 	break
	if e % 100 == 0:
		metrics[e, 4] = eigvec_cost_gram(eigvecGx, y)

		print "Epoch {}, {}".format(
			e,
			", ".join(["{:.4f}".format(m) for m in metrics[e, :]])
		)

C = np.dot(xv.T, yv)/batch_size
b = np.dot(np.linalg.inv(np.dot(xv.T, xv)/batch_size), C)
print np.mean(np.not_equal(np.argmax(np.dot(xv, b), 1), np.argmax(yv, 1)))

Gy = np.dot(y, y.T) / y.shape[0] / y.shape[1]
eigvalGy, eigvecGy = np.linalg.eig(Gy)
eigvecGy = np.real(eigvecGy[:, :eigvecGx.shape[1]])

