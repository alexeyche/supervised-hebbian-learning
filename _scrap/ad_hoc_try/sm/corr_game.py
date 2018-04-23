



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

layer_size = 100

lrate = 0.01
p = 0.09

W = xavier_init(input_size, layer_size, const=0.01)
Mbase = xavier_init(4, layer_size, const=0.1)
M = np.dot(Mbase.T, Mbase)
np.fill_diagonal(M, 0.0)


Wo = xavier_init(layer_size, output_size, const=0.01)
Mobase = xavier_init(4, output_size, const=0.1)
Mo = np.dot(Mobase.T, Mobase)
np.fill_diagonal(Mo, 0.0)

# W = xavier_init(input_size, layer_size)


opt = SGDOpt((0.005,0.001,0.001)) #,0.001))
opt.init(W, M, Wo, Mo)

epochs = 3000
metrics = np.zeros((epochs, 4))

C = np.cov(x.T)
eigval, eigvec = np.linalg.eig(C)


for e in xrange(epochs):
	y = np.dot(x, np.dot(np.linalg.inv(M), W.T).T)
	yo = np.dot(y, Wo)
	
	dW = np.dot(x.T, y)/batch_size - W
	dM = np.dot(y.T, y)/batch_size - M # np.eye(layer_size)	
	dWo = np.dot(y.T, yv)/batch_size - Wo
	
	opt.update(-dW, -dM, -dWo)

	xtv, ytv = ds.next_test_batch()
	yt = np.dot(xtv, np.dot(np.linalg.inv(M), W.T).T)
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
		print "Epoch {}, {}".format(
			e,
			", ".join(["{:.4f}".format(m) for m in metrics[e, :]])
		)

C = np.dot(xv.T, yv)/batch_size
b = np.dot(np.linalg.inv(np.dot(xv.T, xv)/batch_size), C)
print np.mean(np.not_equal(np.argmax(np.dot(xv, b), 1), np.argmax(yv, 1)))


C = np.dot(xv.T, yv)/batch_size
Cy = np.dot(y.T, y)/batch_size
eigvalCy, eigvecCy = np.linalg.eig(Cy)
