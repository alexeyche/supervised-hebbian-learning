
import numpy as np
from poc.opt import *
from datasets import *
from poc.common import *
from cost import *

seed = 10
np.random.seed(seed)

act = Relu()

layer_size = 100
dt = 0.01
omega = 1.0
k = 1.0


batch_size = 1
num_iters = 2000

x = np.asarray([np.sin(np.linspace(0, 0.01*num_iters, num_iters) + np.pi)]).T
x = np.expand_dims(x, 1)

input_size = 1


threshold = np.repeat(0.0,layer_size)

W = np.random.random((input_size, layer_size)) * 0.1 - 0.05
# W = norm(W)

M = np.random.random((layer_size, layer_size)) * 0.1 - 0.05
# M = np.dot(W.T, W)
np.fill_diagonal(M, 0.0)


reg = 0.0
opt = SGDOpt((0.001,0.01))
opt.init(W, M)

tau_m = 300.0

am = np.zeros((layer_size, ))
for e in xrange(300):
	dW = np.zeros(W.shape)
	dM = np.zeros(M.shape)

	a = np.zeros((batch_size, layer_size))
	ah = np.zeros((num_iters, batch_size, layer_size))
	xh = np.zeros((num_iters, batch_size, input_size))

	for i in xrange(num_iters):
		c = np.dot(x[i], W) - np.dot(a, M)
		a += dt * (act(c - threshold) - a)
		
		ah[i] = a.copy()

		dW += np.dot((x[i] - np.dot(a, W.T)).T, a)
		# dM += np.dot((a - np.dot(a, M.T)).T, a) - reg * M	
		dM += np.dot(a.T, a)



		am += (np.mean(a, 0) - am) / tau_m
		
		xh[i] = np.dot(a, W.T)

		threshold = am

	opt.update(-dW, -dM)
	# M = np.dot(W.T, W)
	np.fill_diagonal(M, 0.0)
	
	mse = np.mean(np.square(xh/np.linalg.norm(xh) - x/np.linalg.norm(x)))
	print e, mse

shl(x/np.linalg.norm(x), xh/np.linalg.norm(xh))