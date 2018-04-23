



import numpy as np
from poc.opt import *
from datasets import *
from poc.common import *

def xavier_init(fan_in, fan_out, const=1.0):
    low = -const * np.sqrt(6.0 / (fan_in + fan_out))
    high = const * np.sqrt(6.0 / (fan_in + fan_out))
    return (low + np.random.random((fan_in, fan_out)) * (high - low)).astype(np.float32)

ds = ToyDataset()
act = Relu()


seed = 2
np.random.seed(seed)

(batch_size, input_size), (_, output_size) = ds.train_shape

xv, yv = ds.next_train_batch()
xtv, ytv = ds.next_test_batch()


layer_size = output_size
lrate = 0.01
p = 0.09
W = np.zeros((input_size, layer_size))
# W = xavier_init(input_size, layer_size)


opt = SGDOpt((0.01,))
opt.init(W)

epochs = 100
metrics = np.zeros((epochs, 3))

C = np.dot(xv.T, yv)/batch_size
Ct = np.dot(xtv.T, ytv)/batch_size

print "Bound is {:.4f}".format(np.mean(
	np.not_equal(
		np.argmax(np.dot(xtv, C), 1), 
		np.argmax(ytv, 1)
	)
))

for e in xrange(epochs):
	y = np.dot(xv, W)
	# y = np.dot(xv, np.dot(xv.T, yv)/batch_size)

	dW = np.dot(xv.T, yv)/batch_size - W

	opt.update(-dW)

	yto = np.dot(xtv, W)

	metrics[e, 0] = np.linalg.norm(yto - ytv)
	metrics[e, 1] = np.linalg.norm(W - C)
	metrics[e, 2] = np.mean(
		np.not_equal(
			np.argmax(yto, 1), 
			np.argmax(ytv, 1)
		)
	)

	if e % 10 == 0:
		print "Epoch {}, {}".format(
			e,
			", ".join(["{:.4f}".format(m) for m in metrics[e, :]])
		)

xv0 =xv[np.where(np.argmax(yv, axis=1) == 0)[0]]
xv1 =xv[np.where(np.argmax(yv, axis=1) == 1)[0]]

m0 = np.mean(xv0, 0)
m1 = np.mean(xv1, 0)

s0 = np.dot((xv0 - m0).T, xv0 - m0)/len(xv0)
s1 = np.dot((xv1 - m1).T, xv1 - m1)/len(xv1)

d0 = np.mean(np.square(xtv-m0)/np.diag(s0), 1)
d1 = np.mean(np.square(xtv-m1)/np.diag(s1), 1)
print "NB:", np.mean(np.not_equal(np.argmin(np.asarray([d0, d1]), 0), np.argmax(ytv, 1)))

# s0 = np.mean(xv0 - m0)


b = np.dot(np.linalg.inv(np.dot(xv.T, xv)/batch_size), C)
print "Linear:", np.mean(np.not_equal(np.argmax(np.dot(xtv, b), 1), np.argmax(ytv, 1)))

## So
## When input variables are decorrelated we could use only correlation between x and y