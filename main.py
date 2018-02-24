
import numpy as np
import ctypes as ct
import time
from binding import Config, MatrixFlat, Statistics, Data, State
from binding import run_model, get_structure_info
from util import *
from datasets import XorDataset, to_sparse_ts

np.random.seed(10)

def relu_deriv(x):
    if isinstance(x, float):
        return 1.0 if x > 0.0 else 0.0
    dadx = np.zeros(x.shape)
    dadx[np.where(x > 0.0)] = 1.0
    return dadx

def xavier_init(fan_in, fan_out, const=0.5):
    low = -const * np.sqrt(6.0 / (fan_in + fan_out))
    high = const * np.sqrt(6.0 / (fan_in + fan_out))
    return (low + np.random.random((fan_in, fan_out)) * (high - low)).astype(np.float32)


struc_info = get_structure_info()

input_size = struc_info.InputSize
layer_size = struc_info.LayerSize
output_size = struc_info.OutputSize
batch_size = struc_info.BatchSize
layers_num = struc_info.LayersNum
seq_length = struc_info.SeqLength

def rs(m):
	return m.reshape((m.shape[0], seq_length, m.shape[1]/seq_length))

def preprocess(x, y):
	xt = to_sparse_ts(x, seq_length, at=10, filter_size=2*24, kernel=gauss_filter, sigma=0.005).astype(np.float32)
	yt = to_sparse_ts(y, seq_length, at=10, filter_size=2*24, kernel=gauss_filter, sigma=0.005).astype(np.float32)

	xt = np.transpose(xt, (1, 0, 2))
	yt = np.transpose(yt, (1, 0, 2))

	yt = np.concatenate((yt, yt), axis=2)
	# yt = np.concatenate((yt, np.zeros((yt.shape[0],yt.shape[1],1), dtype=np.float32)), axis=2)

	return (
		xt.reshape((xt.shape[0], xt.shape[1]*xt.shape[2])),
		yt.reshape((yt.shape[0], yt.shape[1]*yt.shape[2])),
	)

def norm(f, l):
	return f/np.linalg.norm(f, l, axis=0)


# F0 = np.ones((input_size, layer_size), dtype=np.float32)
F0 = xavier_init(input_size, layer_size)



# R0 = 0.0*(
# 	np.dot(F0.T, F0) - np.eye(layer_size, dtype=np.float32)
# )

# F1 = np.ones((layer_size, output_size), dtype=np.float32) 
# F1 = norm(F1) * 0.5

F1 = xavier_init(layer_size, output_size)


R0 = np.abs(xavier_init(layer_size, layer_size))

F0_copy = F0.copy()

c = Config()
c.Dt = 1.0
c.TauSyn = 2.0
c.FbFactor = 1.0
c.TauMean = 100.0
c.Threshold = 0.1
c.LearningRate = 0.1 * 1.0
c.Lambda = 0.02

F0 = 5.0*norm(F0, 1)
F1 = 5.0*norm(F1, 1)

# R0 = 0.5*norm(R0, l=2)

c.F0 = MatrixFlat.from_np(F0)
c.R0 = MatrixFlat.from_np(R0)
c.F1 = MatrixFlat.from_np(F1)



# x = np.zeros((seq_length, batch_size, input_size), dtype=np.float32)
# x[1,0,0] = 1.0
# x[10,0,1] = 1.0
# x[20,0,2] = 1.0
# x[30,0,3] = 1.0
# x[40,0,4] = 1.0
# x = smooth_batch_matrix(x, kernel=gauss_filter, sigma=0.0025).astype(np.float32).transpose((1, 0, 2))
# # x = x.transpose((1, 0, 2))

# y = np.zeros((seq_length, batch_size, output_size), dtype=np.float32)
# y[19,0,0] = 1.0
# y[19,0,1] = 1.0
# y = smooth_batch_matrix(y, kernel=gauss_filter, sigma=0.0025).astype(np.float32).transpose((1, 0, 2))

# x = x.reshape((x.shape[0], x.shape[1]*x.shape[2]))
# y = y.reshape((y.shape[0], y.shape[1]*y.shape[2]))
# yt = y

ds = XorDataset()
x, y = ds.next_train_batch()
x, y = preprocess(x, y)

xt, yt = ds.next_test_batch()
xt, yt = preprocess(xt, yt)


# trainInp = Data() 
# trainInp.Input = MatrixFlat.from_np(x)
# trainInp.Output = MatrixFlat.from_np(x)

st_train, st_test = State.alloc(), State.alloc()

epochs = 1
dF0h = np.zeros((epochs, input_size, layer_size))
dF1h = np.zeros((epochs, layer_size, output_size))
# A0mh = np.zeros((epochs, batch_size, layer_size))

# F0 = norm(F0)


# dA0h = np.zeros((epochs*seq_length, batch_size, layer_size))
# A0h = np.zeros((epochs*seq_length, batch_size, layer_size))
# Outputh = np.zeros((epochs*seq_length, batch_size, output_size))

t0 = time.time()

for e in xrange(epochs):
	st, sv = run_model(
		1,
		c,
		st_train, 
		st_test,
		x,
		y,
		x,
		y
	)
	
	dF0h[e] = st_train.dF0
	dF1h[e] = st_train.dF1
	# A0mh[e] = st_train.A0m

	dF0 = np.zeros(F0.shape)

	xx = rs(x)
	for i in xrange(seq_length):
		A = st.A[:, i, :]
		X = xx[:,i,:] #- np.dot(A, F0.T)
		
		# A = np.maximum(A-0.1, 0.0)
		
		dF0 += np.dot(X.T, A) / seq_length
	
	# Outputh[(e*seq_length):((e+1)*seq_length)] = np.transpose(st.Output, (1, 0, 2)).copy()
	# A0h[(e*seq_length):((e+1)*seq_length)] = np.transpose(st.A, (1, 0, 2)).copy()
	# dA0h[(e*seq_length):((e+1)*seq_length)] = np.transpose(st.dA, (1, 0, 2)).copy()
	
	if e > 0:
		F0 += 10.0 * c.LearningRate * st_train.dF0
		F1 += 10.0 * c.LearningRate * st_train.dF1
		
		F0 = 5.0*norm(F0, 1)
		F1 = 5.0*norm(F1, 1)
		
		c.F0 = MatrixFlat.from_np(F0)
		c.F1 = MatrixFlat.from_np(F1)


	if e % 100 == 0:
		
		t1 = time.time()
		print "Epoch {} ({:.3f}s), error: {:.3f}, t.error: {:.3f}, |fb|: {:.3f}".format(
				e,
				t1-t0,
				np.sum(np.square(rs(y) - st.Output)),
				np.sum(np.square(rs(yt) - sv.Output)),
				np.linalg.norm(st.De)
			)
		t0 = time.time()


## Ways to solve this problem:
## 1) Second averaging mechanism on a faster scale
## 2) Neuron care only about super active nodes
## 3) Powerful and fast inhibition

# shm(st.A[1])

shl(dF0[0], st_train.dF0[0], show=False, title="0 syn", labels=["Fake", "Real"])
shl(dF0[1], st_train.dF0[1], show=False, title="1 syn", labels=["Fake", "Real"])


de = np.dot(st.De[1], F1.T) * relu_deriv(st.A[1])


shl(st.A[1,10,:], sv.A[1,10,:], de[10,:], labels=["A", "Av", "De"], show=False)


diff = st.A[1]-de
diff_idx = np.argsort(np.sum(np.square(diff),0))[-5:]
print diff_idx


id=17
mean_train = np.tile(st_train.A0m[1, id], seq_length)
mean_test = np.tile(st_test.A0m[1, id], seq_length)
shl(st.A[1,:,id], sv.A[1,:,id], de[:,id], mean_train, mean_test, labels=["A", "Av", "De", "A0m train", "A0m test"], show=False)

plt.show()


# shl(de[:,2], st.A[1][:,2])

# shl(st.A[1][:,0], sv.A[1][:,0], de[:,0], labels=["A", "Av", "De"], show=False)
# shl(st.A[1][:,1], sv.A[1][:,1], de[:,1], labels=["A", "Av", "De"], show=False)
# shl(st.A[1][:,2], sv.A[1][:,2], de[:,2], labels=["A", "Av", "De"], show=False)
# plt.show()

