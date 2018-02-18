
import numpy as np
import ctypes as ct
import time
from binding import Config, MatrixFlat, Statistics, Data, State
from binding import run_model, get_structure_info
from util import *
from datasets import XorDataset, to_sparse_ts

np.random.seed(10)


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
	xt = to_sparse_ts(x, seq_length, at=3, filter_size=24).astype(np.float32)
	yt = to_sparse_ts(y, seq_length, at=3, filter_size=24).astype(np.float32)

	xt = np.transpose(xt, (1, 0, 2))
	yt = np.transpose(yt, (1, 0, 2))

	yt = np.concatenate((yt, yt), axis=2)
	# yt = np.concatenate((yt, np.zeros((yt.shape[0],yt.shape[1],1), dtype=np.float32)), axis=2)

	return (
		xt.reshape((xt.shape[0], xt.shape[1]*xt.shape[2])),
		yt.reshape((yt.shape[0], yt.shape[1]*yt.shape[2])),
	)


F0 = np.ones((input_size, layer_size), dtype=np.float32) * 0.5
# F0 = xavier_init(input_size, layer_size)
F1 = np.ones((layer_size, output_size), dtype=np.float32) * 0.2
# F1 = xavier_init(layer_size, output_size)

F0_copy = F0.copy()

c = Config()
c.Dt = 1.0
c.TauSyn = 5.0
c.FbFactor = 1.0
c.TauMean = 50.0
c.LearningRate = 0.1 * 1000.0
c.Lambda = 0.0
c.F0 = MatrixFlat.from_np(F0)
c.F1 = MatrixFlat.from_np(F1)



x = np.zeros((seq_length, batch_size, input_size), dtype=np.float32)
x[1,0,0] = 1.0
x[15,0,1] = 1.0
x[30,0,2] = 1.0
# x = smooth_batch_matrix(x, kernel=exp_filter).astype(np.float32).transpose((1, 0, 2))
x = x.transpose((1, 0, 2))

y = np.zeros((seq_length, batch_size, output_size), dtype=np.float32)
y[15,0,0] = 1.0
y[15,0,1] = 1.0
y = smooth_batch_matrix(y, kernel=exp_filter).astype(np.float32).transpose((1, 0, 2))

x = x.reshape((x.shape[0], x.shape[1]*x.shape[2]))
y = y.reshape((y.shape[0], y.shape[1]*y.shape[2]))
yt = y

# ds = XorDataset()
# x, y = ds.next_train_batch()
# x, y = preprocess(x, y)

# xt, yt = ds.next_test_batch()
# xt, yt = preprocess(xt, yt)


# trainInp = Data() 
# trainInp.Input = MatrixFlat.from_np(x)
# trainInp.Output = MatrixFlat.from_np(x)

st_train, st_test = State.alloc(), State.alloc()

epochs = 1000
dF0h = np.zeros((epochs, input_size, layer_size))

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
	dF0h[e] = np.mean(st.dF0, 0)

	F0 += c.LearningRate * st_train.dF0 
	
	F0 = F0/np.linalg.norm(F0, 1, axis=0)
	c.F0 = MatrixFlat.from_np(F0)
	
	if e % 10 == 0:
		t1 = time.time()
		print "Epoch {} ({:.3f}s), train error: {:.4f}, test error: {:.4f}".format(
				e,
				t1-t0,
				np.sum(np.square(rs(y) - st.Output)),
				np.sum(np.square(rs(yt) - sv.Output))
			)
		t0 = time.time()

# shl(dF0h[:,0,0], dF0h[:,1,0], dF0h[:,2,0], labels=["0","1","2"])

shl(st.dF0[:,:,0], show=False, title="dF0")
shl(st.De[0,:,0], sv.De[0,:,0], show=False, title="De", labels=["Train", "Test"])
shl(st.A[0,:,0], sv.A[0, :, 0], title="A", labels=["Train", "Test"])



# shl(ts.U[1,:,:], show=False)
# shl(vs.U[1,:,:], show=True)
# shl(st.A[0], title="Output")
# shl(st.Output[0], title="Output")

# shl(ts.De[1], title="De", show=False)
# shl(ts.Output[1], title="Output")
