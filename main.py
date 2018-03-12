
import numpy as np
import ctypes as ct
import time
from binding import Config, MatrixFlat, Statistics, Data, State
from binding import run_model, get_structure_info
from util import *
from datasets import XorDataset, to_sparse_ts

np.random.seed(12)

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
c.TauMeanLong = 200.0
c.Threshold = 0.0
c.LearningRate = 0.1 * 1.0
c.Lambda = 0.02
c.FeedbackDelay = 15
c.ApicalGain = 10.0

# F0 = 0.1*norm(F0, 1)
# F1 = 0.1*norm(F1, 1)

# R0 = 0.5*norm(R0, l=2)

c.F0 = MatrixFlat.from_np(F0)
c.R0 = MatrixFlat.from_np(R0)
c.F1 = MatrixFlat.from_np(F1)



# x = np.zeros((seq_length, batch_size, input_size), dtype=np.float32)
# x[5,0,0] = 1.0
# x[25,0,1] = 1.0
# x[45,0,2] = 1.0
# x[65,0,3] = 1.0
# x[85,0,4] = 1.0
# x = smooth_batch_matrix(x, kernel=gauss_filter, sigma=0.0025).astype(np.float32).transpose((1, 0, 2))
# # x = x.transpose((1, 0, 2))

# y = np.zeros((seq_length, batch_size, output_size), dtype=np.float32)
# y[44,0,0] = 1.0
# y[44,0,1] = 1.0
# y = smooth_batch_matrix(y, kernel=gauss_filter, sigma=0.0025).astype(np.float32).transpose((1, 0, 2))

# x = x.reshape((x.shape[0], x.shape[1]*x.shape[2]))
# y = y.reshape((y.shape[0], y.shape[1]*y.shape[2]))
# yt = y

ds = XorDataset()
x, y = ds.next_train_batch()
# wt = 0.1*np.random.randn(2, input_size)
# x = np.dot(x, wt)

x, y = preprocess(x, y)

xt, yt = ds.next_test_batch()
# xt = np.dot(xt, wt)
xt, yt = preprocess(xt, yt)



st_train, st_test = State.alloc(), State.alloc()

epochs = 1
dF0h = np.zeros((epochs, input_size, layer_size))
dF1h = np.zeros((epochs, layer_size, output_size))
A0mh = np.zeros((epochs, batch_size, layer_size))
A0mmh = np.zeros((epochs, batch_size, layer_size))

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

	if np.any([np.any(np.isnan(getattr(st, v))) for v in dir(st) if v[:2] != "__"]):
		print "NAN"
		break
	
	dF0h[e] = st_train.dF0
	dF1h[e] = st_train.dF1
	A0mh[e] = st_train.A0m
	A0mmh[e] = st_train.A0mm

	# dF0 = np.sum([np.dot(st.Im[b].T, st.A[b]-1.05*st.Am[b]) for b in xrange(batch_size)], 0) ** 3
	# dF0 = np.sum([np.dot(st.Im[b].T, st.A[b]-st.Am[b]) for b in xrange(batch_size)], 0)

	# dF0 = 0.01*np.mean([np.dot(st.I[b].T, st.A[b]-0.02) for b in xrange(batch_size)], 0)
	
	## work'ish:
	dF0 = 0.1*np.mean([
		np.dot(
			st.I[b].T, 
			np.maximum(np.dot(st.De[b], F1.T) * relu_deriv(st.A[b]), 0.0)
		) for b in xrange(batch_size)
	], 0)/seq_length

	# dF0 = 0.1*np.mean([np.dot(st.I[b].T, st.A[b]-sv.A[b]) for b in xrange(batch_size)], 0)

	# dF0 *= 10.0

	# Outputh[(e*seq_length):((e+1)*seq_length)] = np.transpose(st.Output, (1, 0, 2)).copy()
	# A0h[(e*seq_length):((e+1)*seq_length)] = np.transpose(st.A, (1, 0, 2)).copy()
	# dA0h[(e*seq_length):((e+1)*seq_length)] = np.transpose(st.dA, (1, 0, 2)).copy()
	
	if e > 30:
		F0 += 1.0 * c.LearningRate * dF0 #- F0*0.001
		F1 += 1.0 * c.LearningRate * st_train.dF1 #- F1*0.001
		
		# F0 = 1.0*norm(F0, 1)
		# F1 = 1.0*norm(F1, 1)
		
		# c.F0 = MatrixFlat.from_np(F0)
		# c.F1 = MatrixFlat.from_np(F1)

		# print "|F0| {:.3f} |F1| {:.3f}".format(np.linalg.norm(F0,2), np.linalg.norm(F1,2))
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


print sv.Output[:,10]

#####
# 1. Use apical dendritic source to drive learning
# 2. Use signs of gradient


# shl(dF0[0], st_train.dF0[0], show=False, title="0 syn", labels=["Fake", "Real"])
# shl(dF0[1], st_train.dF0[1], show=False, title="1 syn", labels=["Fake", "Real"])
# shl(dF0[2], st_train.dF0[2], show=False, title="2 syn", labels=["Fake", "Real"])


# de = lambda bi: np.dot(rs(y)[bi] - st.Output[bi], F1.T) * relu_deriv(st.A[bi])
de_fb = np.dot(st.De[1], F1.T) * relu_deriv(st.A[b])


# shl(np.mean(st.A[0] - sv.A[0], 0), np.mean(de(0),0))

shl(
	np.mean(st.A[1],0), # - st_train.A0m[1]), 
	np.mean(de_fb, 0),
	labels=["A", "De"]
)


# plt.show()

# diff = st.A[1]-de
# diff_idx = np.argsort(np.sum(np.square(diff),0))[-5:]
# print diff_idx


# id=9
# shl(st.A[1,:,id], sv.A[1,:,id], de_fb[:,id], st.Am[1,:,id], st.Amm[1,:,id], labels=["A", "Av", "De", "A0m", "A0mm"], show=False)

# plt.show()


# shl(de[:,2], st.A[1][:,2])

# shl(st.A[1][:,0], sv.A[1][:,0], de[:,0], labels=["A", "Av", "De"], show=False)
# shl(st.A[1][:,1], sv.A[1][:,1], de[:,1], labels=["A", "Av", "De"], show=False)
# shl(st.A[1][:,2], sv.A[1][:,2], de[:,2], labels=["A", "Av", "De"], show=False)
# plt.show()



# Epoch 0 (0.001s), error: 17.638, t.error: 17.489, |fb|: 4.199
# Epoch 1000 (0.803s), error: 17.638, t.error: 17.489, |fb|: 4.199
# Epoch 2000 (0.699s), error: 17.638, t.error: 17.489, |fb|: 4.199
# Epoch 3000 (0.690s), error: 17.638, t.error: 17.489, |fb|: 4.199
# Epoch 4000 (0.693s), error: 17.638, t.error: 17.489, |fb|: 4.19