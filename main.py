
import numpy as np
import ctypes as ct
import time
from binding import *
from util import *
from datasets import XorDataset, to_sparse_ts

np.random.seed(12)

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


def xavier_init(fan_in, fan_out, const=0.5):
    low = -const * np.sqrt(6.0 / (fan_in + fan_out))
    high = const * np.sqrt(6.0 / (fan_in + fan_out))
    return (low + np.random.random((fan_in, fan_out)) * (high - low)).astype(np.float32)




c = Config(
	Dt = 1.0,
	LearningRate=1.0,
	FeedbackDelay=1
)


ds = XorDataset()
x, y = ds._x_v, ds._y_v

x, y = preprocess(x, y)

xt, yt = ds._x_v, ds._y_v
xt, yt = preprocess(xt, yt)


ls0 = LayerState(
    TauSoma = 2.0,
    TauMean = 100.0,
    ApicalGain = 1.0,
    Act = RELU,
    F = xavier_init(input_size, layer_size),
    UStat = np.zeros((batch_size, seq_length, layer_size)),
    AStat = np.zeros((batch_size, seq_length, layer_size)),
)

ls1 = LayerState(
    TauSoma = 2.0,
    TauMean = 100.0,
    ApicalGain = 1.0,
    Act = SIGMOID,
    F = xavier_init(layer_size, output_size),
    UStat = np.zeros((batch_size, seq_length, output_size)),
    AStat = np.zeros((batch_size, seq_length, output_size)),
)



run_model(
	1,
	(ls0, ls1),
	c,
	x,
	y,
	xt,
	yt
)

UStat0 = ls0.get_matrix("UStat")
AStat0 = ls0.get_matrix("AStat")

UStat1 = ls1.get_matrix("UStat")
AStat1 = ls1.get_matrix("AStat")