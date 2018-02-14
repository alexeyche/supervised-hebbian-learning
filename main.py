
import numpy as np
import ctypes as ct

from binding import Config, MatrixFlat, Stat
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

def preprocess(x, y):
	xt = to_sparse_ts(x, seq_length, at=3, filter_size=24).astype(np.float32)
	yt = to_sparse_ts(y, seq_length, at=3, filter_size=24).astype(np.float32)

	xt = np.transpose(xt, (1, 0, 2))
	yt = np.transpose(yt, (1, 0, 2))

	yt = np.concatenate((yt, np.zeros((yt.shape[0],yt.shape[1],1), dtype=np.float32)), axis=2)
	return xt, yt



F0 = xavier_init(input_size, layer_size)
F1 = xavier_init(layer_size, output_size)

c = Config()
c.Dt = 0.1
c.SynTau = 5.0
c.FbFactor = 1.0
c.F0 = MatrixFlat.from_np(F0)
c.F1 = MatrixFlat.from_np(F1)


ds = XorDataset()
x, y = ds.next_train_batch()
x, y = preprocess(x, y)

xt, yt = ds.next_test_batch()
xt, yt = preprocess(xt, yt)


s = run_model(
	c,
	x,
	y,
	xt,
	yt
)

shl(s.Input[1, :, 0], s.Input[1, :, 1], show=False)
shl(s.U[1,:,:], show=False)
shl(s.De[1], title="De", show=False)
shl(s.Output[1], title="Output")
