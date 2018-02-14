
import numpy as np
import ctypes as ct

from binding import Config, MatrixFlat, Stat
from binding import run_model, get_structure_info
from util import *
from datasets import XorDataset, to_sparse_ts

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

np.random.seed(10)

F0 = xavier_init(input_size, layer_size)
F1 = xavier_init(layer_size, output_size)

c = Config()
c.Dt = 1.0
c.SynTau = 5.0
c.F0 = MatrixFlat.from_np(F0)
c.F1 = MatrixFlat.from_np(F1)



ds = XorDataset()
x, y = ds.next_train_batch()
xt = to_sparse_ts(x, seq_length, at=5, filter_size=seq_length-1).astype(np.float32)
yt = to_sparse_ts(y, seq_length, at=5, filter_size=seq_length-1).astype(np.float32)

xt = np.transpose(xt, (1, 0, 2))
yt = np.transpose(yt, (1, 0, 2))

s = run_model(
	c,
	xt,
	yt
)

dd = s.Input

shl(dd[0, :, 0], dd[0, :, 1], show=False)
shl(s.U[0,:,:], show=False)
shl(s.Output[0])