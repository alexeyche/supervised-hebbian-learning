
import numpy as np
import ctypes as ct

from binding import Config, MatrixFlat, Stat
from binding import run_model, get_structure_info
from util import *

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



input_data = np.zeros((batch_size, seq_length, input_size)).astype(np.float32)
input_data[0,5,0] = 1.0
input_data[0,5,1] = 1.0

s = run_model(
	c,
	input_data
)

dd = s.Input

shl(dd[0, :, 0], dd[0, :, 1], show=False)
shl(s.U[0,:,:], show=False)
shl(s.Output[0])