
import numpy as np
import ctypes as ct
import time
from binding import *
from util import *
from datasets import *
from sklearn.metrics import log_loss

np.random.seed(10)

def xavier_init(fan_in, fan_out, const=1.0):
    low = -const * np.sqrt(6.0 / (fan_in + fan_out))
    high = const * np.sqrt(6.0 / (fan_in + fan_out))
    return (low + np.random.random((fan_in, fan_out)) * (high - low)).astype(np.float32)


ds = ToyDataset()
x, y = ds._x_v, ds._y_v
# y = np.concatenate((y, y), axis=1) # use it for 1 layer output

xt, yt = ds._xt_v, ds._yt_v
# yt = np.concatenate((yt, yt), axis=1) # use it for 1 layer output


c = NetConfig(
    Dt = 1.0,
    LearningRate=0.01,
    FeedbackDelay=1,
    OutputTau=5.0,
    DeStat = np.zeros((batch_size, seq_length, output_size)),
    YMeanStat = np.zeros((batch_size, seq_length, output_size))
)

net = (
    LayerConfig(
        TauSoma = 1.0,
        TauSyn = 5.0,
        TauMean = 100.0,
        ApicalGain = 1.0,
        FbFactor = 0.0,
        Act = RELU,
        W = xavier_init(input_size, layer_size),
        B = np.zeros((1, layer_size)),
        dW = np.zeros((input_size, layer_size)),
        dB = np.zeros((1, layer_size)),
        UStat = np.zeros((batch_size, seq_length, layer_size)),
        AStat = np.zeros((batch_size, seq_length, layer_size)),
        FbStat = np.zeros((batch_size, seq_length, layer_size)),
    ),
    LayerConfig(
        TauSoma = 1.0,
        TauSyn = 1.0,
        TauMean = 100.0,
        ApicalGain = 1.0,
        FbFactor = 0.0,
        Act = RELU,
        W = xavier_init(layer_size, output_size),
        B = np.zeros((1, output_size)),
        dW = np.zeros((layer_size, output_size)),
        dB = np.zeros((1, output_size)),
        UStat = np.zeros((batch_size, seq_length, output_size)),
        AStat = np.zeros((batch_size, seq_length, output_size)),
        FbStat = np.zeros((batch_size, seq_length, output_size)),
    ),
)

l0, l1 = net

run_model(
    100,
    net,
    c,
    x,
    y,
    xt,
    yt
)

fbStat0 = l0.get("FbStat")
fbStat1 = l1.get("FbStat")

AStat1 = l1.get("AStat")
print np.mean(np.not_equal(np.argmax(AStat1[:,10,:], axis=1), np.argmax(yt,axis=1)))


