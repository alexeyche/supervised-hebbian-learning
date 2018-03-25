
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

x, y = ds.train_data
# y = np.concatenate((y, y), axis=1) # use it for 1 layer output

xt, yt = ds.test_data

# x, y = x[:1000], y[:1000]
# xt, yt = xt[:1000], yt[:1000]

# yt = np.concatenate((yt, yt), axis=1) # use it for 1 layer output

_, input_size, _, output_size = x.shape + y.shape
batch_size = 40
seq_length = 50
layer_size = 50

c = NetConfig(
    Dt = 1.0,
    SeqLength=seq_length,
    BatchSize=batch_size,
    LearningRate=0.001,
    FeedbackDelay=1,
    OutputTau=5.0,
    DeStat = np.zeros((batch_size, seq_length, output_size)),
    YMeanStat = np.zeros((batch_size, seq_length, output_size))
)

net = (
    LayerConfig(
        Size = layer_size,
        TauSoma = 1.0,
        TauSyn = 5.0,
        TauMean = 100.0,
        ApicalGain = 1.0,
        FbFactor = 0.0,
        Act = RELU,
        GradProc = NO_GRADIENT_PROCESSING,
        W = xavier_init(input_size, layer_size),
        B = np.zeros((1, layer_size)),
        dW = np.zeros((input_size, layer_size)),
        dB = np.zeros((1, layer_size)),
        UStat = np.zeros((batch_size, seq_length, layer_size)),
        AStat = np.zeros((batch_size, seq_length, layer_size)),
        FbStat = np.zeros((batch_size, seq_length, layer_size)),
    ),
    LayerConfig(
        Size = output_size,
        TauSoma = 1.0,
        TauSyn = 1.0,
        TauMean = 100.0,
        ApicalGain = 1.0,
        FbFactor = 0.0,
        Act = RELU,
        GradProc = NO_GRADIENT_PROCESSING,
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
    1000,
    net,
    c,
    x,
    y,
    xt,
    yt,
    test_freq = 200
)

fbStat0 = l0.get("FbStat")
fbStat1 = l1.get("FbStat")

AStat0 = l0.get("AStat")
AStat1 = l1.get("AStat")
print np.mean(np.not_equal(np.argmax(AStat1[:,10,:], axis=1), np.argmax(yt,axis=1)))


