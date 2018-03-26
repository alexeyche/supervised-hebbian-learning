
import numpy as np
import ctypes as ct
import time
from binding import *
from util import *
from datasets import *
from sklearn.metrics import log_loss
from poc.common import Relu

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
layer_size = 45

c = NetConfig(
    Dt = 1.0,
    SeqLength=seq_length,
    BatchSize=batch_size,
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
        TauMean = 500.0,
        ApicalGain = 1.0,
        FbFactor = 0.0,
        TauGrad = 100.0,
        LearningRate=0.01,
        Act = RELU,
        GradProc = HEBB,
        W = xavier_init(input_size, layer_size),
        B = np.zeros((1, layer_size)),
        dW = np.zeros((input_size, layer_size)),
        dB = np.zeros((1, layer_size)),
        Am = np.zeros((1, layer_size)),
        UStat = np.zeros((batch_size, seq_length, layer_size)),
        AStat = np.zeros((batch_size, seq_length, layer_size)),
        FbStat = np.zeros((batch_size, seq_length, layer_size)),
    ),
    LayerConfig(
        Size = output_size,
        TauSoma = 1.0,
        TauSyn = 1.0,
        TauMean = 0.0,
        ApicalGain = 1.0,
        FbFactor = 0.0,
        TauGrad = 10.0,
        LearningRate=0.0,
        Act = RELU,
        GradProc = NO_GRADIENT_PROCESSING,
        W = xavier_init(layer_size, output_size),
        B = np.zeros((1, output_size)),
        dW = np.zeros((layer_size, output_size)),
        dB = np.zeros((1, output_size)),
        Am = np.zeros((1, output_size)),
        UStat = np.zeros((batch_size, seq_length, output_size)),
        AStat = np.zeros((batch_size, seq_length, output_size)),
        FbStat = np.zeros((batch_size, seq_length, output_size)),
    ),
)


trainStats, testStats = run_model(
    400,
    net,
    c,
    x,
    y,
    xt,
    yt,
    test_freq = 50
)


l0 = net[0]
l1 = net[1]

# dd = np.zeros((50,))
# for i in xrange(50):
#     de = np.dot(l1.get("W"), l1.get("FbStat")[:,i].T) * Relu().deriv(l0.get("AStat")[:,i]).T
#     den = l0.get("FbStat")[:,i].T

#     dd[i] = np.mean(np.equal(np.sign(den), np.sign(de)))


# shl(de[0], den[0]/15.0)

st = trainStats
SquaredError = st.get("SquaredError")
ClassificationError = st.get("ClassificationError")
SignAgreement = st.get("SignAgreement")
AverageActivity = st.get("AverageActivity")
Sparsity = st.get("Sparsity")

shl(l0.get("AStat")[0,:])
