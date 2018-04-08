
import numpy as np
import ctypes as ct
import time
from binding import *
from util import *
from datasets import *
from sklearn.metrics import log_loss
from poc.common import Relu
from poc.opt import *

np.random.seed(12)

def xavier_init(fan_in, fan_out, const=1.0):
    low = -const * np.sqrt(6.0 / (fan_in + fan_out))
    high = const * np.sqrt(6.0 / (fan_in + fan_out))
    return (low + np.random.random((fan_in, fan_out)) * (high - low)).astype(np.float32)


ds = ToyDataset()

x, y = ds.train_data

xt, yt = ds.test_data


_, input_size, _, output_size = x.shape + y.shape
batch_size = 40
seq_length = 60
layer_size = 45

c = NetConfig(
    Dt = 1.0,
    SeqLength=seq_length,
    BatchSize=batch_size,
    FeedbackDelay=1,
    OutputTau=5.0,
    YMeanStat = np.zeros((batch_size, seq_length, output_size))
)

net = (
    LayerConfig(
        Size = layer_size,
        TauSoma = 5.0,
        TauSyn = 15.0,
        TauSynFb = 5.0,
        TauMean = 100.0,
        P = 0.1,
        Q = 0.1,
        K = 1.0,
        Omega = 1.0,
        FbFactor = 0.0,
        LearningRate=0.001,
        LateralLearnFactor=100.0,
        Act = RELU,
        W = xavier_init(input_size, layer_size),
        B = np.ones((1, layer_size)),
        L = xavier_init(layer_size, layer_size),
        dW = np.zeros((input_size, layer_size)),
        dB = np.zeros((1, layer_size)),
        dL = np.zeros((layer_size, layer_size)),
        Am = np.zeros((1, layer_size)),
        UStat = np.zeros((batch_size, seq_length, layer_size)),
        AStat = np.zeros((batch_size, seq_length, layer_size)),
        FbStat = np.zeros((batch_size, seq_length, layer_size)),
        SynStat = np.zeros((batch_size, seq_length, input_size)),
    ),
    LayerConfig(
        Size = output_size,
        TauSoma = 5.0,
        TauSyn = 15.0,
        TauSynFb = 5.0,
        TauMean = 100.0,
        P = 0.1,
        Q = 0.1,
        K = 1.0,
        Omega = 1.0,
        FbFactor = 0.0,
        LearningRate=0.0001,
        LateralLearnFactor=100.0,
        Act = RELU,
        W = xavier_init(layer_size, output_size),
        B = np.ones((1, output_size)),
        L = xavier_init(output_size, output_size),
        dW = np.zeros((layer_size, output_size)),
        dB = np.zeros((1, output_size)),
        dL = np.zeros((output_size, output_size)),
        Am = np.zeros((1, output_size)),
        UStat = np.zeros((batch_size, seq_length, output_size)),
        AStat = np.zeros((batch_size, seq_length, output_size)),
        FbStat = np.zeros((batch_size, seq_length, output_size)),
        SynStat = np.zeros((batch_size, seq_length, layer_size)),
    ),
)

l0 = net[0]
l1 = net[1]

sa = []
opt = SGDOpt((1.0,))
opt.init(l0.get("W"), l1.get("W"))

for e in xrange(1):
    trainStats, testStats = run_model(
        1,
        net,
        c,
        x,
        y,
        xt,
        yt,
        test_freq = 10
    )

    # opt.update(-dW)
    # l0.set("W", opt.params[0])

    

st = trainStats
SquaredError = st.get("SquaredError")
ClassificationError = st.get("ClassificationError")
SignAgreement = st.get("SignAgreement")
AverageActivity = st.get("AverageActivity")
Sparsity = st.get("Sparsity")

shl(l0.get("AStat")[0,:])
