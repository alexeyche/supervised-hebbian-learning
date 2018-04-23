
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
# y = np.concatenate((y, y), axis=1) # use it for 1 layer output

xt, yt = ds.test_data

# x, y = x[:1000], y[:1000]
# xt, yt = xt[:1000], yt[:1000]

# yt = np.concatenate((yt, yt), axis=1) # use it for 1 layer output

_, input_size, _, output_size = x.shape + y.shape
batch_size = 40
seq_length = 60
layer_size = 45

c = NetConfig(
    Dt = 1.0,
    SeqLength=seq_length,
    BatchSize=batch_size,
    FeedbackDelay=15,
    OutputTau=5.0,
    DeStat = np.zeros((batch_size, seq_length, output_size)),
    YMeanStat = np.zeros((batch_size, seq_length, output_size))
)

net = (
    LayerConfig(
        Size = layer_size,
        TauSoma = 1.0,
        TauSyn = 15.0,
        TauSynFb = 5.0,
        TauMean = 100.0,
        ApicalGain = 1.0,
        FbFactor = 10.0,
        TauGrad = 1000.0,
        LearningRate=0.005,
        Act = RELU,
        GradProc = ACTIVE_HEBB,
        W = xavier_init(input_size, layer_size),
        B = np.zeros((1, layer_size)),
        dW = np.zeros((input_size, layer_size)),
        dB = np.zeros((1, layer_size)),
        Am = np.zeros((1, layer_size)),
        UStat = np.zeros((batch_size, seq_length, layer_size)),
        AStat = np.zeros((batch_size, seq_length, layer_size)),
        FbStat = np.zeros((batch_size, seq_length, layer_size)),
        SynStat = np.zeros((batch_size, seq_length, input_size)),
    ),
    LayerConfig(
        Size = output_size,
        TauSoma = 1.0,
        TauSyn = 1.0,
        TauSynFb = 15.0,
        TauMean = 0.0,
        ApicalGain = 1.0,
        FbFactor = 0.0,
        TauGrad = 10.0,
        LearningRate=0.002,
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
        SynStat = np.zeros((batch_size, seq_length, layer_size)),
    ),
)

l0 = net[0]
l1 = net[1]

sa = []
opt = AdamOpt((1.0,))
opt.init(l0.get("W"))

for e in xrange(100):
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


    # # x_m = np.mean(np.mean(l0.get("SynStat"),0),0)
    y_m = np.mean(np.mean(l0.get("AStat"), 0), 0)

    dUorig = l0.get("FbStat")

    dUhebb = np.transpose([
        l0.get("AStat")[:, i] * np.sign(y_m - np.mean(y_m))
        for i in xrange(seq_length)
    ], (1,0,2))
    
    # shl(np.mean(np.mean(dUhebb, 1),0), np.mean(np.mean(dUorig, 1), 0))
    sa.append(
        np.mean(
            np.sign(np.mean(np.mean(dUhebb, 1),0)) == 
            np.sign(np.mean(np.mean(dUorig, 1),0))
        )
    )

    dU = dUhebb

    dW = np.mean([
        np.dot(
            l0.get("SynStat")[:,i].T,
            dU[:, i]
        )
        for i in xrange(seq_length)
    ], 0)

    opt.update(-dW)
    l0.set("W", opt.params[0])

    
# shl(dUorig[0,:,1], dUhebb[0,:,1])

# shl(y_m * np.sign(y_m - np.mean(y_m)), np.mean(np.mean(l0.get("FbStat"),0), 0))

# dd = np.zeros((50,))
# for i in xrange(50):
#     de = 
#     if i < 49:
#         den = l0.get("FbStat")[:,i+1].T

#     dd[i] = np.mean(np.equal(np.sign(den), np.sign(de)))


# shl(de[0], den[0], labels=["Orig", "Hebb"])

st = trainStats
SquaredError = st.get("SquaredError")
ClassificationError = st.get("ClassificationError")
SignAgreement = st.get("SignAgreement")
AverageActivity = st.get("AverageActivity")
Sparsity = st.get("Sparsity")

shm(l0.get("AStat")[0,:])
