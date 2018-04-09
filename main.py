
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

def positive_random_norm(fan_in, fan_out, p):
    m = np.random.random((fan_in, fan_out))
    m = m/(np.sum(m, 0)/p)
    return m

def bound_weights(l):
    L = l.get("L")
    W = l.get("W")
    B = l.get("B")
    np.fill_diagonal(L, 0.0)
    l.set("W", np.minimum(np.maximum(W, 0.0), l.Omega))
    l.set("L", np.maximum(L, 0.0))
    l.set("B", np.maximum(B, 0.0))


ds = ToyDataset()

x, y = ds.train_data

xt, yt = ds.test_data


y *= 0.1
yt *= 0.1

_, input_size, _, output_size = x.shape + y.shape

batch_size = 40
seq_length = 50

layer_size = 100
p = 0.1
q = 0.1

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
        Omega = 0.5,
        FbFactor = 1.0,
        LearningRate=0.001,
        LateralLearnFactor=100.0,
        Act = RELU,
        W = positive_random_norm(input_size, layer_size, p),
        B = np.ones((1, layer_size)),
        L = np.zeros((layer_size, layer_size)),
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
        Omega = 0.5,
        FbFactor = 1.0,
        LearningRate=0.0001,
        LateralLearnFactor=100.0,
        Act = RELU,
        W = positive_random_norm(layer_size, output_size, p),
        B = np.ones((1, output_size)),
        L = np.zeros((output_size, output_size)),
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
opt = SGDOpt((
    0.001, 0.001, 0.01, 
    0.001, 0.001, 0.01,
))

opt.init(
    l0.get("W"), l0.get("B"), l0.get("L"), 
    l1.get("W"), l1.get("B"), l1.get("L"), 
)


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

    l0.get("AStat")

    opt.update(
        -l0.get("dW"), -l0.get("dB"), -l0.get("dL"), 
        -l1.get("dW"), -l1.get("dB"), -l1.get("dL"), 
    )

    l0.set("W", opt.params[0])
    l0.set("B", opt.params[1])
    l0.set("L", opt.params[2])
    l1.set("W", opt.params[3])
    l1.set("B", opt.params[4])
    l1.set("L", opt.params[5])
    
    bound_weights(l0)
    bound_weights(l1)

    opt.params = [
        l0.get("W"), l0.get("B"), l0.get("L"), 
        l1.get("W"), l1.get("B"), l1.get("L"), 
    ]

    

st = trainStats
SquaredError = st.get("SquaredError")
ClassificationError = st.get("ClassificationError")
SignAgreement = st.get("SignAgreement")
AverageActivity = st.get("AverageActivity")
Sparsity = st.get("Sparsity")

shl(l0.get("AStat")[0,:])


## need metrics
## need good data to test