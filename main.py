
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

def positive_random_norm(fan_in, fan_out, p):
    m = np.random.random((fan_in, fan_out))
    m = m/(np.sum(m, 0)/p)
    return m

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
    Dt = 0.5,
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
        LearningRate=0.0001,
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
        Omega = 1.0,
        FbFactor = 1.0,
        LearningRate=0.0001,
        LateralLearnFactor=10.0,
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

trainStats, testStats = run_model(
    1000,
    net,
    c,
    x,
    y,
    xt,
    yt,
    test_freq = 100
)
    

st = testStats
SquaredError = st.get("SquaredError")
ClassificationError = st.get("ClassificationError")
SignAgreement = st.get("SignAgreement")
AverageActivity = st.get("AverageActivity")
Sparsity = st.get("Sparsity")

shl(l0.get("AStat")[0,:])


## need metrics
## need good data to test

## http://scikit-learn.org/stable/auto_examples/cluster/plot_dict_face_patches.html#sphx-glr-auto-examples-cluster-plot-dict-face-patches-py
