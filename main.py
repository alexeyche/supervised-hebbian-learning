
import numpy as np
import ctypes as ct
import time
from binding import *
from util import *
from datasets import XorDataset, to_sparse_ts

np.random.seed(12)

def xavier_init(fan_in, fan_out, const=0.5):
    low = -const * np.sqrt(6.0 / (fan_in + fan_out))
    high = const * np.sqrt(6.0 / (fan_in + fan_out))
    return (low + np.random.random((fan_in, fan_out)) * (high - low)).astype(np.float32)


ds = XorDataset()
x, y = ds._x_v, ds._y_v
y = np.concatenate((y, y), axis=1)

xt, yt = ds._x_v, ds._y_v
yt = np.concatenate((yt, yt), axis=1)

c = NetConfig(
    Dt = 0.5,
    LearningRate=1.0,
    FeedbackDelay=1,
    DeStat = np.zeros((batch_size, seq_length, output_size))
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
    ),
    LayerConfig(
        TauSoma = 1.0,
        TauSyn = 5.0,
        TauMean = 100.0,
        ApicalGain = 1.0,
        FbFactor = 0.0,
        Act = SIGMOID,
        W = xavier_init(layer_size, output_size),
        B = np.zeros((1, output_size)),
        dW = np.zeros((layer_size, output_size)),
        dB = np.zeros((1, output_size)),
        UStat = np.zeros((batch_size, seq_length, output_size)),
        AStat = np.zeros((batch_size, seq_length, output_size)),
    ),
)

l0, l1 = net

for epoch in xrange(100):
    run_model(
        1,
        net,
        c,
        x,
        y,
        xt,
        yt
    )

    for l in net:
        W, dW = l.get("W"), l.get("dW")
        B, dB = l.get("B"), l.get("dB")

        l.set("W", W + 0.1 * dW)
        l.set("B", B + 0.1 * dB)


    deStat = c.get("DeStat")

    UStat0 = l0.get("UStat")
    AStat0 = l0.get("AStat")

    UStat1 = l1.get("UStat")
    AStat1 = l1.get("AStat")

    print "{}: train error {:.4f}".format(epoch, np.sum(np.square(deStat)))
