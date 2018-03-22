
import numpy as np
import ctypes as ct
import time
from binding import *
from util import *
from datasets import XorDataset, to_sparse_ts
from sklearn.metrics import log_loss

np.random.seed(12)

def xavier_init(fan_in, fan_out, const=0.5):
    low = -const * np.sqrt(6.0 / (fan_in + fan_out))
    high = const * np.sqrt(6.0 / (fan_in + fan_out))
    return (low + np.random.random((fan_in, fan_out)) * (high - low)).astype(np.float32)


# ds = XorDataset()
# x, y = ds._x_v, ds._y_v
# y = np.concatenate((y, y), axis=1)

# xt, yt = ds._x_v, ds._y_v
# yt = np.concatenate((yt, yt), axis=1)


x = np.asarray([
    [0.0, 1.0],
])

y = np.asarray([
    [1.0, 0.0],
])

xt, yt = x, y

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
        W = np.ones((input_size, layer_size)), # xavier_init(input_size, layer_size),
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
        W = np.ones((layer_size, output_size)), #xavier_init(layer_size, output_size),
        B = np.zeros((1, output_size)),
        dW = np.zeros((layer_size, output_size)),
        dB = np.zeros((1, output_size)),
        UStat = np.zeros((batch_size, seq_length, output_size)),
        AStat = np.zeros((batch_size, seq_length, output_size)),
        FbStat = np.zeros((batch_size, seq_length, output_size)),
    ),
)

l0, l1 = net



for epoch in xrange(1):
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

        l.set("W", W + 0.5 * dW)
        l.set("B", B + 0.5 * dB)


    deStat = c.get("DeStat")

    UStat0 = l0.get("UStat")
    AStat0 = l0.get("AStat")

    UStat1 = l1.get("UStat")
    AStat1 = l1.get("AStat")

    

    print "{}: train error {:.4f}, ll {:.4f}".format(
        epoch, 
        np.sum(np.square(deStat)), 
        log_loss(y, AStat1[:, 10 + c.FeedbackDelay, :])
    )


W0, dW0 = l0.get("W"), l0.get("dW")
B0, dB0 = l0.get("B"), l0.get("dB")

W1, dW1 = l1.get("W"), l1.get("dW")
B1, dB1 = l1.get("B"), l1.get("dB")

fbStat0 = l0.get("FbStat")
fbStat1 = l1.get("FbStat")

# shl(AStat0[0,:,0], AStat0[0,:,1], AStat0[0,:,2], labels=["0", "1", "2"], title="a 0", show=False)
# shl(fbStat0[0,:,0], fbStat0[0,:,1], fbStat0[0,:,2], labels=["0", "1", "2"], title="fb 0", show=False)
# shl(AStat1[0,:,0], AStat1[0,:,1], labels=["0", "1"], title="a 1", show=False)
# shl(fbStat1[0,:,0], fbStat1[0,:,1], labels=["0", "1"], title="fb 1", show=False)

# plt.show()



