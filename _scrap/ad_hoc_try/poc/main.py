
import sys
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
from util import *
import numpy as np

from poc.common import *
from poc.opt import *
from datasets import *

# np.random.seed(10)


gradient_postproc = hebb_postproc
# ds = XorDataset()
ds = ToyDataset()
# ds = MNISTDataset()
x_shape, y_shape = ds.train_shape

batch_size, input_size = x_shape
batch_size, output_size = y_shape

xt_shape, yt_shape = ds.test_shape
test_batch_size = xt_shape[0]


net = build_network(input_size, (100, output_size), (100.0, 0.0))


# opt = SGDOpt((1e-05,) * 4)
# opt = MomentumOpt((1e-07,) * 4, 0.9)
opt = AdamOpt((1e-04,) * 4, 0.9)
# opt = AdagradOpt((1e-02,)*4)
# opt = AdadeltaOpt((1e-04,)*4, 0.9)

opt.init(*[t for l in net for t in (l.W, l.b)])

at_stat = [np.zeros((test_batch_size, l.layer_size)) for l in net]
ut_stat = [np.zeros((test_batch_size, l.layer_size)) for l in net]
dE_stat = [np.zeros((batch_size, l.layer_size)) for l in net]

epochs = 15000

stat_h = np.zeros((epochs, len(net)-1, 1))

for epoch in xrange(epochs):
    kwargs = {}
    # if epoch == 999:
    #     kwargs["plot"] = True

    derivatives, st, m  = run_feedforward(
        net, 
        ds, 
        is_train_phase=True, 
        gradient_postproc=gradient_postproc,
        **kwargs
    )

    _, stt, mt = run_feedforward(
        net, 
        ds, 
        is_train_phase=False, 
        gradient_postproc=None
    )

    a_stat, u_stat, stat, dE_stat = st
    at_stat, ut_stat, statt, dE_stat_t = stt

    stat_h[epoch] = stat.copy()

    # [tt for l in net for tt in (l.W, l.b)]
    
    opt.update(*[-t for pair in derivatives for t in pair])

    # for l, (dW, db) in zip(net, derivatives):
    #     l.W += lrate * dW
    #     l.b += lrate * db


    se, ll, er = m
    tse, tll, ter = mt

    if epoch % 100 == 0:
        print "{}: train ll {:.4f}, er {:.4f}; test ll {:.4f}, er {:.4f}".format(epoch, ll, er, tll, ter)

# shl(dE_actual[1], dE[1], a_stat[0][1], labels=["dE_actual", "dE", "A"])
# shl(a_stat[1])