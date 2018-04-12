
import numpy as np
from sklearn.metrics import log_loss
from datasets import one_hot_encode
from util import *

class Act(object):
    def __call__(self, x):
        raise NotImplementedError()

    def deriv(self, x):
        raise NotImplementedError()

class Linear(Act):
    def __call__(self, x):
        return x

    def deriv(self, x):
        if hasattr(x, "shape"):
            return np.ones(x.shape)
        return 1.0

class Sigmoid(Act):
    def __call__(self, x):
        return 1.0/(1.0 + np.exp(-x))

    def deriv(self, x):
        v = self(x)
        return v * (1.0 - v)


class Relu(Act):
    def __call__(self, x):
        return np.maximum(x, 0.0)
        
    def deriv(self, x):
        if isinstance(x, float):
            return 1.0 if x > 0.0 else 0.0
        dadx = np.zeros(x.shape)
        dadx[np.where(x > 0.0)] = 1.0
        return dadx

class ReluBound(Act):
    def __call__(self, x):
        return np.minimum(np.maximum(x, 0.0), 5.0)
        
    def deriv(self, x):
        if isinstance(x, float):
            return 1.0 if x > 0.0 else 0.0
        dadx = np.zeros(x.shape)
        dadx[np.where(x > 0.0)] = 1.0
        return dadx

softplus = lambda x: np.log(1.0 + np.exp(x))
sigmoid = Sigmoid()

class Layer(object):
    def __init__(self, input_size, layer_size, feedback_size, act, tau_m):
        self.input_size = input_size
        self.layer_size = layer_size
        self.feedback_size = feedback_size
        self.act = act
        self.tau_m = tau_m

        self.W = 0.1 - 0.2*np.random.random((input_size, layer_size))
        self.b = np.zeros((layer_size,))
        self.am = np.zeros((layer_size,))

    def run_feedforward(s, I):
        u = np.dot(I, s.W) + s.b
        if s.tau_m > 0.01:
            a = s.act(u - s.am)
            s.am += (np.mean(a, 0) - s.am)/s.tau_m
        else:
            a = s.act(u)
        return u, a

    def __repr__(self):
        return "Layer({};{};{})".format(self.input_size, self.layer_size, self.feedback_size)


    
def build_network(input_size, net_struct, tau_mean_arr):
    return tuple([
        Layer(*params) 
        for params in 
            zip(
                (input_size,) + net_struct[:-1], 
                net_struct, 
                net_struct[1:] + (None,),
                (ReluBound(),)*(len(net_struct)-1) + (Sigmoid(),),
                tau_mean_arr
            )
    ])



def no_gradient_postproc(dE, a):
    return dE

def ltd(dE, a):
    silent_ap = np.where(dE <= 0.0)
    ltd = np.zeros(a.shape)
    ltd[silent_ap] = a[silent_ap]
    return ltd 

def positive_postproc(dE, a):
    return np.maximum(dE, 0.0) - ltd(dE, a)

def nonlinear_postproc(dE, a):
    return \
        1.0*(sigmoid(100.0*np.maximum(dE, 0.0)) - 0.5) - ltd(dE, a)

mean_factor = 3.0

def hebb_postproc(dE, a):
    dE = 10.0*(sigmoid(100.0*np.maximum(dE, 0.0)) - 0.5) + a
    # shl(dE[0], (dE * np.sign(dE - 0.3))[0])
    return dE * np.sign(dE - mean_factor*np.mean(a))

def update_derivatives(
    derivatives, 
    net, 
    u_stat, 
    a_stat, 
    stat,
    x_target,
    y_target, 
    gradient_postproc,
    **kwargs
):
    dE_stat = [None]*len(net)

    y = a_stat[-1]
    dE = y_target - y
    
    derivatives[-1][0] += np.dot(a_stat[-2].T, dE)
    derivatives[-1][1] += np.mean(dE, 0)

    dE_stat[-1] = dE.copy()
    
    for l_idx, (l, l_next, u, a, I) in reversed(
        tuple(
            enumerate(
                zip(net[:-1], net[1:], u_stat[:-1], a_stat[:-1], (x_target,) + tuple(a_stat[:-1]))
            )
        )
    ):
        dE_actual = np.dot(dE, l_next.W.T) * l.act.deriv(u)
        dE = gradient_postproc(dE_actual, a)

        if kwargs.get("plot"):

            shl(dE_actual[0]*10.0, dE[0], a[0], np.asarray([mean_factor*np.mean(a)]*a.shape[1]), labels=["dE actual", "dE", "a", "am"])
        
        stat[l_idx,0] += np.mean(np.sign(dE_actual) == np.sign(dE))
        
        derivatives[l_idx][0] += np.dot(I.T, dE)
        derivatives[l_idx][1] += np.mean(dE, 0)

        dE_stat[l_idx] = dE.copy()

    return derivatives, dE_stat, stat


def run_feedforward(net, ds, is_train_phase, gradient_postproc, **kwargs):
    x_shape, y_shape = ds.train_shape if is_train_phase else ds.test_shape
    batches_num = ds.train_batches_num if is_train_phase else ds.test_batches_num

    stat = np.zeros(((len(net)-1), 1))
 
    batch_size = ds.train_batch_size if is_train_phase else ds.test_batch_size
    
    _, input_size = x_shape
    _, output_size = y_shape

    a_stat = [np.zeros((batch_size, l.layer_size)) for l in net]
    u_stat = [np.zeros((batch_size, l.layer_size)) for l in net]
    dE_stat = None

    derivatives = [[np.zeros(l.W.shape), np.zeros(l.b.shape)] for l in net]

    se, ll, er = 0.0, 0.0, 0.0

    for bi in xrange(batches_num):
        x_target, y_target = ds.next_train_batch() if is_train_phase else ds.next_test_batch()

        I = x_target
        for l_idx, l in enumerate(net):
            u, a = l.run_feedforward(I)

            a_stat[l_idx] = a.copy()
            u_stat[l_idx] = u.copy()

            I = a
        
        se += np.sum(np.square(y_target - a_stat[-1])) / batches_num / batch_size
        ll += log_loss(y_target, a_stat[-1])
        er += np.mean(
            np.not_equal(
                one_hot_encode(np.argmax(a_stat[-1], axis=1), a_stat[-1].shape[1]), 
                y_target
            )
        )

        if is_train_phase:
            _, dE_stat, stat = update_derivatives(
                derivatives,
                net, 
                u_stat, 
                a_stat,
                stat, 
                x_target, 
                y_target, 
                gradient_postproc,
                **kwargs
            )

        
    return derivatives, (a_stat, u_stat, stat, dE_stat), (se, ll, er)



def positive_random_norm(fan_in, fan_out, p):
    m = np.random.random((fan_in, fan_out))
    m = m/(np.sum(m, 0)/p)
    return m

def xavier_init(fan_in, fan_out, const=1.0):
    low = -const * np.sqrt(6.0 / (fan_in + fan_out))
    high = const * np.sqrt(6.0 / (fan_in + fan_out))
    return (low + np.random.random((fan_in, fan_out)) * (high - low)).astype(np.float32)


