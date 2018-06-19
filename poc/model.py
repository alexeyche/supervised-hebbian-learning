
import numpy as np

def tanh(x):
    return (1.0 - np.exp(-2*x))/(1.0 + np.exp(-2*x))

def tanh_derivative(x):
    return (1 + tanh(x))*(1 - tanh(x))

def softplus(x):
    return np.log(1.0 + np.exp(x))

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

class Layer(object):
    def __init__(
            s,
            num_iters,
            batch_size,
            input_size,
            layer_size,
            output_size,
            act,
            weight_factor,
            adapt_gain,
            tau_m,
            tau_syn,
            dt
    ):

        s.num_iters = num_iters
        s.layer_size = layer_size
        s.batch_size = batch_size
        s.output_size = output_size
        s.input_size = input_size
        s.dt = dt
        s.act = act
        s.tau_m = tau_m
        s.tau_syn = tau_syn
        s.adapt_gain = adapt_gain

        s.W = weight_factor * np.random.random((s.input_size, s.layer_size)) - weight_factor / 2.0
        s.Wfb = weight_factor * np.random.random((s.output_size, s.layer_size)) - weight_factor / 2.0

        s.uh = np.zeros((s.num_iters, s.batch_size, s.layer_size))
        s.ah = np.zeros((s.num_iters, s.batch_size, s.layer_size))
        s.amh = np.zeros((s.num_iters, 1))
        s.ffh = np.zeros((s.num_iters, s.batch_size, s.layer_size))
        s.fbh = np.zeros((s.num_iters, s.batch_size, s.layer_size))
        s.eh = np.zeros((s.num_iters, s.batch_size, 1))
        s.reset_state()

    def reset_state(s):
        s.xm = np.zeros((s.batch_size, s.input_size))
        s.u = np.zeros((s.batch_size, s.layer_size))
        s.a = np.zeros((s.batch_size, s.layer_size))
        s.am = np.zeros((1,))
        s.dW = np.zeros(s.W.shape)
        s.dWfb = np.zeros(s.Wfb.shape)


    def run(s, t, xt, yt):
        s.xm += xt - s.xm / s.tau_syn
        ff = np.dot(s.xm, s.W)
        fb = np.dot(yt, s.Wfb)
        
        fb = 2.0*(sigmoid(10.0*np.maximum(fb-0.01, 0.0))-0.5)
        
        du = ff + fb
        
        s.u += s.dt * (du - s.u)
        s.a[:] = s.act(s.u - s.am)
        
        s.am[:] += s.adapt_gain * (np.mean(s.a) - s.am)/s.tau_m

        s.dW += np.dot(s.xm.T, s.a - s.am)/s.num_iters
        s.dWfb += np.dot(yt.T, s.a - s.am)/s.num_iters

        s.ah[t, :] = s.a.copy()
        s.amh[t, :] = s.am.copy()
        s.uh[t, :] = s.u.copy()
        s.ffh[t, :] = ff
        s.fbh[t, :] = fb
        s.eh[t, :] = np.linalg.norm(fb - ff, axis=1, keepdims=True)



class Net(object):
    def __init__(s, *layers):
        s.layers = layers

    def run(s, t, xt, yt, feedback_delay=1, test=False):
        for li, l in enumerate(s.layers):
            x_l = s.layers[li - 1].a if li > 0 else xt
            if li < len(s.layers) - 1:
                if t < feedback_delay:
                    y_l = np.zeros(s.layers[li+1].a.shape)
                else:
                    y_l = s.layers[li + 1].ah[t-feedback_delay]

            else:
                y_l = yt

            fb_factor = 1.0
            if test and li == len(s.layers) - 1:
                y_l = np.zeros(yt.shape)
                fb_factor = 0.0

            l.run(t, x_l, y_l * fb_factor)

    @property
    def size(s):
        return len(s.layers)

    def reset_state(s):
        for l in s.layers:
            l.reset_state()

    def __getitem__(s, key):
        return s.layers[key]