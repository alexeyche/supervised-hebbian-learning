
import numpy as np

def tanh(x):
    return (1.0 - np.exp(-2*x))/(1.0 + np.exp(-2*x))

def tanh_derivative(x):
    return (1 + tanh(x))*(1 - tanh(x))

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
            dt
    ):

        s.num_iters = num_iters
        s.layer_size = layer_size
        s.batch_size = batch_size
        s.output_size = output_size
        s.input_size = input_size
        s.dt = dt
        s.act = act

        s.W = weight_factor * np.random.random((s.input_size, s.layer_size)) - weight_factor / 2.0
        s.Wfb = weight_factor * np.random.random((s.output_size, s.layer_size)) - weight_factor / 2.0

        s.uh = np.zeros((s.num_iters, s.batch_size, s.layer_size))
        s.ah = np.zeros((s.num_iters, s.batch_size, s.layer_size))
        s.ffh = np.zeros((s.num_iters, s.batch_size, s.layer_size))
        s.fbh = np.zeros((s.num_iters, s.batch_size, s.layer_size))
        s.eh = np.zeros((s.num_iters, s.batch_size, 1))
        s.reset_state()

    def reset_state(s):
        s.u = np.zeros((s.batch_size, s.layer_size))
        s.a = np.zeros((s.batch_size, s.layer_size))
        s.dW = np.zeros(s.W.shape)
        s.dWfb = np.zeros(s.Wfb.shape)


    def run(s, t, xt, yt, no_feedback=False):
        ff = np.dot(xt, s.W)
        fb = np.dot(yt, s.Wfb)

        if no_feedback:
            du = ff
        else:
            du = (ff + fb)/2.0

        # du = ff + fb * (fb - ff)

        s.u += s.dt * (du - s.u)
        s.a[:] = s.act(s.u)

        s.dW += np.dot(xt.T, fb - ff)/s.num_iters
        s.dWfb += np.dot(yt.T, fb - ff) / s.num_iters

        s.ah[t, :] = s.a.copy()
        s.uh[t, :] = s.u.copy()
        s.ffh[t, :] = ff
        s.fbh[t, :] = fb
        s.eh[t, :] = np.linalg.norm(fb - ff, axis=1, keepdims=True)



class Net(object):
    def __init__(s, *layers):
        s.layers = layers

    def run(s, t, xt, yt, test=False):
        for li, l in enumerate(s.layers):
            x_l = s.layers[li - 1].a if li > 0 else xt
            y_l = s.layers[li + 1].a if li < len(s.layers) - 1 else yt

            no_feedback = False
            if test and li == len(s.layers) - 1:
                y_l = np.zeros(yt.shape)
                no_feedback = True

            l.run(t, x_l, y_l, no_feedback)

    @property
    def size(s):
        return len(s.layers)

    def __getitem__(s, key):
        return s.layers[key]