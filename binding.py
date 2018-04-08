
import sys
import ctypes as ct
import numpy as np
from numpy.ctypeslib import ndpointer
import os

lib_dir = os.path.dirname(os.path.realpath(__file__))
_shllib = np.ctypeslib.load_library('libshl', lib_dir)



def reshape_from_flat(m_raw, orig_size):
    if len(orig_size) == 3:
        return m_raw.reshape((
            orig_size[0],
            orig_size[1],
            orig_size[2],
        ))
    elif len(orig_size) == 2:
        return m_raw
    else:
        raise Exception("Other shapes are not supported")



class MatrixFlat(ct.Structure):
    @staticmethod
    def from_np(m):
        assert m.dtype == np.float32, "Need float32 matrix"        
        assert len(m.shape) == 2, "Need shape 2 for matrix"

        o = MatrixFlat()
        o.Data = m.ctypes.data_as(ct.POINTER(ct.c_float))
        o.NRows = m.shape[0]
        o.NCols = m.shape[1]
        return o


    _fields_ = [
        ("Data", ct.POINTER(ct.c_float)),
        ("NRows", ct.c_uint),
        ("NCols", ct.c_uint)
    ]



class ComplexStructure(ct.Structure):
    def __init__(self, **kwargs):
        self._fields_dict = dict(self._fields_)
        self._np_values_dict = {}

        for k, v in kwargs.iteritems():
            self.set(k, v)  

    def get(self, k):
        assert k in self._fields_dict, "Unknown key value: {}".format(k)
        if self._fields_dict[k] == MatrixFlat:
            m_raw, orig_shape = self._np_values_dict[k]
            return reshape_from_flat(m_raw, orig_shape)
        else:
            return getattr(self, k)

    def set(self, k, v):
        assert k in self._fields_dict, "Failed to find {} in fields of {}".format(k, self)
        if self._fields_dict[k] == MatrixFlat:
            assert isinstance(v, np.ndarray), \
                "Expecting numpy matrix as input for field {}".format(k)
            
            s = v.shape
            if len(s) == 2:
                self._np_values_dict[k] = (
                    v
                        .copy()
                        .astype(np.float32),
                    v.shape
                )
            elif len(s) == 3:
                self._np_values_dict[k] = (
                    v
                        .copy()
                        .reshape(s[0], s[1]*s[2])
                        .astype(np.float32),
                    v.shape
                )
            else:
                raise Exception("Other shapes ({}) are not supported".format(s))

            setattr(self, k, MatrixFlat.from_np(self._np_values_dict[k][0]))
                    
        else:
            setattr(self, k, v)



class Data(ComplexStructure):
    _fields_ = [
        ("X", MatrixFlat),
        ("Y", MatrixFlat),
        ("BatchSize", ct.c_uint),
    ]

class NetConfig(ComplexStructure):
    _fields_ = [
        ("Dt", ct.c_double),
        ("SeqLength", ct.c_uint),
        ("BatchSize", ct.c_uint),
        ("FeedbackDelay", ct.c_uint),
        ("OutputTau", ct.c_double),
        ("YMeanStat", MatrixFlat),
    ]

class LayerConfig(ComplexStructure):
    _fields_ = [
        ("Size", ct.c_uint),
        ("FbFactor", ct.c_double),
        ("TauSoma", ct.c_double),
        ("TauSyn", ct.c_double),
        ("TauSynFb", ct.c_double),
        ("TauMean", ct.c_double),
        ("P", ct.c_double),
        ("Q", ct.c_double),
        ("K", ct.c_double),
        ("Omega", ct.c_double),
        ("LearningRate", ct.c_double),
        ("LateralLearnFactor", ct.c_double),
        ("TauGrad", ct.c_double),
        ("Act", ct.c_int),
        ("W", MatrixFlat),
        ("B", MatrixFlat),
        ("L", MatrixFlat),
        ("dW", MatrixFlat),
        ("dB", MatrixFlat),
        ("dL", MatrixFlat),
        ("Am", MatrixFlat),
        ("UStat", MatrixFlat),
        ("AStat", MatrixFlat),
        ("FbStat", MatrixFlat),
        ("SynStat", MatrixFlat),
    ]

class Stats(ComplexStructure):
    _fields_ = [
        ("SquaredError", MatrixFlat),
        ("ClassificationError", MatrixFlat),
        ("SignAgreement", MatrixFlat),
        ("AverageActivity", MatrixFlat),
        ("Sparsity", MatrixFlat),
    ]

    @staticmethod
    def construct(epochs):
        return Stats(
            SquaredError = np.zeros((1, epochs)),
            ClassificationError = np.zeros((1, epochs)),
            SignAgreement = np.zeros((1, epochs)),
            AverageActivity = np.zeros((1, epochs)),
            Sparsity = np.zeros((1, epochs)),
        )

_shllib.run_model.restype = ct.c_int
_shllib.run_model.argtypes = [
    ct.c_uint,
    ct.POINTER(LayerConfig),
    ct.c_uint,
    NetConfig,
    Data,
    Data,
    Stats,
    Stats,
    ct.c_uint
]

RELU, SIGMOID = 0, 1
NO_GRADIENT_PROCESSING, LOCAL_LTD, NONLINEAR, HEBB, ACTIVE_HEBB = 0, 1, 2, 3, 4

def run_model(epochs, layers, config, train_input, train_output, test_input, test_output, test_freq=1):
    layers_s = (LayerConfig * len(layers))()
    for li, l in enumerate(layers):
        layers_s[li] = l

    trainInp = Data(X=train_input, Y=train_output, BatchSize=config.BatchSize) 
    testInp = Data(X=test_input, Y=test_output, BatchSize=config.BatchSize) 

    trainStats = Stats.construct(epochs)
    testStats = Stats.construct(epochs)

    retcode = _shllib.run_model(
        epochs,
        layers_s,
        len(layers),
        config,
        trainInp,
        testInp,
        trainStats,
        testStats,
        test_freq
    )
    if retcode != 0:
        raise Exception("Error, see message above")
    return trainStats, testStats
