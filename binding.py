import logging
import sys
import ctypes as ct
import numpy as np
from numpy.ctypeslib import ndpointer
import os
c_float_p = ct.POINTER(ct.c_float)

# lib_dir = os.path.dirname(os.path.realpath(__file__))


lib_dir = "/Users/aleksei/distr/supervised-hebbian-learning"
# lib_dir = "/home/alexeyche/prog/supervised-hebbian-learning"
_shllib = np.ctypeslib.load_library('libshl', lib_dir)


class Structure(ct.Structure):
    _fields_ = [
        ("InputSize", ct.c_int),
        ("LayerSize", ct.c_int),
        ("OutputSize", ct.c_int),
        ("BatchSize", ct.c_int),
        ("LayersNum", ct.c_int),
        ("SeqLength", ct.c_int),
    ]


def get_structure_info():
    return _shllib.get_structure_info()

_shllib.get_structure_info.restype = Structure


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
        o.Data = m.ctypes.data_as(c_float_p)
        o.NRows = m.shape[0]
        o.NCols = m.shape[1]
        return o


    _fields_ = [
        ("Data", c_float_p),
        ("NRows", ct.c_uint),
        ("NCols", ct.c_uint)
    ]




struc_info = get_structure_info()

input_size = struc_info.InputSize
layer_size = struc_info.LayerSize
output_size = struc_info.OutputSize
batch_size = struc_info.BatchSize
layers_num = struc_info.LayersNum
seq_length = struc_info.SeqLength

class ComplexStructure(ct.Structure):
    def __init__(self, **kwargs):
        self._fields_dict = dict(self._fields_)
        self._np_values_dict = {}

        for k, v in kwargs.iteritems():
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
                    raise Exception("Other shapes are not supported")

                setattr(self, k, MatrixFlat.from_np(self._np_values_dict[k][0]))
            else:
                setattr(self, k, v)



    def get_matrix(self, name):
        assert name in self._np_values_dict
        m_raw, orig_shape = self._np_values_dict[name]
        return reshape_from_flat(m_raw, orig_shape)

class Config(ComplexStructure):
    _fields_ = [
        ("Dt", ct.c_double),
        ("LearningRate", ct.c_double),
        ("FeedbackDelay", ct.c_uint)
    ]

class Data(ct.Structure):
    _fields_ = [
        ("X", MatrixFlat),
        ("Y", MatrixFlat),
    ]

class LayerState(ComplexStructure):
    _fields_ = [
        ("TauSoma", ct.c_double),
        ("TauMean", ct.c_double),
        ("ApicalGain", ct.c_double),
        ("Act", ct.c_int),
        ("F", MatrixFlat),
        ("UStat", MatrixFlat),
        ("AStat", MatrixFlat),
    ]



_shllib.run_model.restype = ct.c_int
_shllib.run_model.argtypes = [
    ct.c_uint,
    ct.POINTER(LayerState),
    Config,
    Data,
    Data
]

RELU, SIGMOID = 0, 1

def run_model(epochs, layers, config, train_input, train_output, test_input, test_output):
    assert len(layers) == layers_num, "Expecting {} number of layers".format(layers_num)
    layers_s = (LayerState * len(layers))()
    for li, l in enumerate(layers):
        layers_s[li] = l

    testInp = Data() 
    testInp.X = MatrixFlat.from_np(test_input)
    testInp.Y = MatrixFlat.from_np(test_output)

    trainInp = Data() 
    trainInp.X = MatrixFlat.from_np(train_input)
    trainInp.Y = MatrixFlat.from_np(train_output)

    retcode = _shllib.run_model(
        epochs,
        layers_s,
        config,
        trainInp,
        testInp
    )
    if retcode != 0:
        raise Exception("Bad return code")
    
