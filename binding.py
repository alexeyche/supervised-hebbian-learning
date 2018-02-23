import logging
import sys
import ctypes as ct
import numpy as np
from numpy.ctypeslib import ndpointer
import os
c_float_p = ct.POINTER(ct.c_float)

# lib_dir = os.path.dirname(os.path.realpath(__file__))


lib_dir = "/home/alexeyche/prog/supervised-hebbian-learning"
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


class StructureWithSizeDescr(ct.Structure):

    class Np(object):
        pass

    @classmethod
    def alloc(cls):
        assert hasattr(cls, "_size_"), "Need to define size"
        assert hasattr(cls, "_fields_"), "Need to define fields"

        o = cls.Np()
        for fname, _ in cls._fields_:
            s = cls._size_[fname]
            if len(s) == 2:
                setattr(o, fname, np.zeros((s[0], s[1]), dtype=np.float32))
            elif len(s) == 3:
                setattr(o, fname, np.zeros((s[0], s[1]*s[2]), dtype=np.float32))
            else:
                raise Exception("Other shapes are not supported")

        return o

    @classmethod
    def reshape(cls, o_np):
        assert hasattr(cls, "_size_"), "Need to define size"
        assert hasattr(cls, "_fields_"), "Need to define fields"

        for fname, _ in cls._fields_:
            m_raw = getattr(o_np, fname)
            orig_size = cls._size_[fname]
            if len(orig_size) == 3:
                setattr(
                    o_np, 
                    fname, 
                    m_raw.reshape((
                        orig_size[0],
                        orig_size[1],
                        orig_size[2],
                    ))
                )
            elif len(orig_size) == 2:
                pass
            else:
                raise Exception("Other shapes are not supported")
        return o_np

    @classmethod
    def from_np(cls, s):
        assert hasattr(cls, "_size_"), "Need to define size"
        assert hasattr(cls, "_fields_"), "Need to define fields"

        r = cls()
        for fname, _ in cls._fields_:
            setattr(r, fname, MatrixFlat.from_np(getattr(s, fname)))
        return r



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



class Config(ct.Structure):
    _fields_ = [
        ("F0", MatrixFlat),
        ("R0", MatrixFlat),
        ("F1", MatrixFlat),
        ("Dt", ct.c_double),
        ("TauSyn", ct.c_double),
        ("TauMean", ct.c_double),
        ("FbFactor", ct.c_double),
        ("LearningRate", ct.c_double),
        ("Lambda", ct.c_double)
    ]


class Data(ct.Structure):
    _fields_ = [
        ("Input", MatrixFlat),
        ("Output", MatrixFlat),
    ]

class State(StructureWithSizeDescr):
    _fields_ = [
        ("A0m", MatrixFlat),
        ("dF0", MatrixFlat),
        ("dF1", MatrixFlat),
    ]

    _struc_info_ = get_structure_info()
    _size_ = {
        "A0m": (_struc_info_.BatchSize, _struc_info_.LayerSize),
        "dF0": (_struc_info_.InputSize, _struc_info_.LayerSize),
        "dF1": (_struc_info_.LayerSize, _struc_info_.OutputSize),
    }


class Statistics(StructureWithSizeDescr):
    _fields_ = [
        ("Input", MatrixFlat),
        ("U", MatrixFlat),
        ("A", MatrixFlat),
        ("dA", MatrixFlat),
        ("Output", MatrixFlat),
        ("De", MatrixFlat),
        ("dF0", MatrixFlat),
    ]

    _struc_info_ = get_structure_info()
    _size_ = {
        "Input": (_struc_info_.BatchSize, _struc_info_.SeqLength, _struc_info_.InputSize),
        "U": (_struc_info_.BatchSize, _struc_info_.SeqLength, _struc_info_.LayerSize),
        "A": (_struc_info_.BatchSize, _struc_info_.SeqLength, _struc_info_.LayerSize),
        "dA": (_struc_info_.BatchSize, _struc_info_.SeqLength, _struc_info_.LayerSize),
        "Output": (_struc_info_.BatchSize, _struc_info_.SeqLength, _struc_info_.OutputSize),
        "De": (_struc_info_.BatchSize, _struc_info_.SeqLength, _struc_info_.OutputSize),
        "dF0": (_struc_info_.SeqLength, _struc_info_.InputSize, _struc_info_.LayerSize),
    }


_shllib.run_model.restype = ct.c_int
_shllib.run_model.argtypes = [
    ct.c_uint,
    Config,
    State,
    State,
    Data,
    Data,
    Statistics,
    Statistics
]



def run_model(epochs, config, train_state, test_state, train_input, train_output, test_input, test_output):
    struc_info = get_structure_info()

    input_size = struc_info.InputSize
    layer_size = struc_info.LayerSize
    output_size = struc_info.OutputSize
    batch_size = struc_info.BatchSize
    layers_num = struc_info.LayersNum
    seq_length = struc_info.SeqLength

    trainStatistics = Statistics.alloc()
    trainStatisticsFlat = Statistics.from_np(trainStatistics)

    testStatistics = Statistics.alloc()
    testStatisticsFlat = Statistics.from_np(testStatistics)
    
    trainInp = Data() 
    trainInp.Input = MatrixFlat.from_np(train_input)
    trainInp.Output = MatrixFlat.from_np(train_output)

    testInp = Data() 
    testInp.Input = MatrixFlat.from_np(test_input)
    testInp.Output = MatrixFlat.from_np(test_output)

    trainStateFlat = State.from_np(train_state)
    testStateFlat = State.from_np(test_state)
    
    retcode = _shllib.run_model(
        epochs,
        config,
        trainStateFlat,
        testStateFlat,
        trainInp,
        testInp,
        trainStatisticsFlat,
        testStatisticsFlat
    )
    if retcode != 0:
        raise Exception("Bad return code")
    return Statistics.reshape(trainStatistics), Statistics.reshape(testStatistics)
