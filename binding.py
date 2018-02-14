import logging
import sys
import ctypes as ct
import numpy as np
from numpy.ctypeslib import ndpointer
import os

# lib_dir = os.path.dirname(os.path.realpath(__file__))


lib_dir = "/home/alexeyche/prog/supervised-hebbian-learning"

c_float_p = ct.POINTER(ct.c_float)



class MatrixFlat(ct.Structure):
    @staticmethod
    def from_np(m):
        o = MatrixFlat()
        assert m.dtype == np.float32, "Need float32 matrix"
        
        if len(m.shape) == 3:
            m = m.reshape((m.shape[0], m.shape[1]*m.shape[2]))
        elif len(m.shape) == 2:
            pass
        else:
            raise Exception("Can't deal with shape more that 3 dimensions")

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
        ("F1", MatrixFlat),
        ("Dt", ct.c_double),
        ("SynTau", ct.c_double),
    ]

class Structure(ct.Structure):
    _fields_ = [
        ("InputSize", ct.c_int),
        ("LayerSize", ct.c_int),
        ("OutputSize", ct.c_int),
        ("BatchSize", ct.c_int),
        ("LayersNum", ct.c_int),
        ("SeqLength", ct.c_int),
    ]



class Stat(ct.Structure):
    _fields_ = [
        ("Input", MatrixFlat),
        ("U", MatrixFlat),
        ("A", MatrixFlat),
        ("Output", MatrixFlat),
    ]

    
    class Np(object):
        pass

    @staticmethod
    def alloc():
        struc_info = get_structure_info()
        size = {
            "Input": (struc_info.BatchSize, struc_info.SeqLength, struc_info.InputSize),
            "U": (struc_info.BatchSize, struc_info.SeqLength, struc_info.LayerSize),
            "A": (struc_info.BatchSize, struc_info.SeqLength, struc_info.LayerSize),
            "Output": (struc_info.BatchSize, struc_info.SeqLength, struc_info.OutputSize),
        }
        o = Stat.Np()
        for fname, _ in Stat._fields_:
            s = size[fname]
            setattr(o, fname, np.zeros(s, dtype=np.float32))
        return o

    @staticmethod
    def from_np(s):
        r = Stat()
        for fname, _ in Stat._fields_:
            setattr(r, fname, MatrixFlat.from_np(getattr(s, fname)))
        return r

_shllib = np.ctypeslib.load_library('libshl', lib_dir)

_shllib.run_model.restype = ct.c_int
_shllib.run_model.argtypes = [
    Config,
    MatrixFlat,
    Stat
]

_shllib.get_structure_info.restype = Structure

def get_structure_info():
    return _shllib.get_structure_info()

def run_model(config, data):
    struc_info = get_structure_info()

    input_size = struc_info.InputSize
    layer_size = struc_info.LayerSize
    output_size = struc_info.OutputSize
    batch_size = struc_info.BatchSize
    layers_num = struc_info.LayersNum
    seq_length = struc_info.SeqLength

    stat = Stat.alloc()
    statFlat = Stat.from_np(stat)

    dataFlat = MatrixFlat.from_np(data)

    retcode = _shllib.run_model(
        config,
        dataFlat,
        statFlat
    )
    if retcode != 0:
        raise Exception("Bad return code")
    return stat
