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




_shllib = np.ctypeslib.load_library('libshl', lib_dir)

_shllib.run_model.restype = ct.c_int
_shllib.run_model.argtypes = [
    Config,
    MatrixFlat,
    MatrixFlat
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


    outStat = np.zeros((batch_size, input_size*seq_length), dtype=np.float32)

    outStatFlat = MatrixFlat.from_np(outStat)
    dataFlat = MatrixFlat.from_np(data)

    retcode = _shllib.run_model(
        config,
        dataFlat,
        outStatFlat
    )
    if retcode != 0:
        raise Exception("Bad return code")
    return outStat
