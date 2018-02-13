import logging
import sys
import ctypes as ct
import numpy as np
from numpy.ctypeslib import ndpointer
import os

lib_dir = os.path.dirname(os.path.realpath(__file__))


# lib_dir = "/home/alexeyche/prog/supervised-hebbian-learning"



class LayerConfig(ct.Structure):
    _fields_ = [
        ("layer_size", ct.c_uint)
    ]


class Config(ct.Structure):
    _fields_ = [
        ("layer_num", ct.c_uint),
        ("layer_configs", ct.POINTER(LayerConfig))
    ]


class Matrix(ct.Structure):
    _fields_ = [
        ("data", ct.POINTER(ct.c_float)),
        ("nrows", ct.c_uint),
        ("ncols", ct.c_uint)
    ]




_shllib = np.ctypeslib.load_library('libshl', lib_dir)

_shllib.run_model.restype = None
_shllib.run_model.argtypes = [
    Config,
    Matrix
]




def run_model(config, data):
    m = Matrix()
    m.data = (
        data
            .astype(np.float32)
            .ctypes
            .data_as(ct.POINTER(ct.c_float))
    )
    m.nrows = data.shape[0]
    m.ncols = data.shape[1]

    _shllib.run_model(
        config,
        m
    )

