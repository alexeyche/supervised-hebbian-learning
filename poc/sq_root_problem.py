#!/usr/bin/env python

import numpy as np
from util import *

x = np.asarray([
    [-1.0, -1.0],
    [-1.0, 1.0],
    [1.0, -1.0],
    [1.0, 1.0]
])

y = np.asarray([
    [-1.0],
    [1.0],
    [1.0],
    [-1.0],
])

weight_factor = 0.1

W = weight_factor * np.random.random((2, 2)) - weight_factor / 2.0
Wfb = weight_factor * np.random.random((1, 2)) - weight_factor / 2.0

for _ in xrange(100):
    ff = np.dot(x, W)
    fb = np.dot(y, Wfb)

    e = ff - fb

    W += 0.1 * np.dot(x.T, fb-ff)

    print np.linalg.norm(e)


