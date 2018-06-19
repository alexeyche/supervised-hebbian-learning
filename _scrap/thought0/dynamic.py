

import numpy as np
from util import *

x = 1.0
y = 0.1

dt = 0.1


W2 = 1.0

def system(u0, u1, W0, W1):
    ff0 = np.dot(x, W0)
    fb0 = np.dot(u1, W1)

    du0 = (ff0 + fb0)/2.0 - u0

    ff1 = np.dot(u0, W1)
    fb1 = np.dot(y, W2)

    du1 = (ff1 + fb1)/2.0 - u1

    dW0 = x * (fb0 - ff0)
    dW1 = u0 * (fb1 - ff1)
    return du0, du1, dW0, dW1, ff0, fb0, ff1, fb1


u0, u1 = 0.0, 0.0
W0 = 0.0
W1 = 1.0

num_iters = 1000

u0h, u1h, W0h, W1h = np.zeros(num_iters), np.zeros(num_iters), np.zeros(num_iters), np.zeros(num_iters)
ff0h, fb0h, ff1h, fb1h = np.zeros(num_iters), np.zeros(num_iters), np.zeros(num_iters), np.zeros(num_iters)

for ti in xrange(num_iters):
    du0, du1, dW0, dW1, ff0, fb0, ff1, fb1 = system(u0, u1, W0, W1)

    u0 += dt * du0
    u1 += dt * du1
    W0 += dt * dW0
    W1 += dt * dW1

    u0h[ti], u1h[ti], W0h[ti], W1h[ti] = u0, u1, W0, W1
    ff0h[ti], fb0h[ti], ff1h[ti], fb1h[ti] = ff0, fb0, ff1, fb1


shl(u0h, u1h, labels=("u0", "u1"), show=False)
shl(W0h, W1h, labels=("W0", "W1"), show=False)
shl(ff0h, fb0h, np.square(fb0h - ff0h), labels=("ff0", "fb0", "D"), show=False)
shl(ff1h, fb1h, np.square(fb1h - ff1h), labels=("ff1", "fb1", "D"), show=False)
plt.show()


