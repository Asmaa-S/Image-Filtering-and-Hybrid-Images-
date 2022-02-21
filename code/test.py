from helpers import conv_forward

import numpy as np

np.random.seed(7)

X = np.random.randn(10,4,4,3)
W = np.random.randn(2,2,3,8)
b = np.random.randn(1,1,1,8)
pad = 2
stride = (1,1)

Z = conv_forward(X, W, b, stride)

print("Z's mean =", np.mean(Z))
#print("Z[3,2,1] =", Z[3,2,1])
