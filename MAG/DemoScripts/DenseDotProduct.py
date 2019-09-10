"""
Test the dot product against cython
Require to be on Discretize/Cython_dotProduct branch
"""

from discretize.utils import prodAvec, prodAtvec
import numpy as np
import scipy as sp
import time

m, n = 100000, 10000

A = np.ones((m,n), dtype=np.float32)
x = sp.randn(m)

@profile
def npdotProduct(A, x):
    # Create a dense matrix
    y = np.zeros(A.shape[1]).astype(np.float64)
    vec = np.zeros(A.shape[1]).astype(np.float64)
    for ii in range(4):
        strt = time.time()
        y = np.dot(A.T,x.astype(np.float64),vec)
        print("Numpy.dot: " + str(time.time()-strt) + " s")

    return y

y_numpy = npdotProduct(A, x)

# @profile
# def cythondotProduct(A, x):
#     # Create a dense matrix
#     y = np.zeros(A.shape[0])
#     for ii in range(4):


#         strt = time.time()
#         y += prodAvec(A, x)
#         print("Cython.dot: " + str(time.time()-strt) + " s")

#     return y
# y_cython = cythondotProduct(A, x)

# print("Residual:" + str(np.linalg.norm(y_numpy-y_cython)))
