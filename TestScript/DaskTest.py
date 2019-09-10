import numpy as np
import SimPEG.PF as PF
import dask as dd
import scipy.constants as constants
from SimPEG import Problem
from SimPEG import Utils
from SimPEG import Props
from SimPEG.Utils import mkvc
import scipy.sparse as sp
import os
import dask.array as da
from dask.distributed import Client
from sparse import COO
import dask
import multiprocessing
import time
from multiprocessing.pool import ThreadPool

dask.config.set(scheduler='threads')
dask.config.set(pool=ThreadPool(8))


# if __name__ == '__main__':
    # client = Client(n_workers=4,)
    # client

# Create random problem class
# Create random problem class

N = 10000

class forward(object):

    A = None

    @property
    def A(self):

        if getattr(self, '_A', None) is None:

            self._A = da.random.random_sample(size=(N,N), chunks=[int(N/2),int(N/2)])

        return self._A

    def fields(self, model):
        print("fields")
        return da.dot(self.A, model)

    def Jvec(self, v):

        vec = da.dot(self.A, v)
        row = dask.delayed(
            sp.csr_matrix.dot)(
                sp.eye(N), vec
            )

        return da.from_delayed(row, dtype=float, shape=[N])

    def Jtvec(self, v):
        row = dask.delayed(
            sp.csr_matrix.dot)(
                sp.eye(N), v
            )

        row = da.from_delayed(row, dtype=float, shape=[N])
        vec = da.dot(self.A, row)


        return vec #da.from_delayed(row, dtype=float, shape=[N])

    def JtJvec(self, v):

        vec = da.dot(self.A, v)

        row = dask.delayed(
            sp.csr_matrix.dot)(
                sp.eye(N), vec
            )

        jvec = da.from_delayed(row, dtype=float, shape=([N]))

        jtjvec = da.dot(self.A, jvec)

        return jtjvec #da.from_delayed(jtjvec, dtype=float, shape=(1, N))


    def deriv(self, m):

#         f = self.fields(m)

        row = self.Jvec(np.ones(N))

        return da.from_delayed(row, dtype=float, shape=(1, N))

@dask.delayed
def rowSum(arr):
    sumIt = 0
    for i in range(len(arr)):
        sumIt += arr[i]
    return sumIt


# Create a list of forward
F = [forward() for ii in range(10)]

fields = np.zeros(N)
tc = time.time()
for f in F:

    fields += f.Jvec(f.Jtvec(np.ones(N)).compute()).compute()

print("Run 1 in: ", time.time()-tc)



# tc = time.time()
# fields = []
# out = np.ones(N)
# for f in F:

#     row = dask.delayed(
#         sp.csr_matrix.dot)(
#             sp.eye(N), f.fields(np.ones(N))
#         )

#     aa = da.from_delayed(row, dtype=float, shape=(1, N))

#     fields += [aa]

# print(rowSum(fields).compute())
# print("Run 2 in: ", time.time()-tc)

tc = time.time()
deriv = []
for f in F:

    jvec = f.Jvec(np.ones(N))
    jtjvec = f.Jtvec(jvec)
#     jtjvec = f.JtJvec(np.ones(N))


    deriv += [jtjvec]



aa = rowSum(deriv).compute()
print("Run 2 in: ", time.time()-tc)


# F[0].Jvec(vec).compute()
