import numpy as np
import sys
from helpers import bra, ket, sym_vectors
np.set_printoptions(precision=3)

M = [np.array([2,0]), np.array([1,1]), np.array([0,2])] #1x2
N = [np.array([2,0]), np.array([1,1]), np.array([0,2])] #1x2
m = [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])] #1x2
n = [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])] #1x2

# def sym_ket(i, dims):
#     x = np.zeros((dims,1))
#     for ii in i:
#         x[ii]=1
#     return x

# def sym_bra(i, dims):
#     x = np.zeros((dims,1))
#     for ii in i:
#         x[ii]=1
#     return x.reshape(1,dims)

def C(m,n):
    k=len(m)
    C=np.zeros((m.shape[0],n.shape[0]))
    for i in range(2):
        for j in range(2):
            mm = m.copy()
            mm[i] = mm[i]-1
            nn = n.copy()
            nn[j] = nn[j]-1
            C[i,j] = np.sqrt(m[i]*n[j])/k *np.dot(mm.T,nn)
    return C


def bose_trace_channel(d, k):
    from scipy.special import binom
    """
    C_T = Σ_mn |m><n| ⊗ Σ_ij |i><j| c_ij(m,n)
    """
    d_sym = int(binom(d+k-1,k))
    C_T = np.zeros((d_sym,d_sym,d,d))
    for i,m in enumerate(np.asarray(sym_vectors(d,k))):
        for j,n in enumerate(np.asarray(sym_vectors(d,k))):
            mm = ket(i,d_sym)
            nn = bra(j,d_sym)
            C_T += np.tensordot(mm*nn, C(m,n), axes=0)
            # C_T += np.tensordot(np.dot(sym_ket(m,3),sym_bra(n,3)), C(M[i],N[j]), axes=0)
    return C_T.transpose(2,3,0,1).reshape(d**2,d_sym**2)

