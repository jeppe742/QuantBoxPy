import numpy as np
from helpers import bra, ket, sym_vectors

def C(m, n):
    """                                                    
    Calculate C = Σ_ij |i><j| c_ij(m,n)
    where  c_ij(m,n) = sqrt(m_i*m_j)/k <n'(i)|m'(j)>

    :param m: "symmetric subspace" vector |m>. Needs to be a numpy array
    :param n: same as m

    EX:
    >>> C(np.array([2,0]),np.array([1,1]))
    [[0 0.707]
     [0   0  ]]
    """
    k=sum(m)
    d = m.shape[0]
    C=np.zeros((d,d))
    for i in range(d):
        for j in range(d):
            #Since we still need m, n later, we need to create a new copy of the array
            mm = m.copy()
            nn = n.copy()
            #Reduce the occupancy number by one
            mm[i] = mm[i]-1
            nn[j] = nn[j]-1
            C[i, j] = np.sqrt(m[i]*n[j])/k * int(all(mm.T==nn)) #The last term here, just correspods to the kronecker delta, between nn and mm
    return C


def bose_trace_channel(d, k):
    from scipy.special import binom
    """
    Returns the bose-trace channel, using the choi representation:
    C_T = (id ⊗ tr_B^k-1)|Ω><Ω|
    = Σ_mn |m><n| ⊗ Σ_ij |i><j| c_ij(m,n)

    This corresponds to tracing out all but 1, of the systems in the symmetric subspace.
    The channel has been reshaped, such that the map can be applied by matrix multiplication:
    tr_B^k-1(X) = C_T * X.flattend()
    """
    d_sym = int(binom(d+k-1, k)) #dimensions of the symmetric subspace
    C_T = np.zeros((d_sym, d_sym, d, d)) #Initialize the matrix
    for i, m in enumerate(np.asarray(sym_vectors(d, k))):
        for j, n in enumerate(np.asarray(sym_vectors(d, k))):
            mm = ket(i, d_sym) # conver into the computational basis
            nn = bra(j, d_sym)
            C_T += np.tensordot(mm*nn, C(m, n), axes=0)
    return C_T.transpose(2, 3, 0, 1).reshape(d**2, d_sym**2)

