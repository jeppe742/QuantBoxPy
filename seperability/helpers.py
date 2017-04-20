import numpy as np

def ket(x, dims):
    if not isinstance(x, list):
        x=[x]
    #Single qubit
    if len(x)==1:
        val = np.zeros((dims,1))
        val[x] = 1
        return val.reshape(dims,1)
    #multiple qubits. we need to tensor then together
    val = 1 #initialize variable, so we have something to tensor with, the first time
    for i in x:
        val = np.tensordot(val,ket([i],dims), axes=0)
    return val.reshape(val.shape[0],1)

def sym_vectors(d, k):
    """
    Caculates all the symmetric vectors that span Sym^k(ℂ^d).
    This corresponds, to finding all combinations of numbers n=(n_1, n_2,..,n_d) such that Σ n_i=k

    :param d: local dimensions of our system. Or the number of elements in n
    :param k: Number of symmetric extensions. Or the sum of elements in n
    
    """
    vectors = []
    if d == 1:
        vectors.append(k)
        return vectors
    for i in range(k, -1, -1):
        for vector in sym_vectors(d-1, k-i):
            #Make sure vector is a list
            if not isinstance(vector, list):
                vector = [vector]
            vector.extend([i])
            vectors.append(vector)
    return vectors


def bra(x, dims):
    if not isinstance(x, list):
        x=[x]
    #Single qubit
    if len(x)==1:
        val = np.zeros((dims,1))
        val[x] = 1
        return val.reshape(1,dims)
    #multiple qubits. we need to tensor then together
    val = 1 #initialize variable, so we have something to tensor with, the first time
    for i in x:
        val = np.tensordot(val,ket([i],dims), axes=0)
    return val.reshape(1,val.shape[0])

if __name__=="__main__":
    aa=sym_vectors(3,2)
    print(ket([1,0],2))
    print(bra([1,0],2))