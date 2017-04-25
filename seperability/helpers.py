import numpy as np

def bra(i, dims):
    """
    creates a basis vector from the state <i|
    :param i: the state number <i|. Can either be an in or a list of ints
    :param dims: the dimensions of the state

    EX:
    >>> bra([0,1],2)
    [1 0 0 0]
    """
    if not isinstance(i, list):
        i=[i]
    #Single qubit
    if len(i)==1:
        val = np.zeros((dims,1))
        val[i] = 1
        return val.reshape(1,dims)
    #multiple qubits. we need to tensor them together
    val = np.ones((1,1)) #initialize variable, so we have something to tensor with, the first time
    for x in i:
        val = np.tensordot(val,ket([x],dims), axes=0).transpose(0,2,1,3)
        val = val.reshape(val.shape[0]*val.shape[1],val.shape[2]*val.shape[3])
    return val.reshape(1,val.shape[0])


def ket(i, dims):
    """
    creates a basis vector from the state |i>

    :param i: the state number |i>. Can either be an int or list of ints
    :param dims: the dimensions of the state

    EX:
    >>> ket([1,0],2)
    [[0] 
     [0]
     [1]
     [0]]
    """
    if not isinstance(i, list):
        i=[i]
    #Single qubit
    if len(i)==1:
        val = np.zeros((dims,1))
        val[i] = 1
        return val.reshape(dims,1)
    #multiple qubits. we need to tensor them together
    val = np.ones((1,1)) #initialize variable, so we have something to tensor with, the first time
    for x in i:
        val = np.tensordot(val,ket([x],dims), axes=0).transpose(0,2,1,3)
        val = val.reshape(val.shape[0]*val.shape[1],val.shape[2]*val.shape[3])
    return val.reshape(val.shape[0],1)

def sym_vectors(d, k):
    """
    Caculates all the symmetric vectors that span Sym^k(ℂ^d).
    This corresponds, to finding all combinations of numbers n=(n_1, n_2,..,n_d) such that Σ n_i=k

    :param d: local dimensions of our system. Or the number of elements in n
    :param k: Number of symmetric extensions. Or the sum of elements in n
    

    EX:
    >>> sym_vectors(2,3) 
    [[3,0]
     [2,1]
     [1,2]
     [0,3]]
    """
    vectors = []
    if d == 1: #tail of the recursive call. 
        vectors.append(k)
        return vectors
    #Try the different combinations for the first number, and recursively call the rest of the d-1 numbers, for which the sum now needs to be k-i
    for i in range(k, -1, -1):
        for vector in sym_vectors(d-1, k-i):
            tmp_vector = []
            #Make sure vector is a list
            if not isinstance(vector, list):
                vector = [vector]
            #Stitch the results together in a list
            tmp_vector.extend([i])
            tmp_vector.extend(vector)
            vectors.append(tmp_vector)
    return vectors



if __name__=="__main__":
    sym_vectors(3,2)
    print(ket([0,1],2))
    print(bra([1,0],2))