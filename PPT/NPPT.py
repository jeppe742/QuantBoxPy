import numpy as np
from numba import autojit
np.set_printoptions(precision=2)

# @autojit
# def partial_transpose_numba(rho, block_size):
#     (n, m) = rho.shape
#     output = np.zeros((n,m))
#     for i in range(n/block_size[0]):
#         for j in range(m/block_size[0]):
#             for k in range(block_size[0]):
#                 for l in range(block_size[1]):
#                     output[i+k,j+l] = rho[i+l,j+k]
#     return output


# def partial_transpose(rho, block_size):
#     (n, m) = rho.shape
#     rho = rho.reshape(int(n/block_size[0]),block_size[0], int(m/block_size[1]), block_size[1] ).transpose(0,2,1,3)
#     rho_T = rho.transpose(0,1,3,2)
#     rho = rho_T.transpose(0,2,1,3).reshape(n,m)

# def partial_transpose2(rho, block_size):
#     (n, m) = rho.shape
#     rho = rho.reshape(int(n/block_size[0]),block_size[0], int(m/block_size[1]), block_size[1] ).transpose(0,2,1,3).transpose(0,1,3,2).transpose(0,2,1,3).reshape(n,m)
    # rho_T = rho.transpose(0,1,3,2)
    # rho = rho_T.transpose(0,2,1,3).reshape(n,m)

def NPPT(rho, dim_A, dim_B, verbose=0):
    (n, m) = rho.shape
    if(verbose==1):
        print("rho=")
        print(rho)
    
    if n != m:
        print("Matrix is not a square matrix")
        return
    if np.trace(rho)!=1:
        print("Trace!=1. Matrix is not a density matrix")
        print("Trace(rho) = %3.4f"%np.trace(rho))    
        return
    if np.min(np.linalg.eigvals(rho))<-1e-10:
        print("matrix is not positive")
    rho = rho.reshape(dim_A, dim_B, dim_A, dim_B).transpose(0,2,1,3)
    rho_T = rho.transpose(0,1,3,2)
    rho = rho_T.transpose(0,2,1,3).reshape(n,m)

    if(verbose==1):
        print("(T_B)(rho)=")
        print(rho)
    
    eigvals=np.linalg.eigvals(rho)
    if(verbose==1):
        print("eigenvals : ",eigvals)
    
    if np.min(eigvals)<-1e-10:
        print("State is NPPT")
        return
    print("State is PPT")
    return
a=0.5
rho = (1/(7*a+1))*np.array([
                [a,0,0,0,0,a,0,0],
                [0,a,0,0,0,0,a,0],
                [0,0,a,0,0,0,0,a],
                [0,0,0,a,0,0,0,0],
                [0,0,0,0,0.5*(1+a),0,0,0.5*np.sqrt(1-a**2)],
                [a,0,0,0,0,a,0,0],
                [0,a,0,0,0,0,a,0],
                [0,0,a,0,0.5*np.sqrt(1-a**2),0,0,0.5*(1+a)]
                ])
# p=0.334
# rho = 1/4*np.array([
#                 [1-p,0,0,0],
#                 [0,p+1,-2*p,0],
#                 [0,-2*p,p+1,0],
#                 [0,0,0,1-p]
#     ])
# rho = np.array([
#                 [1,2,3,4,5,6,7,8],
#                 [9,10,11,12],
#                 [13,14,15,16]
#                 ])
# rho = 1/8 *np.eye(8)
NPPT(rho,2,4, verbose=1)