# -*- coding: utf-8 -*-
import picos
import cvxopt as cvx
import numpy as np
from scipy.special import binom
from bose_trace import bose_trace_channel
np.set_printoptions(precision=3)

def bose_trace(sigma_AB, dim_A, dim_B, k):
    """
     Given a state in 𝓗_A ⊗ Sym^k(𝓗_B), trace out k-1 of the systems in the symmetric B systems
    
    tr_B^k-1: 𝓛(𝓗_A ⊗ Sym^k(𝓗_B)) -> 𝓛(𝓗_A ⊗ 𝓗_B)

    :param sigma_AB: the state to perform the map on
    :param dim_A: dimensions of  𝓗_A
    :param dim_B: dimensions of  𝓗_B
    :param k: how many extensions we have
    
    
    """
    #Create both the idendity and bose-trace channel
    C_id = np.eye(dim_A**2, dim_A**2)
    C_bose = bose_trace_channel(dim_B, k)
    C_T = np.tensordot(C_id, C_bose, axes=0)
    #We now just need to do a series of reshaping and transposing, in order for the indecies to be correct
    C_T = C_T.transpose(0,2,1,3)\
    .reshape(dim_A,dim_A,dim_B,dim_B,dim_A,dim_A,int(binom(dim_B+k-1,k)),int(binom(dim_B+k-1,k)))\
    .transpose(0,2,1,3,4,6,5,7)\
    .reshape(dim_A**2*dim_B**2, int(binom(dim_B+k-1,k))**2*dim_A**2)
    
    C_T = cvx.matrix(C_T, tc='z') #Since picos uses cvx, cast the matrix to a cvx matrix

    #This is where the magic happens. Picos stores matricies as X = X*factors + constant, where X is the matrix flattend. 
    #To apply a channel, simply multiply the factor and channel together: T(x) = X*C_T*factors + constant
    newfacs = {}
    for x in sigma_AB.factors:
        newfacs[x] = C_T * sigma_AB.factors[x]
    if sigma_AB.constant: #Not sure if needed. Copied from picos partial trace, just in case
        cons = C_T * sigma_AB.constant
    else:
        cons = None

    return picos.AffinExp(newfacs, cons, (dim_A*dim_B, dim_A*dim_B), 'Tr_B^N-1'  + '(' + sigma_AB.string + ')')
    
def check_exstendibility(rho, sigma_AB, dim_A, dim_B, k):
    
    '''
    Check if sigma_AB is an extension, by checking constraints

    :param ρ: input state
    :param sigma_AB: solution to the proposed extension sigma_AB. sigma_AB should be 𝓗_A ⊗ 𝓗_B^(⊗k)
    :param dim_A: dimensions of system ρ_A
    :param dim_B: dimsenions of system ρ_B
    '''
    print("----------------------------------------------------")
    print("Checking that the solution fulfills the constraints:")
    print("----------------------------------------------------")

    #Checking the partial trace, with a tolerence of 1e-7
    if all((np.real(picos.trace(sigma_AB).value)-1)<1e-7):
        print("tr(σ_AB) = 1          :    TRUE")
    else:
        print("tr(σ_AB) = 1          :    FALSE")
    
    #Checking that each extension is equal to ρ
    sigma_i_constraints=np.allclose(bose_trace(sigma_AB, dim_A, dim_B, k).value, rho.value)
    if  sigma_i_constraints:
        print("tr_B^N-1(sigma_AB) = ρ   :    TRUE")
    else:
        print("tr_B^N-1(sigma_AB) = ρ   :    FALSE")

    if all((np.linalg.eigvals(np.asarray(sigma_AB.value))+1e-7)>0): #Check if the matrix is positive with a tolerence of 1e-7
        print("sigma_AB > 0              :    TRUE")
    else:
        print("sigma_AB > 0              :    FALSE")
        print("eigenvals are :")
        print(np.linalg.eigvals(np.asarray(sigma_AB.value)))

def extendibility(rho, dim_A, dim_B, k=2, verbose=0):
    '''
    Checks if the state ρ is k-extendible.
    --------------------------------------
    Given an input state ρ ∈ 𝓗_A ⊗ 𝓗_B. Try to find an extension σ_AB_1..B_k ∈ 𝓗_A ⊗ 𝓗_B^(⊗k), such that (σ_AB)_i=ρ
    Not that the extensions are only the B-system.
    :param ρ: The state we want to check
    :param dim_A: Dimensions of system A
    :param dim_B: Dimensions of system B
    :param k: The extendibility order
    
    '''

    #Define variables, and create problem
    rho = picos.new_param('ρ',rho)
    problem = picos.Problem()
    
    sigma_AB = problem.add_variable('σ_AB', (dim_A*binom(dim_B+k-1,k), dim_A*binom(dim_B+k-1,k)),'hermitian')
    #Set objective to a feasibility problem. The second argument is ignored by picos, so set some random scalar function.
    problem.set_objective('find', picos.trace(sigma_AB))

    #Add constrains
    problem.add_constraint(sigma_AB>>0) 
    problem.add_constraint(picos.trace(sigma_AB)==1)
    problem.add_constraint(bose_trace(sigma_AB, dim_A, dim_B, k)==rho )

    print("\nChecking for %d extendibility..."%(k))

    #Solve the SDP either silently or verbose
    if verbose:
        try:
            print(problem)  
            problem.solve(verbose=verbose, solver='mosek')
            print(problem.status)
            check_exstendibility(rho, sigma_AB, dim_A, dim_B, k)   #Run a solution check if the user wants
        except UnicodeEncodeError:
            print("!!!Can't print the output due to your terminal not supporting unicode encoding!!!\nThis can be solved by setting verbose=0, or running the function using ipython instead.")
    else:
        problem.solve(verbose=verbose, solver='mosek')
        print(problem.status)
    return sigma_AB
