# -*- coding: utf-8 -*-
import picos
import cvxopt as cvx
import numpy as np
from scipy.special import binom
from bose_trace import bose_trace_channel
np.set_printoptions(precision=3)

def bose_trace(σ_AB, dim_A, dim_B, k, extend_system=1):
    """
     Given a state in 𝓗_A ⊗ Sym^k(𝓗_B), trace out k-1 of the systems in the symmetric 
    
    tr_B^k-1: 𝓛(𝓗_A ⊗ Sym^k(𝓗_B)) -> 𝓛(𝓗_A ⊗ 𝓗_B)

    :param σ_AB: the state to perform the map on
    :param dim_A: dimensions of  𝓗_A
    :param dim_B: dimensions of  𝓗_B
    :param k: how many extensions we have
    :param extend_system: Which system has been extended. 𝓗_A is extended if 0, and 𝓗_B if 1
    
    """
    if extend_system == 1:
        #Create both the idendity and bose-trace channel
        C_id = np.eye(dim_A**2, dim_A**2)
        C_bose = bose_trace_channel(dim_B, k)
        C_T = np.tensordot(C_id, C_bose, axes=0)
        #We now just need to do a series of reshaping and transposing, in order for the indecies to be correct
        C_T = C_T.transpose(0,2,1,3)\
        .reshape(dim_A,dim_A,dim_B,dim_B,dim_A,dim_A,int(binom(dim_B+k-1,k)),int(binom(dim_B+k-1,k)))\
        .transpose(0,2,1,3,4,6,5,7)\
        .reshape(dim_A**2*dim_B**2, int(binom(dim_B+k-1,k))**2*dim_A**2)
    else:
        C_id = np.eye(dim_B**2, dim_B**2)
        C_bose = bose_trace_channel(dim_A, k)
        C_T = np.tensordot(C_bose, C_id, axes=0) #The joint channel is defined as the tensorproduct 
        #We now just need to do a series of reshaping and transposing, in order for the indecies to be correct
        C_T = C_T.transpose(0,2,1,3)\
        .reshape(dim_B,dim_B,dim_A,dim_A,dim_B,dim_B,int(binom(dim_A+k-1,k)),int(binom(dim_A+k-1,k)))\
        .transpose(0,2,1,3,4,6,5,7)\
        .reshape(dim_A**2*dim_B**2, int(binom(dim_A+k-1,k))**2*dim_B**2)

    #TODO: implement bose-trace using sparse matrix    
    C_T = cvx.matrix(C_T, tc='z') #Since picos uses cvx, cast the matrix to a cvx matrix

    #This is where the magic happens. Picos stores matricies as X = X*factors + constant, where X is the matrix flattend. 
    #To apply a channel, simply multiply the factor and channel together: T(x) = X*C_T*factors + constant
    newfacs = {}
    for x in σ_AB.factors:
        newfacs[x] = C_T * σ_AB.factors[x]
    if σ_AB.constant: #Not sure if needed. Copied from picos partial trace, just in case
        cons = C_T * σ_AB.constant
    else:
        cons = None

    return picos.AffinExp(newfacs, cons, (dim_A*dim_B, dim_A*dim_B), 'Tr_B^N-1'  + '(' + σ_AB.string + ')')
    
def check_exstendibility(ρ, σ_AB, dim_A, dim_B, k,extend_system=1):
    
    '''
    Check if σ_AB is an extension, by checking constraints

    :param ρ: input state
    :param σ_AB: solution to the proposed extension σ_AB. σ_AB should be 𝓗_A ⊗ 𝓗_B^(⊗k)
    :param dim_A: dimensions of system ρ_A
    :param dim_B: dimsenions of system ρ_B
    :param extend_system: Which system that is extended. Specify either 0 for system A or 1 for system B.
    '''
    print("----------------------------------------------------")
    print("Checking that the solution fulfills the constraints:")
    print("----------------------------------------------------")

    #Checking the partial trace, with a tolerence of 1e-7
    if all((np.real(picos.trace(σ_AB).value)-1)<1e-7):
        print("tr(σ_AB) = 1          :    TRUE")
    else:
        print("tr(σ_AB) = 1          :    FALSE")
    
    #Checking that each extension is equal to ρ
    σ_i_constraints=np.allclose(bose_trace(σ_AB, dim_A, dim_B, k, extend_system=extend_system).value, ρ.value)
    if  σ_i_constraints:
        print("tr_B^N-1(σ_AB) = ρ   :    TRUE")
    else:
        # for i, σ_i in enumerate(σ_i_constraints):  #Loop over the extensions which does not equal ρ
        #     if not σ_i:
        print("tr_B^N-1(σ_AB) = ρ   :    FALSE")

    if all((np.linalg.eigvals(np.asarray(σ_AB.value))+1e-7)>0): #Check if the matrix is positive with a tolerence of 1e-7
        print("σ_AB > 0              :    TRUE")
    else:
        print("σ_AB > 0              :    FALSE")
        print("eigenvals are :")
        print(np.linalg.eigvals(np.asarray(σ_AB.value)))

def extendibility(ρ, dim_A, dim_B, k=2, verbose=0, extend_system=1):
    '''
    Checks if the state ρ is k-extendible.
    --------------------------------------
    Given an input state ρ ∈ 𝓗_A ⊗ 𝓗_B. Try to find an extension σ_AB_1..B_k ∈ 𝓗_A ⊗ 𝓗_B^(⊗k), such that (σ_AB)_i=ρ

    :param ρ: The state we want to check
    :param dim_A: Dimensions of system A
    :param dim_B: Dimensions of system B
    :param k: The extendibility order
    :param extend_system: Which system to create the copies from. Specify either 0 for system A or 1 for system B.
    '''

    #Define variables, and create problem
    ρ = picos.new_param('ρ',ρ)
    problem = picos.Problem()

    if extend_system==1:
        σ_AB = problem.add_variable('σ_AB', (dim_A*binom(dim_B+k-1,k), dim_A*binom(dim_B+k-1,k)),'hermitian')
    else:
        σ_AB = problem.add_variable('σ_AB', (binom(dim_A+k-1,k)*dim_B, binom(dim_A+k-1,k)*dim_B),'hermitian')
    #Set objective to a feasibility problem. The second argument is ignored by picos, so set some random scalar function.
    problem.set_objective('find', picos.trace(σ_AB))

    #Add constrains
    problem.add_constraint(σ_AB>>0) 
    problem.add_constraint(picos.trace(σ_AB)==1)
    problem.add_constraint(bose_trace(σ_AB, dim_A, dim_B, k, extend_system=extend_system)==ρ )

    print("\nChecking for %d extendibility..."%(k))

    #Solve the SDP either silently or verbose
    if verbose:
        try:
            print(problem)  
            problem.solve(verbose=verbose, solver='mosek')
            print(problem.status)
            check_exstendibility(ρ, σ_AB, dim_A, dim_B, k, extend_system=extend_system)   #Run a solution check if the user wants
        except UnicodeEncodeError:
            print("!!!Can't print the output due to your terminal not supporting unicode encoding!!!\nThis can be solved by setting verbose=0, or running the function using ipython instead.")
    else:
        problem.solve(verbose=verbose, solver='mosek')
        print(problem.status)
    return σ_AB

if __name__=='__main__':

    import numpy as np
    a=0.5   
    ρ = (1/(7*a+1))*cvx.matrix([
                    [a,0,0,0,0,a,0,0],
                    [0,a,0,0,0,0,a,0],
                    [0,0,a,0,0,0,0,a],
                    [0,0,0,a,0,0,0,0],
                    [0,0,0,0,0.5*(1+a),0,0,0.5*np.sqrt(1-a**2)],
                    [a,0,0,0,0,a,0,0],
                    [0,a,0,0,0,0,a,0],
                    [0,0,a,0,0.5*np.sqrt(1-a**2),0,0,0.5*(1+a)]
                    ])
    # p=0.4
    # ρ = 1.0/4.0*cvx.matrix([
    #                     [1-p,0,0,0],
    #                     [0,p+1,-2*p,0],
    #                     [0,-2*p,p+1,0],
    #                     [0,0,0,1-p]
    #                     ])

    # ρ = 1.0/4*np.eye(4,4)
    # ρ = cvx.matrix([[0.2,2,3],[4,0.6,6],[1,0.2,1]])

    # Maximally entangled state
    # ρ = 1/2*cvx.matrix([[1,0,0,1],
    #                     [0,0,0,0],
    #                     [0,0,0,0],
    #                     [1,0,0,1]])
    # theta = np.pi/6
    # b = 0.04
    # cth = np.cos(theta)
    # ph = np.exp(1j*theta)
    # A = np.matrix([[2*cth ,0  ,0],
    #                 [0     ,1/b,0],
    #                 [0     ,0  ,b]])

    # B = np.matrix([[0   ,-ph , 0 ], 
    #                 [-cth,  0 , 0],
    #                 [0   ,  0 , 0]])

    # C = np.matrix([[0   , 0 ,-np.conj(ph)], 
    #                 [0   , 0 ,     0      ], 
    #                 [-cth, 0 ,     0     ]])

    # D = np.matrix([[0  ,  0  , 0  ], 
    #                 [0  ,  0  ,-ph ],
    #                 [0  , -cth, 0  ]])

    # E = np.matrix([[b  ,  0  , 0  ],
    #                 [0  ,2*cth, 0  ],
    #                 [0  ,  0  , 1/b]])

    # F = np.matrix([[1/b,  0  ,0],
    #                 [0  ,  b  ,0],
    #                 [0  ,  0  ,2*cth]])
    # rho = np.array([[A  , B   ,C],
    #                 [B.H, E   ,D],
    #                 [C.H, D.H ,F]])
    # rho = rho.transpose(0,2,1,3).reshape(9,9) #Transform into 2d matrix
    # rho = rho/np.trace(rho) #Normalize
    # extendibility(rho, 3, 3, verbose=0, k=6, extend_system=0)
    # p = 0
    # d = 4
    # Ω = np.eye(d).flatten().reshape(d**2,1)
    # ρ_Ω=picos.new_param('ρ_Ω',Ω.dot(Ω.T))
    # 𝔽 = picos.partial_transpose(ρ_Ω, (d,d),0)
    # 𝔽 = np.asarray(𝔽.value)

    # id = np.eye(d*d)
    # σ_sym = 1/(d**2+d)*(id + 𝔽)
    # σ_asym = 1/(d**2-d)*(id - 𝔽)

    # ρ = p*σ_sym + (1-p)*σ_asym
    def isotropic_state(alpha, d):
        Ω = np.eye(d).flatten().reshape(d**2,1)
        return (1-alpha)/d**2 *(np.eye(d*d)) + alpha*(np.dot(Ω,Ω.T)/d)
    extendibility(ρ, 2, 2, verbose=1, k=2, extend_system=1)
    # extendibility(isotropic_state(0.4,2),2,2, verbose=1, k=2, extend_system=0)