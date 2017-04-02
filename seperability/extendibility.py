import picos
import cvxopt as cvx

def get_σ_AB_i(σ_AB, i, dim_A, dim_B):

    if σ_AB.size==(dim_A,dim_B):
        return σ_AB
    
    get_σ_AB_i(picos.partial_trace(σ_AB,))

def check_exstendibility(ρ, σ_AB, dim_A, dim_B, k):
    '''
    Check if σ_AB is an extension, by checking constraints

    :param ρ: input state
    :param σ_AB: solution to the proposed extension σ_AB. σ_AB should be 𝓗_A ⊗ 𝓗_B^(⊗k)
    :param dim_A: dimensions of system ρ_A
    :param dim_B: dimsenions of system ρ_B
    '''
    print("----------------------------------------------------")
    print("Checking that the solution fulfills the constraints:")
    print("----------------------------------------------------")

    #Checking the partial trace, with a tolerence of 1e-7
    if all((np.real(picos.trace(σ_AB).value)-1)<1e-7):
        print("tr(σ_AB) = 1    :    TRUE")
    else:
        print("tr(σ_AB) = 1    :    FALSE")
    
    #Checking that each extension is equal to ρ
    σ_i_constraints=[np.allclose(picos.partial_trace(σ_AB,i,(dim_A,dim_B,dim_B)).value,ρ.value) for i in range(1,3)]
    if  all(σ_i_constraints):
        print("(σ_AB)_i = ρ   :    TRUE")
    else:
        for i, σ_i in enumerate(σ_i_constraints):  #Loop over the extensions which does not equal ρ
            if not σ_i:
                print("(σ_AB)_%d = ρ   :    FALSE"%(i))

    if all((np.linalg.eigvals(np.asarray(σ_AB.value))+1e-7)>0): #Check if the matrix is positive with a tolerence of 1e-7
        print("σ_AB > 0        :    TRUE")
    else:
        print("σ_AB > 0        :    FALSE")
        print("eigenvals are :")
        print(np.linalg.eigvals(np.asarray(σ_AB.value)))

def extendibility(ρ, dim_A, dim_B, k=2, verbose=0):
    '''
    Checks if the state ρ is k-extendible.
    --------------------------------------
    Given an input state ρ in 𝓗_A ⊗ 𝓗_B. Try to find an extension σ_AB_1..B_k ∈ 𝓗_A ⊗ 𝓗_B^(⊗k), such that (σ_AB)_i=ρ

    :param ρ: The state we want to check
    :param dim_A: Dimensions of system A
    :param dim_B: Dimensions of system B
    :param k: The extendibility order
    '''

    #Define variables, and create problem
    ρ = picos.new_param('ρ',ρ)
    problem = picos.Problem()
    σ_AB = problem.add_variable('σ_AB', (dim_A*dim_B**k, dim_A*dim_B**k),'hermitian')

    #Set objective to a feasibility problem. The second argument is ignored by picos, so set some random scalar function.
    problem.set_objective('find', picos.trace(σ_AB))

    #Add constrains
    problem.add_constraint(σ_AB>>0) 
    problem.add_constraint(picos.trace(σ_AB)==1)
    
    problem.add_list_of_constraints([picos.partial_trace(σ_AB, i, (dim_A, dim_B, dim_B))==ρ for i in range(1, k+1)],'i','1...'+str(k))

    print("\nChecking for %d extendibility..."%(k))

    #Solve the SDP either silently or verbose
    if verbose:
        try:
            print(problem)  
            problem.solve(verbose=verbose, solver='mosek')
            get_σ_AB_i(σ_AB,1, dim_A, dim_B)
            check_exstendibility(ρ, σ_AB, dim_A, dim_B, k)   #Run a solution check if the user wants
        except UnicodeEncodeError:
            print("!!!Can't print the output due to your terminal not supporting unicode encoding!!!\nThis can be solved by setting verbose=0, or running the function using ipython instead.")
    else:
        problem.solve(verbose=verbose, solver='mosek')


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
    extendibility(ρ,2,4, verbose=1)