import picos
import cvxopt as cvx
import numpy as np

def get_sigma_AB_i( sigma_AB, dim_A, dim_B, i, k, extend_system=1):
    '''
    Get the i'th extension of σ_AB
    --------------------------------------------------------------
    Given a σ_AB_1...B_k ∈ 𝓗_A ⊗ 𝓗_B^(⊗k) calculate 
    σ_AB_i = tr_B1..B_(i-1)B_(i+1)...B_k(σ_AB_1...B_k)

    :param sigma_AB: input state including all extensions
    :param dim_A: dimensions of system σ_A
    :param dim_B: dimsenions of system σ_B
    :param i: The system for which we want the reduced density matrix
    :param k: number of extensions we have
    '''

    index = i #This is used to keep track of which system not to trace out

    #Create a list of the dimensions of our system
    if extend_system==1:
        dim = [dim_A]
        dim.extend([dim_B for _ in range(k)]) # Dimensions of our system

        #Calculate first trace
        if index==1:
            sigma_AB_i = picos.partial_trace(sigma_AB, index+1, dim )
        else:
            sigma_AB_i = picos.partial_trace(sigma_AB, index-1, dim )
            index -= 1

        #Loop over the rest of the traces
        for j in range(k-2):
            dim = [dim_A]
            dim.extend([dim_B for i in range(k-1-j)])
            if index==1:
                sigma_AB_i = picos.partial_trace(sigma_AB_i, index+1, dim )
            else:
                sigma_AB_i = picos.partial_trace(sigma_AB_i, index-1, dim )
                index -= 1

    else:
        dim = [dim_A for _ in range(k)]
        dim.append(dim_B) # Dimensions of our system
    
    #Calculate first trace
        if index==0:
            sigma_AB_i = picos.partial_trace(sigma_AB, index+1, dim )
        else:
            sigma_AB_i = picos.partial_trace(sigma_AB, index-1, dim )
            index -= 1

        #Loop over the rest of the traces
        for j in range(k-2):
            dim = [dim_A for _ in range(k-1-j)]
            dim.append(dim_B)
            if index==0:
                sigma_AB_i = picos.partial_trace(sigma_AB_i, index+1, dim )
            else:
                sigma_AB_i = picos.partial_trace(sigma_AB_i, index-1, dim )
                index -= 1

    return sigma_AB_i
 
def check_exstendibility(rho, sigma_AB, dim_A, dim_B, k,extend_system=1):
    '''
    Check if σ_AB is an extension, by checking constraints

    :param rho: input state
    :param sigma_AB: solution to the proposed extension σ_AB. σ_AB should be 𝓗_A ⊗ 𝓗_B^(⊗k)
    :param dim_A: dimensions of system ρ_A
    :param dim_B: dimsenions of system ρ_B
    :param extend_system: Which system that is extended. Specify either 0 for system A or 1 for system B.
    '''
    print("----------------------------------------------------")
    print("Checking that the solution fulfills the constraints:")
    print("----------------------------------------------------")

    #Checking the partial trace, with a tolerence of 1e-7
    if all((np.real(picos.trace(sigma_AB).value)-1)<1e-7):
        print("tr(σ_AB) = 1    :    TRUE")
    else:
        print("tr(σ_AB) = 1    :    FALSE")
    
    #Checking that each extension is equal to ρ
    sigma_i_constraints=[np.allclose(get_sigma_AB_i(sigma_AB, dim_A, dim_B, i, k, extend_system=extend_system).value,rho.value) for i in range(1,k+1)]
    if  all(sigma_i_constraints):
        print("(σ_AB)_i = ρ    :    TRUE")
    else:
        for i, sigma_i in enumerate(sigma_i_constraints):  #Loop over the extensions which does not equal ρ
            if not sigma_i:
                print("(σ_AB)_%d = ρ   :    FALSE"%(i))

    if all((np.linalg.eigvals(np.asarray(sigma_AB.value))+1e-7)>0): #Check if the matrix is positive with a tolerence of 1e-7
        print("σ_AB > 0        :    TRUE")
    else:
        print("σ_AB > 0        :    FALSE")
        print("eigenvals are :")
        print(np.linalg.eigvals(np.asarray(sigma_AB.value)))

def extendibility(rho, dim_A, dim_B, k=2, verbose=0, extend_system=1):
    '''
    Checks if the state ρ is k-extendible.
    --------------------------------------
    Given an input state ρ ∈ 𝓗_A ⊗ 𝓗_B. Try to find an extension σ_AB_1..B_k ∈ 𝓗_A ⊗ 𝓗_B^(⊗k), such that (σ_AB)_i=ρ

    :param rho: The state we want to check
    :param dim_A: Dimensions of system A
    :param dim_B: Dimensions of system B
    :param k: The extendibility order
    :param extend_system: Which system to create the copies from. Specify either 0 for system A or 1 for system B.
    '''

    #Define variables, and create problem
    rho = picos.new_param('ρ',rho)
    problem = picos.Problem()
    if extend_system==1:
        sigma_AB = problem.add_variable('σ_AB', (dim_A*dim_B**k, dim_A*dim_B**k),'hermitian')
    else:
        sigma_AB = problem.add_variable('σ_AB', (dim_A**k*dim_B, dim_A**k*dim_B),'hermitian')
    #Set objective to a feasibility problem. The second argument is ignored by picos, so set some random scalar function.
    problem.set_objective('find', picos.trace(sigma_AB))

    #Add constrains
    problem.add_constraint(sigma_AB>>0) 
    problem.add_constraint(picos.trace(sigma_AB)==1)
    problem.add_list_of_constraints([get_sigma_AB_i(sigma_AB, dim_A, dim_B, i, k, extend_system=extend_system)==rho for i in range(1, k+1)],'i','1...'+str(k))

    print("\nChecking for %d extendibility..."%(k))

    #Solve the SDP either silently or verbose
    if verbose:
        try:
            print(problem)  
            problem.solve(verbose=verbose, solver='mosek')
            check_exstendibility(rho, sigma_AB, dim_A, dim_B, k, extend_system=extend_system)   #Run a solution check if the user wants
        except UnicodeEncodeError:
            print("!!!Can't print the output due to your terminal not supporting unicode encoding!!!\nThis can be solved by setting verbose=0, or running the function using ipython instead.")
    else:
        problem.solve(verbose=verbose, solver='mosek')
