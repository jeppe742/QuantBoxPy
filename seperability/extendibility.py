import picos
import cvxopt as cvx

def check_exstendibility(rho, sigma_AB):
    a=1

def extendibility(rho, dim_A, dim_B, k=2):
    '''
    Checks if the input state is k-exstendible

    :param rho: The state we want to check
    :param dim_A: Dimensions of system A
    :param dim_B: Dimensions of system B
    :param k: The extendibility order
    '''

    #Define variables, and create problem
    rho = picos.new_param('rho',rho)
    problem = picos.Problem()
    sigma_AB = problem.add_variable('sigma_AB',(dim_A*dim_B**2,dim_A*dim_B**2),'hermitian')

    #Set objective to a feasibility problem
    problem.set_objective('max',picos.trace(sigma_AB))

    #Add constrains
    problem.add_constraint(sigma_AB>>0) 
    problem.add_constraint(picos.trace(sigma_AB)==1)
    problem.add_list_of_constraints([picos.partial_trace(sigma_AB,i,(dim_A,dim_B,dim_B))==rho for i in range(1,k+1)],'i','1...'+str(k))

    problem.set_option('handleBarVars',False)       #Needed in order for Mosek to work with SDPs
    print("\n Checking for %d extendibility. The SDP is described as :"%(k))
    print(problem)  
    problem.solve(verbose=1, solver='mosek')
    print(sigma_AB.value)
    print("Checking if solution is correct...")
    print((picos.trace(sigma_AB).value))
    check_exstendibility(rho, sigma_AB)
if __name__=='__main__':
    import numpy as np
    # a=0.5   
    # rho = (1/(7*a+1))*cvx.matrix([
    #                 [a,0,0,0,0,a,0,0],
    #                 [0,a,0,0,0,0,a,0],
    #                 [0,0,a,0,0,0,0,a],
    #                 [0,0,0,a,0,0,0,0],
    #                 [0,0,0,0,0.5*(1+a),0,0,0.5*np.sqrt(1-a**2)],
    #                 [a,0,0,0,0,a,0,0],
    #                 [0,a,0,0,0,0,a,0],
    #                 [0,0,a,0,0.5*np.sqrt(1-a**2),0,0,0.5*(1+a)]
    #                 ])
    # p=0.4
    # rho = 1.0/4.0*cvx.matrix([
    #                     [1-p,0,0,0],
    #                     [0,p+1,-2*p,0],
    #                     [0,-2*p,p+1,0],
    #                     [0,0,0,1-p]
    #                     ])
    rho = 1.0/4*np.eye(4,4)
    # rho = cvx.matrix([[0.2,2,3],[4,0.6,6],[1,0.2,1]])
    extendibility(rho,2,2)