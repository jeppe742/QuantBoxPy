import picos as pic
import cvxopt as cvx
import numpy as np

# K = cvx.matrix([[1+1j,2,3],
#                 [2,5+3j,4],
#                 [3+8j,4,9-1j]])

# k = cvx.matrix([[1+1j,2,3,1j],
#                 [2,5+3j,4,8],
#                 [3+8j,4,9-1j,-6j],
#                 [1,2,3,4]])
# k = cvx.matrix([[1,2,3,1],
#                 [2,5,4,8],
#                 [3,4,9,-6],
#                 [1,2,3,4]])

k = cvx.matrix([[1,2,3,1],
                [2,5,4,8],
                [3,4,9,1],
                [1,2,3,4]])

P = pic.Problem()

X = P.add_variable('X',(4,4),'hermitian')
Y = P.add_variable('Y',(4,4),'hermitian')
K = pic.new_param('K',k)
# P.set_objective('min','I'|0.5*(X+Y))
P.set_objective('min',0.5*pic.trace(X)+0.5*pic.trace(Y))
P.add_constraint(((X & -K.H)//(-K & Y)) >>0 )
P.add_constraint(X>>0)
P.add_constraint(Y>>0)
# P.add_constraint(X.real>0)
# P.add_constraint(Y.real>0)
print(P)

P.solve(verbose=1, solver='mosek')
print('SDP status: '+P.status)
print('\nSDP value = {:.3f}'.format(P.obj_value()))
print('TraceNorm = %3.3f'%(sum(np.linalg.svd(k)[1])))
print(X)
print(Y)
print(np.linalg.svd(X.value)[1])
print(np.linalg.svd(Y.value)[1])