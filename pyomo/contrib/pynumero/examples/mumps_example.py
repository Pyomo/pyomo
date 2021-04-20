import numpy as np
import scipy.sparse as sp
from scipy.linalg import hilbert
from pyomo.contrib.pynumero.linalg.mumps_interface import MumpsCentralizedAssembledLinearSolver

# create the matrix and the right hand sides
N = 1000
A = sp.coo_matrix(hilbert(N) + np.identity(N))  # a well-condition, symmetric, positive-definite matrix with off-diagonal entries
true_x1 = np.arange(N)
true_x2 = np.array(list(reversed(np.arange(N))))
b1 = A * true_x1
b2 = A * true_x2

# solve
solver = MumpsCentralizedAssembledLinearSolver()
x1 = solver.solve(A, b1)
x2 = solver.solve(A, b2)
assert np.allclose(x1, true_x1)
assert np.allclose(x2, true_x2)

# only perform factorization once
solver = MumpsCentralizedAssembledLinearSolver()
solver.do_symbolic_factorization(A)
solver.do_numeric_factorization(A)
x1 = solver.do_back_solve(b1)
x2 = solver.do_back_solve(b2)
assert np.allclose(x1, true_x1)
assert np.allclose(x2, true_x2)

# Tell Mumps the matrix is symmetric
# Note that the answer will be incorrect if both the lower
# and upper portions of the matrix are given.
solver = MumpsCentralizedAssembledLinearSolver(sym=2)
A_lower_triangular = sp.tril(A)
x1 = solver.solve(A_lower_triangular, b1)
assert np.allclose(x1, true_x1)

# Tell Mumps the matrix is symmetric and positive-definite
solver = MumpsCentralizedAssembledLinearSolver(sym=1)
A_lower_triangular = sp.tril(A)
x1 = solver.solve(A_lower_triangular, b1)
assert np.allclose(x1, true_x1)

# Set options
solver = MumpsCentralizedAssembledLinearSolver(icntl_options={11: 2}) # compute error stats
solver.set_cntl(2, 1e-4) # set the stopping criteria for iterative refinement
solver.set_icntl(10, 5) # set the maximum number of iterations for iterative refinement to 5
solver.solve(A, b1)
assert np.allclose(x1, true_x1)


# Get information after the solve
print('Number of iterations of iterative refinement performed: ', solver.get_infog(15))
print('scaled residual: ', solver.get_rinfog(6))
