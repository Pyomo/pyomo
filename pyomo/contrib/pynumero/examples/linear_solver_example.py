from pyomo.contrib.pynumero.linalg.solvers import MUMPSSymLinearSolver
from scipy.sparse import coo_matrix
import numpy as np

# create matrix
row = np.array([0, 1, 2, 1, 2, 2, 0, 0, 1])
col = np.array([0, 0, 0, 1, 1, 2, 1, 2, 2])
data = np.array([1.0, 7.0, 3.0, 4.0, -5.0, 6.0, 7.0, 3.0, -5.0], dtype='d')
A = coo_matrix((data, (row, col)), shape=(3, 3))

# create rhs
b = np.array([1, 1, 0], dtype='d')

# create symmetric linear solver
linear_solver = MUMPSSymLinearSolver()
x = linear_solver.solve(A, b)

# solution
print("solution =",x)
print("residual =", np.linalg.norm(b - A.dot(x), ord=np.inf))
