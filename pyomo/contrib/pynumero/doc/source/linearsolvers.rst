Linear Solvers
==============

Efficient implementations of nonlinear optimization algorithms require fast and reliable linear solvers. PyNumero provides access to several libraries for the solution of the sparse linear systems that arise in nonlinear programming. Because PyNumero stores matrices in Numpy/Scipy objects, all subroutines available in the Numpy ecosystem can be used when writing algorithms in PyNumero. This includes the  Scipy direct and iterative solvers as well as any other Python package based on Numpy such as PyTrillinos, Petsc4py, Cysparse, and  Krypy. PyNumero also supports symmetric indefinite linear solvers that provide inertia information. Interfaces to PyMumps and the HSL linear solvers MA27 and MA57 are available within  PyNumero to solve sparse linear systems. These latter solvers are particularly important in constrained optimization as they provide critical information like the number of negative eigenvalues.

.. code-block:: python

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

.. note::

  The interaces to direct linear algebra solvers in PyNumero call Fortran and C code from python. Solving linear systems is often the dominant computational step in nonlinear optimization algorithms. For this reason it is key to solve the linear system in precompiled code. PyNumero achieves this by using Ctypes to call Fortran and C libraries that have state-of-the-art linear solver subroutins. Among them PyNumero supports MA27 and MA57 from the HSL and MUMPS.

  The direct linear solver interfaces in pynumero take an sparse matrix and right-hand-side vector and return a solution vector. The format of the sparse matrix can be a **scipy.sparse.spmatrix** or a **BlockMatrix**. Likewise, the right-hand-side vector can be a **numpy.ndarray** or a **BlockVector**.
