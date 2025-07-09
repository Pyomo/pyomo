Linear Solver Interfaces
========================

PyNumero's interfaces to linear solvers are very thin wrappers, and,
hence, are rather low-level. It is relatively easy to wrap these again
for specific applications. For example, see the linear solver
interfaces in
https://github.com/Pyomo/pyomo/tree/main/pyomo/contrib/interior_point/linalg,
which wrap PyNumero's linear solver interfaces.

The motivation to keep PyNumero's interfaces as such thin wrappers is
that different linear solvers serve different purposes. For example,
HSL's MA27 can factorize symmetric indefinite matrices, while MUMPS
can factorize unsymmetric, symmetric positive definite, or general
symmetric matrices. PyNumero seeks to be independent of the
application, giving more flexibility to algorithm developers.

Interface to MA27
-----------------

.. doctest::
   :skipif: not numpy_available or not scipy_available or not ma27_available

   >>> import numpy as np
   >>> from scipy.sparse import coo_matrix
   >>> from scipy.sparse import tril
   >>> from pyomo.contrib.pynumero.linalg.ma27_interface import MA27
   >>> row = np.array([0, 1, 0, 1, 0, 1, 2, 3, 3, 4, 4, 4])
   >>> col = np.array([0, 1, 3, 3, 4, 4, 4, 0, 1, 0, 1, 2])
   >>> data = np.array([1.67025575, 2, -1.64872127,  1, -1, -1, -1, -1.64872127, 1, -1, -1, -1])
   >>> A = coo_matrix((data, (row, col)), shape=(5,5))
   >>> A.toarray()
   array([[ 1.67025575,  0.        ,  0.        , -1.64872127, -1.        ],
          [ 0.        ,  2.        ,  0.        ,  1.        , -1.        ],
          [ 0.        ,  0.        ,  0.        ,  0.        , -1.        ],
          [-1.64872127,  1.        ,  0.        ,  0.        ,  0.        ],
          [-1.        , -1.        , -1.        ,  0.        ,  0.        ]])
   >>> rhs = np.array([-0.67025575, -1.2,  0.1,  1.14872127,  1.25])
   >>> solver = MA27()
   >>> solver.set_cntl(1, 1e-6)  # set the pivot tolerance
   >>> status = solver.do_symbolic_factorization(A)
   >>> status = solver.do_numeric_factorization(A)
   >>> x, status = solver.do_back_solve(rhs)
   >>> np.max(np.abs(A*x - rhs)) <= 1e-15
   np.True_


Interface to MUMPS
------------------

.. doctest::
   :skipif: not numpy_available or not scipy_available or not mumps_available

   >>> import numpy as np
   >>> from scipy.sparse import coo_matrix
   >>> from scipy.sparse import tril
   >>> from pyomo.contrib.pynumero.linalg.mumps_interface import MumpsCentralizedAssembledLinearSolver
   >>> row = np.array([0, 1, 0, 1, 0, 1, 2, 3, 3, 4, 4, 4])
   >>> col = np.array([0, 1, 3, 3, 4, 4, 4, 0, 1, 0, 1, 2])
   >>> data = np.array([1.67025575, 2, -1.64872127,  1, -1, -1, -1, -1.64872127, 1, -1, -1, -1])
   >>> A = coo_matrix((data, (row, col)), shape=(5,5))
   >>> A.toarray()
   array([[ 1.67025575,  0.        ,  0.        , -1.64872127, -1.        ],
          [ 0.        ,  2.        ,  0.        ,  1.        , -1.        ],
          [ 0.        ,  0.        ,  0.        ,  0.        , -1.        ],
          [-1.64872127,  1.        ,  0.        ,  0.        ,  0.        ],
          [-1.        , -1.        , -1.        ,  0.        ,  0.        ]])
   >>> rhs = np.array([-0.67025575, -1.2,  0.1,  1.14872127,  1.25])
   >>> solver = MumpsCentralizedAssembledLinearSolver(sym=2, par=1, comm=None)  # symmetric matrix; solve in serial
   >>> solver.do_symbolic_factorization(A)
   >>> solver.do_numeric_factorization(A)
   >>> x = solver.do_back_solve(rhs)
   >>> np.max(np.abs(A*x - rhs)) <= 1e-15
   True

Of course, SciPy solvers can also be used. See SciPy documentation for details.
