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

.. code-block:: python
		
  >>> import numpy as np
  >>> from scipy.sparse import coo_matrix
  >>> from scipy.sparse import tril
  >>> from pyomo.contrib.pynumero.linalg.ma27 import MA27Interface
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
  >>> solver = MA27Interface()
  >>> solver.set_cntl(1, 1e-6)  # set the pivot tolerance
  >>> A_tril = tril(A)  # extract lower triangular portion of A
  >>> status = solver.do_symbolic_factorization(dim=5, irn=A_tril.row, icn=A_tril.col)
  >>> status = solver.do_numeric_factorization(dim=5, irn=A_tril.row, icn=A_tril.col, entries=A_tril.data)
  >>> x = solver.do_backsolve(rhs)
  >>> A*x - rhs
  array([-3.33066907e-16,  2.22044605e-16,  0.00000000e+00,  2.22044605e-16,
          0.00000000e+00])


Interface to MUMPS
------------------

.. code-block:: python
		
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
  >>> A_tril = tril(A)  # extract lower triangular portion of A
  >>> solver.do_symbolic_factorization(A_tril)
  >>> solver.do_numeric_factorization(A_tril)
  >>> x = solver.do_back_solve(rhs)
  >>> A*x - rhs
  array([-3.33066907e-16,  0.00000000e+00,  2.77555756e-17,  2.22044605e-16,
          4.44089210e-16])

Of course, SciPy solvers can also be used. See SciPy documentation for details.
