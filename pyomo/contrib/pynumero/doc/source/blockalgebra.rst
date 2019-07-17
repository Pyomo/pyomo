Block Algebra
=============

General nonlinear optimization algorithms often deal with block algebra operations. In equality constrained optimization, for instance, the KKT system consists of a :math:`2\times2` block matrix composed by the Jacobian of the constraints and the Hessian of the Lagrangian function, a solution vector composed by the primal and dual variables, and a right-hand-side vector composed by the gradient of the Lagrangian and the evaluation of the constraints. 

.. code-block:: python

	kkt = BlockMatrix(2, 2)
	kkt[0, 0] = hess_lag
	kkt[0, 1] = jac_g.transpose()
	kkt[1, 0] = jac_g
	print("KKT system\n", kkt.toarray())  # array([[-288.,    0., -144.,    0.,    8.],
		                              #        [   0.,    2.,    0.,    8.,    0.],
					      #        [-144.,    0.,  192.,    1.,    1.],
					      #        [   0.,    8.,    1.,    0.,    0.],
					      #        [   8.,    0.,    1.,    0.,    0.]]))

Our experience indicates that being able to represent and manipulate the KKT system using its block submatrices greatly simplifies the implementation of nonlinear optimization algorithms. In general, the numerical core of many NLP algorithms is a set of linear algebra operations dealing with block-structures (like those shown above). If a particular optimization problem has structure, then there is an inherent structure in the matrices and vectors involved in these operations (e.g. stochastic programming problems or time dependent optimization problems). Internal linear decomposition approaches, for instance, permit parallel solution of an NLP by exploiting this structure in all the scale-dependent linear algebra operations of a particular host algorithm. These techniques have been shown to be very efficient on many problems. However, they are typically very difficult to implement, and require strong programming expertise in low-level computer languages. A major contribution of PyNumero is towards the development of a flexible and efficient framework for developing these type of approaches by providing block algebra classes that leverages block-structures. We concentrate on the development of an effective BlockVector and BlockMatrix classes, that can be manipulated from Python and that compute expensive operations in C/C++. This strategy is becoming popular in Python. In PyNumero relies on the fact that it offers 

PyNumero subclasses the ndarray object in Numpy and sparse matrix objects from Scipy to facilitate block-matrix-vector operations
