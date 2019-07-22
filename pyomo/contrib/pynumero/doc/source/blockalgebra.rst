Block Algebra
=============

General nonlinear optimization algorithms often deal with block algebra operations. In equality constrained optimization, for instance, the KKT system consists of a :math:`2\times2` block-matrix and two :math:`2\times1` block-vectors. The block-matrix is composed by the Jacobian of the constraints and the Hessian of the Lagrangian function, while the solution vector is composed by the primal and dual variables and the right-hand-side vector by the gradient of the Lagrangian and the evaluation of the constraints. The implementation of such system in PyNumero is presented below

.. code-block:: python

	# Building the KKT matrix
	kkt = BlockMatrix(2, 2)
	kkt[0, 0] = hess_lag
	kkt[0, 1] = jac_g.transpose()
	kkt[1, 0] = jac_g

	# Building the RHS
	rhs = BlockVector(2)
	rhs[0] = nlp.grad_objective(x) + nlp.jacobian_g(x).T.dot(y)
	rhs[1] = nlp.evaluate_g(x)


Our experience indicates that being able to represent and manipulate the KKT system using its block submatrices greatly simplifies the implementation of nonlinear optimization algorithms. In general, the numerical core of many NLP algorithms is a set of linear algebra operations dealing with block-structures (like those shown above). If a particular optimization problem has structure, then there is an inherent structure in the matrices and vectors involved in these operations (e.g. stochastic programming problems or time dependent optimization problems). Internal linear decomposition approaches, for instance, permit parallel solution of an NLP by exploiting this structure in all the scale-dependent linear algebra operations of a particular host algorithm. These techniques have been shown to be very efficient on many problems. However, they are typically very difficult to implement, and require strong programming expertise in low-level computer languages. A major contribution of PyNumero is towards the development of a flexible and efficient framework for developing these type of approaches by providing block algebra classes that can leverage block-structures. We concentrate on the development of efficient implementations of **BlockVector** and **BlockMatrix** classes that can be manipulated from Python and that compute expensive operations in C/C++.

PyNumeros **BlockVector** behaves pretty much like a standard numpy array. In fact, **BlockVector** subclasses from the **numpy.ndarray** object. This has several benefits--- First, users familiar with Numpy/Scipy don't need to learn new syntax when working with our **BlockVector**. Second, since **BlockVector** belongs to the Numpy ecosystem (because of the inheritance), a number of numpy functions can be directly invoked on it. These are called **ufuncs** and a list of them together with a brief tutorial is presented in `here <https://docs.scipy.org/doc/numpy/reference/ufuncs.html>`_. Third, because the blocks within the **BlockVector** are numpy objects expensive operations are performed in compiled code. Additionally, subclassing from numpy allows us to further accelerate our code with just-in-time compiling tools like `Numba <https://numba.pydata.org/>`_. The following code demonstrates some features of PyNumeros **BlockVector** class

.. code-block:: python

	from pyomo.contrib.pynumero.sparse import BlockVector
	import numpy as np

	# create some arrays
	a = np.array([1.0, 2.0, 3.0])
	b = np.array([4.0, 5.0, 6.0])
	c = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

	# create one block vector with assignment of each block
	bv = BlockVector(2)
	bv[0] = a
	bv[1] = -b

	# get number of blocks
	nblocks = bv.nblocks

	# get overall size of the vector
	size = bv.size

	# Get unstructured vector
	flat_bv = bv.flatten()

	# create one block vector directly by passing a list of vectors
	bv2 = BlockVector([a, b])

	# compute absolute value
	res1 = np.absolute(bv)

	# compute reciprocal
	res2 = np.reciprocal(bv)

	# add two block vectors
	res3 = np.add(bv,bv2)
	res4 = np.add(bv, c)
	res5 = bv + c # this is the same to res4 and to c + bv

	# building block vectors from arrays or from other BlockVectors
	bv.copyfrom(c)
	bv.copyfrom(bv2)
	new_bv = bv.copy_structure()
	bv2.copyto(new_bv)

	# creating copies of block vectors
	new_bv = bv.copy()


.. note::
   The **flatten** function is very handy for returning an unstructured array (no blocks). This function basically concatenate all the blocks in the **BlockVector** and returns a **numpy.array**

   PyNumero supports operations between **numpy.arrays** and **BlockVectors**. Operations with Block-Objects preserve the block-structure. These operations may be used with caution to not affect performance. We encourage users to used them only for testing purpuses. To maximize performance, operations should be done between structured components.

   **BlockVector** is implemented using recursion. For this reason, Users may have BlockVectors within BlockVectors and everything will work as expected. This is very handy when working on optimization problems with structure. Consider for intance the solution vector of the KKT system descrived above for an stochastic problem. At the higher level the solution vector will have two blocks. In the first block there will be another **BlockVector** with primal variables of the scenarios and in the second block a **BlockVector** with the dual variables of the scenario constraints.

   There is a series of methods to easily copy **BlockVectors**. See **copyfrom**, **copyto**, **clone**, **copy_structure**

The **BlockMatrix** object behaves like **spmatrix** from Scipy. Similarly like the **BlockVector** follows the syntax of **numpy.ndarray**, the **BlockMatrix** uses the same syntax of **scipy.sparse**. Users can think of **BlockMatrix** as an object that stacks together submatrices. These submatrices can be objects from **scipy.sparse** (e.g. **coo_matrix**, **csr_matrix** and **csc_matrix**.) or other BlockMatrices. The following block of code demonstrates some functionality

.. code-block:: python

	from pyomo.contrib.pynumero.sparse import BlockMatrix
	from scipy.sparse import coo_matrix
	import numpy as np


	# create one block
	row = np.array([0, 1, 2, 3])
	col = np.array([0, 1, 0, 1])
	data = np.array([1., 1., 1., 1.])
	m1 = coo_matrix((data, (row, col)), shape=(4, 2))

	# create another block
	row = np.array([0, 3, 1, 2, 3, 0])
	col = np.array([0, 0, 1, 2, 3, 3])
	data = np.array([2., 1, 3, 4, 5, 1])
	m2 = coo_matrix((data, (row, col)), shape=(4, 4))

	# create block matrix
	bm = BlockMatrix(2,2)
	bm[0, 0] = m1
	bm[1, 1] = m2

	# get block shape of matrix
	nbrows, nbcols = bm.bshape

	# get number of nonzero values in matrix
	nnz = bm.nnz

	# get shape of matrix
	nrows, ncols = bm.shape

	# create scipy matrices from BlockMatrix
	coo_mat = bm.tocoo()
	csc_mat = bm.tocsc()
	csr_mat = bm.tocsr()

	# create numpy array from BlockMatrix
	dense_mat = bm.toarray()

	# transpose matrix
	bm_t = bm.T # or bm.transpose()

	# add and substract matrices
	res1_bm = bm + bm
	res2_bm = bm + coo_mat
	res3_bm = csr_mat - bm

	# matrix matrix product
	rr = bm * bm_t

	# get absolute value of a matrix
	abs_bm = abs(bm)

	# scale a matrix
	scaled_bm = 2.0 * bm

	# building BlockMatrices from spmatrices or other BlockMatrices
	copy_bm = bm.copy()
	new_bm = bm.copy_structure()
	new_bm.copyfrom(csr_mat)
	new_bm.copyfrom(scaled_bm)

.. note::

   The blocks in a **BlockMatrix** can be **scipy.sparse.spmatrix** or other **BlockMatrix** but not dense 2D **numpy.ndarray**.
   
   The methods **tocoo**, **tocsc**, **tocsr** return an unstructured copy of the **BlockMatrix** in the corresponding format. These methods can be handy when using Scipy methods on sparse matrices that have been built using **BlockMatrix**. They can be thought as equivalents of **BlockVector.flatten** in the sense that they return unstructured classes. Since the data of these matrices is stored in **numpy.ndarrays** (e.g. row, col, data), the transformation from a **BlockMatrix** to a **spmatrix** keeps the data in C as it only concatenates the numpy arrays from the blocks.

   The method **toarray** returns a dense representation of the **BlockMatrix** with the explicit zeros. This method can be handy when looking at the matrices. Users may also call **print** on a **BlockMatrix** for getting info regarding the structure of the matrix and the dimensions of the blocks.

   Operations with **BlockMatrix** behave the same as **spmatrix**. However, structure is always preserved. For example, adding a **BlockMatrix** with a **spmatrix** returns a **BlockMatrix**. These operations may be used with caution to not affect performance. We encourage users to used them only for testing purpuses. To maximize performance, operations should be done between structured components.

   There is a number of methods on **BlockMatrix** to facilitate copying and copying block matrices. See **copy**, **copyto**, **copyfrom**, **copy_structure**
