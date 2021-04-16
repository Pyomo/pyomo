Block Vectors and Matrices
==========================

Block vectors and matrices
(:py:class:`~pyomo.contrib.pynumero.sparse.block_vector.BlockVector`
and
:py:class:`~pyomo.contrib.pynumero.sparse.block_matrix.BlockMatrix`)
provide a mechanism to perform linear algebra operations with very
structured matrices and vectors.

When a BlockVector or BlockMatrix is constructed, the number of blocks
must be specified.

.. code-block:: python

  >>> import numpy as np
  >>> from scipy.sparse import coo_matrix
  >>> from pyomo.contrib.pynumero.sparse import BlockVector, BlockMatrix
  >>> v = BlockVector(3)
  >>> m = BlockMatrix(3, 3)

Setting blocks:

.. code-block:: python

  >>> v.set_block(0, np.array([-0.67025575, -1.2]))
  >>> v.set_block(1, np.array([0.1, 1.14872127]))
  >>> v.set_block(2, np.array([1.25]))
  >>> v.flatten()
  array([-0.67025575, -1.2       ,  0.1       ,  1.14872127,  1.25      ])

The `flatten` method converts the BlockVector into a NumPy array.

.. code-block:: python

  >>> m.set_block(0, 0, coo_matrix(np.array([[1.67025575, 0], [0, 2]])))
  >>> m.set_block(0, 1, coo_matrix(np.array([[0, -1.64872127], [0, 1]])))
  >>> m.set_block(0, 2, coo_matrix(np.array([[-1.0], [-1]])))
  >>> m.set_block(1, 0, coo_matrix(np.array([[0, -1.64872127], [0, 1]])).transpose())
  >>> m.set_block(1, 2, coo_matrix(np.array([[-1.0], [0]])))
  >>> m.set_block(2, 0, coo_matrix(np.array([[-1.0], [-1]])).transpose())
  >>> m.set_block(2, 1, coo_matrix(np.array([[-1.0], [0]])).transpose())
  >>> m.tocoo().toarray()
  array([[ 1.67025575,  0.        ,  0.        , -1.64872127, -1.        ],
         [ 0.        ,  2.        ,  0.        ,  1.        , -1.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        , -1.        ],
         [-1.64872127,  1.        ,  0.        ,  0.        ,  0.        ],
         [-1.        , -1.        , -1.        ,  0.        ,  0.        ]])

The `tocoo` method converts the `BlockMatrix` to a SciPy sparse `coo_matrix`.

Once the dimensions of a block have been set, they cannot be changed:

.. code-block:: python

  >>> v.set_block(0, np.ones(3))
  ValueError: Incompatible dimensions for block 0; got 3; expected 2

Properties:

.. code-block:: python

  >>> v.shape
  (5,)
  >>> v.size
  5
  >>> v.nblocks
  3
  >>> v.bshape
  (3,)
  >>> m.shape
  (5, 5)
  >>> m.bshape
  (3, 3)
  >>> m.nnz
  12

Much of the `BlockVector` API matches that of NumPy arrays:

.. code-block:: python

  >>> v.sum()
  0.62846552
  >>> v.max()
  1.25
  >>> np.abs(v).flatten()
  array([0.67025575, 1.2       , 0.1       , 1.14872127, 1.25      ])
  >>> (2*v).flatten()
  array([-1.3405115 , -2.4       ,  0.2       ,  2.29744254,  2.5       ])
  >>> (v + v).flatten()
  array([-1.3405115 , -2.4       ,  0.2       ,  2.29744254,  2.5       ])
  >>> v.dot(v)
  4.781303326558476

Similarly, `BlockMatrix` behaves very similarly to SciPy sparse matrices:

.. code-block:: python

  >>> (2*m).tocoo().toarray()
  array([[ 3.3405115 ,  0.        ,  0.        , -3.29744254, -2.        ],
         [ 0.        ,  4.        ,  0.        ,  2.        , -2.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        , -2.        ],
         [-3.29744254,  2.        ,  0.        ,  0.        ,  0.        ],
         [-2.        , -2.        , -2.        ,  0.        ,  0.        ]])
  >>> (m - m).tocoo().toarray()
  array([[0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0.]])
  >>> m * v
  BlockVector(3,)
  >>> (m * v).flatten()
  array([-4.26341971, -2.50127873, -1.25      , -0.09493509,  1.77025575])

