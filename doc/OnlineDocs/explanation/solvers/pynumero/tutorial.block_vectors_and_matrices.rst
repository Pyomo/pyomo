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

.. doctest::
   :skipif: not numpy_available or not scipy_available

   >>> import numpy as np
   >>> from scipy.sparse import coo_matrix
   >>> from pyomo.contrib.pynumero.sparse import BlockVector, BlockMatrix
   >>> v = BlockVector(3)
   >>> m = BlockMatrix(3, 3)

Setting blocks:

.. doctest::
   :skipif: not numpy_available or not scipy_available

   >>> v.set_block(0, np.array([-0.67025575, -1.2]))
   >>> v.set_block(1, np.array([0.1, 1.14872127]))
   >>> v.set_block(2, np.array([1.25]))
   >>> v.flatten()
   array([-0.67025575, -1.2       ,  0.1       ,  1.14872127,  1.25      ])

The `flatten` method converts the BlockVector into a NumPy array.

.. doctest::
   :skipif: not numpy_available or not scipy_available

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

.. doctest::
   :skipif: not numpy_available or not scipy_available

   >>> v.set_block(0, np.ones(3))
   Traceback (most recent call last):
     ...
   ValueError: Incompatible dimensions for block 0; got 3; expected 2

Properties:

.. doctest::
   :skipif: not scipy_available or int(np.__version__[0]) >= 2

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

.. doctest::
   :hide:
   :skipif: not scipy_available or int(np.__version__[0]) < 2

   >>> v.shape
   (np.int64(5),)
   >>> v.size
   np.int64(5)
   >>> v.nblocks
   3
   >>> v.bshape
   (3,)
   >>> m.shape
   (np.int64(5), np.int64(5))
   >>> m.bshape
   (3, 3)
   >>> m.nnz
   12

Much of the `BlockVector` API matches that of NumPy arrays:

.. doctest::
   :skipif: not scipy_available or int(np.__version__[0]) >= 2

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

.. doctest::
   :hide:
   :skipif: not scipy_available or int(np.__version__[0]) < 2

   >>> v.sum()
   np.float64(0.62846552)
   >>> v.max()
   np.float64(1.25)
   >>> np.abs(v).flatten()
   array([0.67025575, 1.2       , 0.1       , 1.14872127, 1.25      ])
   >>> (2*v).flatten()
   array([-1.3405115 , -2.4       ,  0.2       ,  2.29744254,  2.5       ])
   >>> (v + v).flatten()
   array([-1.3405115 , -2.4       ,  0.2       ,  2.29744254,  2.5       ])
   >>> v.dot(v)
   np.float64(4.781303326558476)

Similarly, `BlockMatrix` behaves very similarly to SciPy sparse matrices:

.. doctest::
   :skipif: not numpy_available or not scipy_available

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

Accessing blocks

.. doctest::
   :skipif: not numpy_available or not scipy_available

   >>> v.get_block(1)
   array([0.1       , 1.14872127])
   >>> m.get_block(1, 0).toarray()
   array([[ 0.        ,  0.        ],
          [-1.64872127,  1.        ]])

Empty blocks in a `BlockMatrix` return `None`:

.. doctest::
   :skipif: not numpy_available or not scipy_available

   >>> print(m.get_block(1, 1))
   None

The dimensions of a blocks in a `BlockMatrix` can be set without setting a block:

.. doctest::
   :skipif: not numpy_available or not scipy_available

   >>> m2 = BlockMatrix(2, 2)
   >>> m2.set_row_size(0, 5)
   >>> m2.set_block(0, 0, m.get_block(0, 0))
   Traceback (most recent call last):
     ...
   ValueError: Incompatible row dimensions for row 0; got 2; expected 5.0

Note that operations on `BlockVector` and `BlockMatrix` cannot be performed until the dimensions are fully specified:

.. doctest::
   :skipif: not numpy_available or not scipy_available

   >>> v2 = BlockVector(3)
   >>> v + v2
   Traceback (most recent call last):
     ...
   NotFullyDefinedBlockVectorError: Operation not allowed with None blocks.
   >>> m2 = BlockMatrix(3, 3)
   >>> m2 * 2
   Traceback (most recent call last):
     ...
   NotFullyDefinedBlockMatrixError: Operation not allowed with None rows. Specify at least one block in every row

The `has_none` property can be used to see if a `BlockVector` is fully
specified. If `has_none` returns `True`, then there are `None` blocks,
and the `BlockVector` is not fully specified.

.. doctest::
   :skipif: not numpy_available or not scipy_available

   >>> v.has_none
   False
   >>> v2.has_none
   True

For `BlockMatrix`, use the `has_undefined_row_sizes()` and `has_undefined_col_sizes()` methods:

.. doctest::
   :skipif: not numpy_available or not scipy_available

   >>> m.has_undefined_row_sizes()
   False
   >>> m.has_undefined_col_sizes()
   False
   >>> m2.has_undefined_row_sizes()
   True
   >>> m2.has_undefined_col_sizes()
   True

To efficiently iterate over non-empty blocks in a `BlockMatrix`, use
the `get_block_mask()` method, which returns a 2-D array indicating
where the non-empty blocks are:

.. doctest::
   :skipif: not numpy_available or not scipy_available

   >>> m.get_block_mask(copy=False)
   array([[ True,  True,  True],
          [ True, False,  True],
          [ True,  True, False]])
   >>> for i, j in zip(*np.nonzero(m.get_block_mask(copy=False))):
   ...     assert m.get_block(i, j) is not None

Copying data:

.. doctest::
   :skipif: not numpy_available or not scipy_available

   >>> v2 = v.copy()
   >>> v2.flatten()
   array([-0.67025575, -1.2       ,  0.1       ,  1.14872127,  1.25      ])
   >>> v2 = v.copy_structure()
   >>> v2.block_sizes()  # doctest: +SKIP
   array([2, 2, 1])
   >>> v2.copyfrom(v)
   >>> v2.flatten()
   array([-0.67025575, -1.2       ,  0.1       ,  1.14872127,  1.25      ])
   >>> m2 = m.copy()
   >>> (m - m2).tocoo().toarray()
   array([[0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0.]])
   >>> m2 = m.copy_structure()
   >>> m2.has_undefined_row_sizes()
   False
   >>> m2.has_undefined_col_sizes()
   False
   >>> m2.copyfrom(m)
   >>> (m - m2).tocoo().toarray()
   array([[0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0.]])

Nested blocks:

.. doctest::
   :skipif: not numpy_available or not scipy_available

   >>> v2 = BlockVector(2)
   >>> v2.set_block(0, v)
   >>> v2.set_block(1, np.ones(2))
   >>> v2.block_sizes()  # doctest: +SKIP
   array([5, 2])
   >>> v2.flatten()
   array([-0.67025575, -1.2       ,  0.1       ,  1.14872127,  1.25      ,
           1.        ,  1.        ])
   >>> v3 = v2.copy_structure()
   >>> v3.fill(1)
   >>> (v2 + v3).flatten()
   array([ 0.32974425, -0.2       ,  1.1       ,  2.14872127,  2.25      ,
           2.        ,  2.        ])
   >>> np.abs(v2).flatten()
   array([0.67025575, 1.2       , 0.1       , 1.14872127, 1.25      ,
          1.        , 1.        ])
   >>> v2.get_block(0)
   BlockVector(3,)

Nested `BlockMatrix` applications work similarly.

For more information, see the :mod:`API documentation <pyomo.contrib.pynumero>`.
