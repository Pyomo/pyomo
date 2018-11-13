#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
"""
Mimics scipy matrices to extract upper and lower triangular matrices

"""

from pyomo.contrib.pynumero.sparse.base import SparseBase


def tril(A, k=0, format=None):
    """Return the lower triangular portion of a matrix in sparse format

    Returns the elements on or below the k-th diagonal of the matrix A.
        - k = 0 corresponds to the main diagonal
        - k > 0 is above the main diagonal
        - k < 0 is below the main diagonal

    Parameters
    ----------
    A : dense or sparse matrix
        Matrix whose lower triangular portion is desired.
    k : integer : optional
        The top-most diagonal of the lower triangle.
    format : string
        Sparse format of the result, e.g. format="csr", etc.

    Returns
    -------
    L : sparse matrix
        Lower triangular portion of A in sparse format.

    See Also
    --------
    triu : upper triangle in sparse format

    Examples
    --------
    >>> from pyomo.contrib.pynumero.sparse import CSRMatrix, tril
    >>> A = csr_matrix([[1, 2, 0, 0, 3], [4, 5, 0, 6, 7], [0, 0, 8, 9, 0]],
    ...                dtype='int32')
    >>> A.toarray()
    array([[1, 2, 0, 0, 3],
           [4, 5, 0, 6, 7],
           [0, 0, 8, 9, 0]])
    >>> tril(A).toarray()
    array([[1, 0, 0, 0, 0],
           [4, 5, 0, 0, 0],
           [0, 0, 8, 0, 0]])
    >>> tril(A).nnz
    4
    >>> tril(A, k=1).toarray()
    array([[1, 2, 0, 0, 0],
           [4, 5, 0, 0, 0],
           [0, 0, 8, 9, 0]])
    >>> tril(A, k=-1).toarray()
    array([[0, 0, 0, 0, 0],
           [4, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]])
    >>> tril(A, format='csc')
    <3x5 sparse matrix of type '<class 'numpy.int32'>'
            with 4 stored elements in Compressed Sparse Column format>

    """
    from pyomo.contrib.pynumero.sparse import COOMatrix
    # convert to COOrdinate format where things are easy
    A = COOMatrix(A, copy=False)
    mask = A.row + k >= A.col
    return _masked_coo(A, mask).asformat(format)


def triu(A, k=0, format=None):
    """Return the upper triangular portion of a matrix in sparse format

    Returns the elements on or above the k-th diagonal of the matrix A.
        - k = 0 corresponds to the main diagonal
        - k > 0 is above the main diagonal
        - k < 0 is below the main diagonal

    Parameters
    ----------
    A : dense or sparse matrix
        Matrix whose upper triangular portion is desired.
    k : integer : optional
        The bottom-most diagonal of the upper triangle.
    format : string
        Sparse format of the result, e.g. format="csr", etc.

    Returns
    -------
    L : sparse matrix
        Upper triangular portion of A in sparse format.

    See Also
    --------
    tril : lower triangle in sparse format

    Examples
    --------
    >>> from pyomo.contrib.pynumero.sparse import CSRMatrix, triu
    >>> A = CSRMatrix([[1, 2, 0, 0, 3], [4, 5, 0, 6, 7], [0, 0, 8, 9, 0]],
    ...                dtype='int32')
    >>> A.toarray()
    array([[1, 2, 0, 0, 3],
           [4, 5, 0, 6, 7],
           [0, 0, 8, 9, 0]])
    >>> triu(A).toarray()
    array([[1, 2, 0, 0, 3],
           [0, 5, 0, 6, 7],
           [0, 0, 8, 9, 0]])
    >>> triu(A).nnz
    8
    >>> triu(A, k=1).toarray()
    array([[0, 2, 0, 0, 3],
           [0, 0, 0, 6, 7],
           [0, 0, 0, 9, 0]])
    >>> triu(A, k=-1).toarray()
    array([[1, 2, 0, 0, 3],
           [4, 5, 0, 6, 7],
           [0, 0, 8, 9, 0]])

    """
    from pyomo.contrib.pynumero.sparse import COOMatrix
    # convert to COOrdinate format where things are easy
    if isinstance(A, SparseBase):
        if A.is_symmetric:
            A = A.tofullmatrix().tocoo()
        else:
            A = COOMatrix(A, copy=False)
    else:
        A = COOMatrix(A, copy=False)
    mask = A.row + k <= A.col
    return _masked_coo(A, mask).asformat(format)


def _masked_coo(mat, mask):
    from pyomo.contrib.pynumero.sparse import COOMatrix
    row = mat.row[mask]
    col = mat.col[mask]
    data = mat.data[mask]
    return COOMatrix((data, (row, col)), shape=mat.shape, dtype=mat.dtype)



