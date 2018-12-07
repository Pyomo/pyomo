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
The pyomo.contrib.pynumero.sparse.coo module includes methods that extend
linear algebra operations in scipy.sparse. In particular pynumero
adds functionality for dealing efficiently with symmetric matrices

All classes in this module subclass from the corresponding scipy.sparse
class. Hence, scipy documentation is the same for the methods here, unless
explicitly stated.

.. rubric:: Contents

"""

from pyomo.contrib.pynumero.sparse.block_vector import BlockVector
from scipy.sparse import coo_matrix as scipy_coo_matrix
from scipy.sparse._sparsetools import coo_tocsr, coo_todense
from scipy.sparse.sputils import (upcast,
                                  isdense,
                                  isscalarlike,
                                  get_index_dtype)
from scipy.sparse import issparse

from pyomo.contrib.pynumero.sparse.utils import is_symmetric_dense

try:
    from pyomo.contrib.pynumero.extensions.sparseutils import sym_coo_matvec
except ImportError as e:
    print('{}'.format(e))
    raise ImportError('Error importing sparseutils while running coo interface. '
                      'Make sure libpynumero_SPARSE is installed and added to path.')
from pyomo.contrib.pynumero.sparse.base import SparseBase
import numpy as np


__all__ = ['empty_matrix',
           'diagonal_matrix']


# this mimics an empty matrix
class empty_matrix(scipy_coo_matrix):

    def __init__(self, nrows, ncols):

        """

        Parameters
        ----------
        nrows : int
            Number of rows of sparse matrix
        ncol : int
            Number of columns of sparse matrix
        """

        data = np.zeros(0)
        irows = np.zeros(0)
        jcols = np.zeros(0)
        arg1 = (data, (irows, jcols))
        super(empty_matrix, self).__init__(arg1, shape=(nrows, ncols), dtype=np.double, copy=False)


class diagonal_matrix(scipy_coo_matrix):

    def __init__(self, values, eliminate_zeros=False):
        """

        Parameters
        ----------
        values : array-like
            vector with diagonal values
        """
        data = np.array(values, dtype=np.double)
        nrowcols = len(data)
        if eliminate_zeros:
            irows = np.nonzero(data)[0]
            jcols = irows
            data = data[irows]
        else:
            irows = np.arange(0, nrowcols)
            jcols = np.arange(0, nrowcols)
        arg1 = (data, (irows, jcols))
        super(diagonal_matrix, self).__init__(arg1, shape=(nrowcols, nrowcols), dtype=np.double, copy=False)

    def __repr__(self):
        return 'diagonal_matrix{}'.format(self.shape)

    def inv(self):

        """
        Returns inverse of diagonal matrix

        Returns
        -------
        diagonal_matrix
        """
        data = 1.0 / self.data
        return diagonal_matrix(data)



