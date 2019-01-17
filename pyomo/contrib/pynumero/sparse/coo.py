#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from scipy.sparse import coo_matrix as scipy_coo_matrix

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



