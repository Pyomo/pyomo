#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.config import ConfigValue, document_kwargs_from_configdict
from pyomo.common.dependencies import numpy as np
from pyomo.common.gc_manager import PauseGC
from pyomo.common.numeric_types import native_numeric_types
from pyomo.core import Var

from pyomo.opt import WriterFactory
from pyomo.repn.parameterized import ParameterizedLinearRepnVisitor
from pyomo.repn.plugins.standard_form import (
    LinearStandardFormInfo,
    LinearStandardFormCompiler,
    _LinearStandardFormCompiler_impl,
)
from pyomo.util.config_domains import ComponentDataSet


@WriterFactory.register(
    'compile_parameterized_standard_form',
    'Compile an LP to standard form (`min cTx s.t. Ax <= b`) treating some '
    'variables as data (e.g., variables decided by the outer problem in a '
    'bilevel optimization problem).',
)
class ParameterizedLinearStandardFormCompiler(LinearStandardFormCompiler):
    r"""Compiler to convert a "Parameterized" LP to the matrix representation
    of the standard form:

    .. math::

        \min\ & c^Tx \\
        s.t.\ & Ax \le b

    by treating the variables specified in the ``wrt`` list as data
    (constants).  The resulting compiled representation is returned as
    NumPy arrays and SciPy sparse matrices in a
    :py:class:`LinearStandardFormInfo` .

    """

    CONFIG = LinearStandardFormCompiler.CONFIG()
    CONFIG.declare(
        'wrt',
        ConfigValue(
            default=None,
            domain=ComponentDataSet(Var),
            description="Vars to treat as data for the purposes of compiling "
            "the standard form",
            doc="""
            Optional list of Vars to be treated as data while compiling the 
            standard form.

            For example, if this is the standard form of an inner problem in a
            multilevel optimization problem, then the outer problem's Vars would
            be specified in this list since they are not variables from the
            perspective of the inner problem.
            """,
        ),
    )

    @document_kwargs_from_configdict(CONFIG)
    def write(self, model, ostream=None, **options):
        r"""Convert a model to standard form treating the Vars specified in
        ``wrt`` as data.

        Returns
        -------
        LinearStandardFormInfo

        Parameters
        ----------
        model: ConcreteModel
            The concrete Pyomo model to write out.

        ostream: None
            This is provided for API compatibility with other writers
            and is ignored here.

        """
        config = self.config(options)

        # Pause the GC, as the walker that generates the compiled LP
        # representation generates (and disposes of) a large number of
        # small objects.
        with PauseGC():
            return _ParameterizedLinearStandardFormCompiler_impl(config).write(model)


class _SparseMatrixBase(object):
    def __init__(self, matrix_data, shape):
        (data, indices, indptr) = matrix_data
        (nrows, ncols) = shape

        self.data = np.array(data)
        self.indices = np.array(indices, dtype=int)
        self.indptr = np.array(indptr, dtype=int)
        self.shape = (nrows, ncols)

    def __eq__(self, other):
        return self.todense() == other


class _CSRMatrix(_SparseMatrixBase):
    def __init__(self, matrix_data, shape):
        super().__init__(matrix_data, shape)
        if len(self.indptr) != self.shape[0] + 1:
            raise ValueError(
                "Shape specifies the number of rows as %s but the index "
                "pointer has length %s. The index pointer must have length "
                "nrows + 1: Check the 'shape' and 'matrix_data' arguments."
                % (self.shape[0], len(self.indptr))
            )

    def tocsc(self):
        """Implements the same algorithm as scipy's csr_tocsc function from
        sparsetools.
        """
        csr_data = self.data
        col_index = self.indices
        row_index_ptr = self.indptr
        nrows = self.shape[0]

        num_nonzeros = len(csr_data)
        csc_data = np.empty(csr_data.shape[0], dtype=object)
        row_index = np.empty(num_nonzeros, dtype=int)
        # tally the nonzeros in each column
        col_index_ptr = np.zeros(self.shape[1], dtype=int)
        for i in col_index:
            col_index_ptr[int(i)] += 1

        # cumulative sum the tally to get the column index pointer
        cum_sum = 0
        for i, tally in enumerate(col_index_ptr):
            col_index_ptr[i] = cum_sum
            cum_sum += tally
        # We have now initialized the col_index_ptr to the *starting* position
        # of each column in the data vector.  Note that col_index_ptr is only
        # num_cols long: we have ignored the last entry in the standard CSC
        # col_index_ptr (the total number of nonzeros).  This will get resolved
        # below when we shift this vector by one position.

        # Now we are actually going to mess up what we just did while we
        # construct the row index: We can imagine that col_index_ptr holds the
        # position of the *next* nonzero in each column, so each time we move a
        # data element into a column we will increment that col_index_ptr by
        # one.  This is beautiful because by "messing up" the col_index_pointer,
        # we are just transforming the vector of *starting* indices for each
        # column to a vector of *ending* indices (actually 1 past the last
        # index) of each column. Thank you, scipy.
        for row in range(nrows):
            for j in range(row_index_ptr[row], row_index_ptr[row + 1]):
                col = col_index[j]
                dest = col_index_ptr[col]
                row_index[dest] = row
                # Note that the data changes order because now we are looking
                # for nonzeros through the columns rather than through the rows.
                csc_data[dest] = csr_data[j]

                col_index_ptr[col] += 1

        # Fix the column index pointer by inserting 0 at the beginning. The
        # col_index_ptr currently holds pointers to 1 past the last element of
        # each column, which is really the starting index for the next
        # column. Inserting the 0 (the starting index for the firsst column)
        # shifts everything by one column, "converting" the vector to the
        # starting indices of each column, and extending the vector length to
        # num_cols + 1 (as is expected by the CSC matrix).
        col_index_ptr = np.insert(col_index_ptr, 0, 0)

        return _CSCMatrix((csc_data, row_index, col_index_ptr), self.shape)

    def todense(self):
        """Implements the algorithm from scipy's csr_todense function
        in sparsetools.
        """
        nrows = self.shape[0]
        col_index = self.indices
        row_index_ptr = self.indptr
        data = self.data

        dense = np.zeros(self.shape, dtype=object)

        for row in range(nrows):
            for j in range(row_index_ptr[row], row_index_ptr[row + 1]):
                dense[row, col_index[j]] = data[j]

        return dense


class _CSCMatrix(_SparseMatrixBase):
    def __init__(self, matrix_data, shape):
        super().__init__(matrix_data, shape)
        if len(self.indptr) != self.shape[1] + 1:
            raise ValueError(
                "Shape specifies the number of columns as %s but the index "
                "pointer has length %s. The index pointer must have length "
                "ncols + 1: Check the 'shape' and 'matrix_data' arguments."
                % (self.shape[1], len(self.indptr))
            )

    def todense(self):
        """Implements the algorithm from scipy's csr_todense function
        in sparsetools.
        """
        ncols = self.shape[1]
        row_index = self.indices
        col_index_ptr = self.indptr
        data = self.data

        dense = np.zeros(self.shape, dtype=object)

        for col in range(ncols):
            for j in range(col_index_ptr[col], col_index_ptr[col + 1]):
                dense[row_index[j], col] = data[j]

        return dense

    def tocsr(self):
        """Implements the same algorithm as scipy's csr_tocsc function from
        sparsetools.
        """
        csc_data = self.data
        row_index = self.indices
        col_index_ptr = self.indptr
        ncols = self.shape[1]

        num_nonzeros = len(csc_data)
        csr_data = np.empty(csc_data.shape[0], dtype=object)
        col_index = np.empty(num_nonzeros, dtype=int)
        # tally the nonzeros in each column
        row_index_ptr = np.zeros(self.shape[0], dtype=int)
        for i in row_index:
            row_index_ptr[int(i)] += 1

        # cumulative sum the tally to get the column index pointer
        cum_sum = 0
        for i, tally in enumerate(row_index_ptr):
            row_index_ptr[i] = cum_sum
            cum_sum += tally
        # We have now initialized the row_index_ptr to the *starting* position
        # of each column in the data vector.  Note that row_index_ptr is only
        # num_rows long: we have ignored the last entry in the standard CSR
        # row_index_ptr (the total number of nonzeros).  This will get resolved
        # below when we shift this vector by one position.

        # Now we are actually going to mess up what we just did while we
        # construct the row index: We can imagine that row_index_ptr holds the
        # position of the *next* nonzero in each column, so each time we move a
        # data element into a column we will increment that row_index_ptr by
        # one.  This is beautiful because by "messing up" the row_index_pointer,
        # we are just transforming the vector of *starting* indices for each
        # column to a vector of *ending* indices (actually 1 past the last
        # index) of each column. Thank you, scipy.
        for col in range(ncols):
            for j in range(col_index_ptr[col], col_index_ptr[col + 1]):
                row = row_index[j]
                dest = row_index_ptr[row]
                col_index[dest] = col
                # Note that the data changes order because now we are looking
                # for nonzeros through the columns rather than through the rows.
                csr_data[dest] = csc_data[j]

                row_index_ptr[row] += 1

        # Fix the row index pointer by inserting 0 at the beginning. The
        # row_index_ptr currently holds pointers to 1 past the last element of
        # each row, which is really the starting index for the next
        # row. Inserting the 0 (the starting index for the first column)
        # shifts everything by one column, "converting" the vector to the
        # starting indices of each row, and extending the vector length to
        # num_rows + 1 (as is expected by the CSR matrix).
        row_index_ptr = np.insert(row_index_ptr, 0, 0)

        return _CSRMatrix((csr_data, col_index, row_index_ptr), self.shape)

    def sum_duplicates(self):
        """Implements the algorithm from scipy's csr_sum_duplicates function
        in sparsetools.

        Note that this only removes duplicates that are adjacent, so it will remove
        all duplicates if the incoming CSC matrix has sorted indices. (In particular
        this will be true if it was just converted from CSR).
        """
        ncols = self.shape[1]
        row_index = self.indices
        col_index_ptr = self.indptr
        data = self.data

        num_non_zeros = 0
        col_end = 0
        for i in range(ncols):
            jj = col_end
            col_end = col_index_ptr[i + 1]
            while jj < col_end:
                j = row_index[jj]
                x = data[jj]
                jj += 1
                while jj < col_end and row_index[jj] == j:
                    x += data[jj]
                    jj += 1
                row_index[num_non_zeros] = j
                data[num_non_zeros] = x
                num_non_zeros += 1
            col_index_ptr[i + 1] = num_non_zeros

        # [ESJ 11/11/24]: I'm not 100% sure how scipy handles this, but we need
        # to remove the "extra" entries from the data and row_index arrays.
        self.data = data[:num_non_zeros]
        self.row_index = row_index[:num_non_zeros]

    def eliminate_zeros(self):
        """Implements the algorithm from scipy's csr_eliminate_zeros function
        in sparsetools.
        """
        ncols = self.shape[1]
        row_index = self.indices
        col_index_ptr = self.indptr
        data = self.data

        num_non_zeros = 0
        col_end = 0
        for i in range(ncols):
            jj = col_end
            col_end = col_index_ptr[i + 1]
            while jj < col_end:
                j = row_index[jj]
                x = data[jj]
                if x.__class__ not in native_numeric_types or x != 0:
                    row_index[num_non_zeros] = j
                    data[num_non_zeros] = x
                    num_non_zeros += 1
                jj += 1
            col_index_ptr[i + 1] = num_non_zeros


class _ParameterizedLinearStandardFormCompiler_impl(_LinearStandardFormCompiler_impl):
    _csc_matrix = _CSCMatrix
    _csr_matrix = _CSRMatrix

    def _get_visitor(self, subexpression_cache, var_recorder):
        wrt = self.config.wrt
        if wrt is None:
            wrt = []
        return ParameterizedLinearRepnVisitor(
            subexpression_cache, wrt=wrt, var_recorder=var_recorder
        )

    def _to_vector(self, data, N, vector_type):
        # override this to not attempt conversion to float since that will fail
        # on the Pyomo expressions
        return np.array(list(data))
