#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.config import (
    ConfigValue,
    document_kwargs_from_configdict
)
from pyomo.common.dependencies import numpy as np
from pyomo.common.gc_manager import PauseGC

from pyomo.opt import WriterFactory
from pyomo.repn.parameterized_linear import ParameterizedLinearRepnVisitor
from pyomo.repn.plugins.standard_form import (
    LinearStandardFormInfo,
    LinearStandardFormCompiler,
    _LinearStandardFormCompiler_impl
)
from pyomo.util.var_list_domain import var_component_set


@WriterFactory.register(
    'parameterized_standard_form_compiler',
    'Compile an LP to standard form (`min cTx s.t. Ax <= b`) treating some '
    'variables as data (e.g., variables decided by the outer problem in a '
    'bilevel optimization problem).'
)
class ParameterizedLinearStandardFormCompiler(LinearStandardFormCompiler):
    CONFIG = LinearStandardFormCompiler.CONFIG()
    CONFIG.declare(
        'wrt',
        ConfigValue(
            default=None,
            domain=var_component_set,
            description="Vars to treat as data for the purposes of compiling"
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
        """Convert a model to standard form (`min cTx s.t. Ax <= b`) treating the
        Vars specified in 'wrt' as data

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


class _ParameterizedLinearStandardFormCompiler_impl(_LinearStandardFormCompiler_impl):
    def _get_visitor(self, var_map, var_order, sorter):
        wrt = self.config.wrt
        if wrt is None:
            wrt = []
        return ParameterizedLinearRepnVisitor({}, var_map, var_order, sorter, wrt=wrt)

    def _get_data_list(self, linear_repn):
        # override this to not attempt conversion to float since that will fail
        # on the Pyomo expressions
        return [v for v in linear_repn.values()]

    def _compile_matrix(self, data, index, index_ptr, nrows, ncols):
        return _CSRMatrix(data, index, index_ptr, nrows, ncols).tocsc()


class _CSRMatrix(object):
    def __init__(self, data, col_index, row_index_ptr, nrows, ncols):
        # We store the indices and index pointers as numpy arrays for the sake
        # of duck typing, but not the data because that can contain Pyomo
        # expressions, so we just use a list.
        self.data = data
        self.indices = np.array(col_index)
        self.indptr = np.array(row_index_ptr)
        self.shape = (nrows, ncols)

    def tocsc(self):
        # Implements the same algorithm as scipy's csr_tocsc function
        csr_data = self.data
        col_index = self.indices
        row_index_ptr = self.indptr
        nrows = self.shape[0]

        num_nonzeros = len(csr_data)
        csc_data = [None for x in csr_data]
        row_index = np.empty(num_nonzeros)#[None for i in range(num_nonzeros)]
        # tally the nonzeros in each column
        col_index_ptr = np.zeros(num_nonzeros)#[0 for i in range(num_nonzeros)]
        for i in col_index:
            col_index_ptr[int(i)] += 1

        # cumulative sum the tally to get the column index pointer
        cum_sum = 0
        for i, tally in enumerate(col_index_ptr):
            col_index_ptr[i] = cum_sum
            cum_sum += tally
        # we leave off the last entry (the number of nonzeros) because we are
        # going to do the cute scipy thing and it will get 'added' at the end
        # when we shift right (by which I mean it will conveniently already be
        # there)

        # Now we are actually going to mess up what we just did while we
        # construct the row index, but it's beautiful because "mess up" just
        # means we increment everything by one, so we can fix it at the end.
        for row in range(nrows):
            for j in range(row_index_ptr[row], row_index_ptr[row + 1]):
                col = int(col_index[j])
                dest = int(col_index_ptr[col])
                row_index[dest] = row
                # Note that the data changes order because now we are looking
                # for nonzeros through the columns rather than through the rows.
                csc_data[dest] = csr_data[j]

                col_index_ptr[col] += 1
        
        # fix the column index pointer by inserting 0 at the beginning--the rest
        # is right because we know each entry got incremented by 1 in the loop
        # above.
        col_index_ptr = np.insert(col_index_ptr, 0, 0)

        return _CSCMatrix(csc_data, row_index, col_index_ptr, *self.shape)


class _CSCMatrix(object):
    def __init__(self, data, row_index, col_index_ptr, nrows, ncols):
        # We store the indices and index pointers as numpy arrays for the sake
        # of duck typing, but not the data because that can contain Pyomo
        # expressions, so we just use a list.
        self.data = data
        self.indices = np.array(row_index)
        self.indptr = np.array(col_index_ptr)
        self.shape = (nrows, ncols)


if __name__ == '__main__':
    A = _CSRMatrix([5, 8, 3, 6], [0, 1, 2, 1], [0, 1, 2, 3, 4], 4, 4)

    thing = A.tocsc()

    print(thing.data)
    print(thing.indices)
    print(thing.indptr)
