#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import random
import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.common.dependencies import networkx_available
from pyomo.common.dependencies import scipy_available
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.contrib.incidence_analysis.interface import (
    get_structural_incidence_matrix,
)
from pyomo.contrib.incidence_analysis.connected import (
    get_independent_submatrices,
)
from pyomo.contrib.incidence_analysis.tests.models_for_testing import (
    make_gas_expansion_model,
    make_dynamic_model,
)

import pyomo.common.unittest as unittest

if scipy_available:
    import scipy as sp


@unittest.skipIf(not networkx_available, "NetworkX is not available")
@unittest.skipIf(not scipy_available, "SciPy is not available")
class TestIndependentSubmatrices(unittest.TestCase):

    def test_decomposable_matrix(self):
        """
        The following matrix decomposes into two independent diagonal
        blocks.
        | x         |
        | x x       |
        |     x x   |
        |       x x |
        |         x |
        """
        row = [0, 1, 1, 2, 2, 3, 3, 4]
        col = [0, 0, 1, 2, 3, 3, 4, 4]
        data = [1, 1, 1, 1, 1, 1, 1, 1]
        N = 5
        coo = sp.sparse.coo_matrix(
            (data, (row, col)),
            shape=(N, N),
        )
        row_blocks, col_blocks = get_independent_submatrices(coo)
        self.assertEqual(len(row_blocks), 2)
        self.assertEqual(len(col_blocks), 2)

        # One of the independent submatrices must be the first two rows/cols
        self.assertTrue(
            set(row_blocks[0]) == {0, 1} or set(row_blocks[1]) == {0, 1}
        )
        self.assertTrue(
            set(col_blocks[0]) == {0, 1} or set(col_blocks[1]) == {0, 1}
        )
        # The other independent submatrix must be last three rows/columns
        self.assertTrue(
            set(row_blocks[0]) == {2, 3, 4} or set(row_blocks[1]) == {2, 3, 4}
        )
        self.assertTrue(
            set(col_blocks[0]) == {2, 3, 4} or set(col_blocks[1]) == {2, 3, 4}
        )

    def test_decomposable_matrix_permuted(self):
        """
        Same matrix as above, but now permuted into a random order.
        """
        row = [0, 1, 1, 2, 2, 3, 3, 4]
        col = [0, 0, 1, 2, 3, 3, 4, 4]
        data = [1, 1, 1, 1, 1, 1, 1, 1]
        N = 5
        row_perm = list(range(N))
        col_perm = list(range(N))
        random.shuffle(row_perm)
        random.shuffle(col_perm)
        row = [row_perm[i] for i in row]
        col = [col_perm[i] for i in col]
        coo = sp.sparse.coo_matrix(
            (data, (row, col)),
            shape=(N, N),
        )
        row_blocks, col_blocks = get_independent_submatrices(coo)
        self.assertEqual(len(row_blocks), 2)
        self.assertEqual(len(col_blocks), 2)


if __name__ == "__main__":
    unittest.main()
