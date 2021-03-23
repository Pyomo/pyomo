#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.contrib.matching.maximum_matching import maximum_matching
from pyomo.contrib.matching.block_triang import block_triangularize
# TODO: Check if scipy is available
import scipy.sparse as sps

import pyomo.common.unittest as unittest


class TestTriangularize(unittest.TestCase):
    def test_identity(self):
        N = 5
        matrix = sps.identity(N).tocoo()
        row_block_map, col_block_map = block_triangularize(matrix)
        row_values = set(row_block_map.values())
        col_values = set(row_block_map.values())

        # For a (block) diagonal matrix, the order of diagonal
        # blocks is arbitary, so we can't perform any strong
        # checks here.
        #
        # Perfect matching is unique, but order of strongly
        # connected components is not.

        self.assertEqual(len(row_block_map), N)
        self.assertEqual(len(col_block_map), N)
        self.assertEqual(len(row_values), N)
        self.assertEqual(len(col_values), N)

        for i in range(N):
            self.assertIn(i, row_block_map)
            self.assertIn(i, col_block_map)
            self.assertIn(i, row_values)
            self.assertIn(i, col_values)

    def test_lower_tri(self):
        """
        This matrix has a unique maximal matching and SCC
        order, making it a good test for a "fully decomposable"
        matrix.
        |x        |
        |x x      |
        |  x x    |
        |    x x  |
        |      x x|
        """
        N = 5
        row = []
        col = []
        data = []
        # Diagonal
        row.extend(range(N))
        col.extend(range(N))
        data.extend(1 for _ in range(N))

        # Below diagonal
        row.extend(range(1, N))
        col.extend(range(N-1))
        data.extend(1 for _ in range(N-1))

        matrix = sps.coo_matrix((data, (row, col)), shape=(N, N))

        row_block_map, col_block_map = block_triangularize(matrix)
        row_values = set(row_block_map.values())
        col_values = set(row_block_map.values())
        
        self.assertEqual(len(row_values), N)
        self.assertEqual(len(col_values), N)

        for i in range(N):
            self.assertEqual(row_block_map[i], i)
            self.assertEqual(col_block_map[i], i)

    def test_upper_tri(self):
        """
        This matrix has a unique maximal matching and SCC
        order, making it a good test for a "fully decomposable"
        matrix.
        |x x      |
        |  x x    |
        |    x x  |
        |      x x|
        |        x|
        """
        N = 5
        row = []
        col = []
        data = []
        # Diagonal
        row.extend(range(N))
        col.extend(range(N))
        data.extend(1 for _ in range(N))

        # Below diagonal
        row.extend(range(N-1))
        col.extend(range(1, N))
        data.extend(1 for _ in range(N-1))

        matrix = sps.coo_matrix((data, (row, col)), shape=(N, N))

        row_block_map, col_block_map = block_triangularize(matrix)
        row_values = set(row_block_map.values())
        col_values = set(row_block_map.values())
        
        self.assertEqual(len(row_values), N)
        self.assertEqual(len(col_values), N)

        for i in range(N):
            # The block_triangularize function permutes
            # to lower triangular form, so rows and
            # columns are transposed to assemble the blocks.
            self.assertEqual(row_block_map[i], N-1-i)
            self.assertEqual(col_block_map[i], N-1-i)

    # TODO:
    # - Test non-decomposable matrix
    # - Test partially decomposable matrices
    # - Test exceptions


if __name__ == "__main__":
    unittest.main()
