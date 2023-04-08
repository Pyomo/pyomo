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

from pyomo.contrib.incidence_analysis.matching import maximum_matching
from pyomo.common.dependencies import scipy, scipy_available, networkx_available

if scipy_available:
    sps = scipy.sparse

import pyomo.common.unittest as unittest


@unittest.skipUnless(networkx_available, "networkx is not available")
@unittest.skipUnless(scipy_available, "scipy is not available")
class TestMatching(unittest.TestCase):
    def test_identity(self):
        N = 5
        matrix = sps.identity(N)
        matching = maximum_matching(matrix)
        self.assertEqual(len(matching), N)
        for i in range(N):
            self.assertIn(i, matching)
            self.assertEqual(i, matching[i])

        matrix = matrix.tocoo()
        matching = maximum_matching(matrix)
        self.assertEqual(len(matching), N)
        for i in range(N):
            self.assertIn(i, matching)
            self.assertEqual(i, matching[i])

        matrix = matrix.tocsc()
        matching = maximum_matching(matrix)
        self.assertEqual(len(matching), N)
        for i in range(N):
            self.assertIn(i, matching)
            self.assertEqual(i, matching[i])

    def test_low_rank_diagonal(self):
        N = 5
        omit = N // 2
        row = [i for i in range(N) if i != omit]
        col = [j for j in range(N) if j != omit]
        data = [1 for _ in range(N - 1)]
        matrix = sps.coo_matrix((data, (row, col)), shape=(N, N))
        matching = maximum_matching(matrix)

        self.assertEqual(len(matching), N - 1)
        for i in range(N):
            if i != omit:
                self.assertIn(i, matching)
                self.assertEqual(i, matching[i])

    def test_bordered(self):
        N = 5
        row = []
        col = []
        data = []
        for i in range(N - 1):
            # Bottom row
            row.append(N - 1)
            col.append(i)
            data.append(1)

            # Right column
            row.append(i)
            col.append(N - 1)
            data.append(1)

            # Diagonal
            row.append(i)
            col.append(i)
            data.append(1)

        matrix = sps.coo_matrix((data, (row, col)), shape=(N, N))
        matching = maximum_matching(matrix)

        self.assertEqual(len(matching), N)
        values = set(matching.values())
        for i in range(N):
            self.assertIn(i, matching)
            self.assertIn(i, values)

    def test_hessenberg(self):
        """
        |x x      |
        |    x    |
        |      x  |
        |        x|
        |x x x x x|
        """
        N = 5
        row = []
        col = []
        data = []
        for i in range(N):
            # Bottom row
            row.append(N - 1)
            col.append(i)
            data.append(1)

            if i == 0:
                # Top left entry
                row.append(0)
                col.append(i)
                data.append(1)
            else:
                # One-off diagonal
                row.append(i - 1)
                col.append(i)
                data.append(1)

        matrix = sps.coo_matrix((data, (row, col)), shape=(N, N))
        matching = maximum_matching(matrix)

        self.assertEqual(len(matching), N)
        values = set(matching.values())
        for i in range(N):
            self.assertIn(i, matching)
            self.assertIn(i, values)

    def test_low_rank_hessenberg(self):
        """
        |x x      |
        |         |
        |      x  |
        |        x|
        |x x x x x|
        Know that first and last row and column will be in
        the imperfect matching.
        """
        N = 5
        omit = N // 2
        row = []
        col = []
        data = []
        for i in range(N):
            # Bottom row
            row.append(N - 1)
            col.append(i)
            data.append(1)

            if i == 0:
                # Top left entry
                row.append(0)
                col.append(i)
                data.append(1)
            else:
                # One-off diagonal
                if i != omit:
                    row.append(i - 1)
                    col.append(i)
                    data.append(1)

        matrix = sps.coo_matrix((data, (row, col)), shape=(N, N))
        matching = maximum_matching(matrix)
        values = set(matching.values())

        self.assertEqual(len(matching), N - 1)
        self.assertIn(0, matching)
        self.assertIn(N - 1, matching)
        self.assertIn(0, values)
        self.assertIn(N - 1, values)

    def test_nondecomposable_hessenberg(self):
        """
        |x x      |
        |  x x    |
        |    x x  |
        |      x x|
        |x x x x x|
        """
        N = 5
        row = []
        col = []
        data = []
        for i in range(N):
            # Bottom row
            row.append(N - 1)
            col.append(i)
            data.append(1)

            # Diagonal
            row.append(i)
            col.append(i)
            data.append(1)
            # ^ This will put another entry at (N-1, N-1).
            # This is fine.

            if i != 0:
                # One-off diagonal
                row.append(i - 1)
                col.append(i)
                data.append(1)

        matrix = sps.coo_matrix((data, (row, col)), shape=(N, N))
        matching = maximum_matching(matrix)
        values = set(matching.values())

        self.assertEqual(len(matching), N)
        for i in range(N):
            self.assertIn(i, matching)
            self.assertIn(i, values)

    def test_low_rank_nondecomposable_hessenberg(self):
        """
        |  x      |
        |x   x    |
        |  x   x  |
        |    x   x|
        |      x  |
        """
        N = 5
        # For N odd, a matrix with this structure does not have
        # a perfect matching.
        row = []
        col = []
        data = []
        for i in range(N - 1):
            # Below diagonal
            row.append(i + 1)
            col.append(i)
            data.append(1)

            # Above diagonal
            row.append(i)
            col.append(i + 1)
            data.append(1)

        matrix = sps.coo_matrix((data, (row, col)), shape=(N, N))
        matching = maximum_matching(matrix)
        values = set(matching.values())

        self.assertEqual(len(matching), N - 1)
        self.assertEqual(len(values), N - 1)


if __name__ == "__main__":
    unittest.main()
