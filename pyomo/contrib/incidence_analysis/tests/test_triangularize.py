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

import random
from pyomo.contrib.incidence_analysis.matching import maximum_matching
from pyomo.contrib.incidence_analysis.triangularize import (
    get_scc_of_projection,
    block_triangularize,
    map_coords_to_block_triangular_indices,
    get_diagonal_blocks,
)
from pyomo.common.dependencies import (
    scipy,
    scipy_available,
    networkx as nx,
    networkx_available,
)

if scipy_available:
    sps = scipy.sparse
if networkx_available:
    nxb = nx.algorithms.bipartite

import pyomo.common.unittest as unittest


@unittest.skipUnless(networkx_available, "networkx is not available")
@unittest.skipUnless(scipy_available, "scipy is not available")
class TestGetSCCOfProjection(unittest.TestCase):
    def test_graph_decomposable_tridiagonal_shuffled(self):
        """
        This is the same graph as in test_decomposable_tridiagonal_shuffled
        below, but now we convert the matrix into a bipartite graph and
        use get_scc_of_projection.

        The matrix decomposes into 2x2 blocks:
        |x x      |
        |x x      |
        |  x x x  |
        |    x x  |
        |      x x|
        """
        N = 11
        row = []
        col = []
        data = []

        # Diagonal
        row.extend(range(N))
        col.extend(range(N))
        data.extend(1 for _ in range(N))

        # Below diagonal
        row.extend(range(1, N))
        col.extend(range(N - 1))
        data.extend(1 for _ in range(N - 1))

        # Above diagonal
        row.extend(i for i in range(N - 1) if not i % 2)
        col.extend(i + 1 for i in range(N - 1) if not i % 2)
        data.extend(1 for i in range(N - 1) if not i % 2)

        # Same results hold after applying a random permutation.
        row_perm = list(range(N))
        col_perm = list(range(N))
        random.shuffle(row_perm)
        random.shuffle(col_perm)

        row = [row_perm[i] for i in row]
        col = [col_perm[j] for j in col]

        matrix = sps.coo_matrix((data, (row, col)), shape=(N, N))
        graph = nxb.matrix.from_biadjacency_matrix(matrix)
        row_nodes = list(range(N))
        sccs = get_scc_of_projection(graph, row_nodes)

        self.assertEqual(len(sccs), (N + 1) // 2)

        for i in range((N + 1) // 2):
            # Note that these rows and cols are in the permuted space
            rows = set(r for r, _ in sccs[i])
            cols = set(c - N for _, c in sccs[i])

            pred_rows = {row_perm[2 * i]}
            pred_cols = {col_perm[2 * i]}

            if 2 * i + 1 < N:
                pred_rows.add(row_perm[2 * i + 1])
                pred_cols.add(col_perm[2 * i + 1])

            self.assertEqual(pred_rows, rows)
            self.assertEqual(pred_cols, cols)

    def test_scc_exceptions(self):
        graph = nx.Graph()
        graph.add_nodes_from(range(3))
        graph.add_edges_from([(0, 1), (0, 2), (1, 2)])
        top_nodes = [0]
        msg = "graph is not bipartite"
        with self.assertRaisesRegex(RuntimeError, msg):
            sccs = get_scc_of_projection(graph, top_nodes=top_nodes)

        graph = nx.Graph()
        graph.add_nodes_from(range(3))
        graph.add_edges_from([(0, 1), (0, 2)])
        top_nodes[0]
        msg = "bipartite sets of different cardinalities"
        with self.assertRaisesRegex(RuntimeError, msg):
            sccs = get_scc_of_projection(graph, top_nodes=top_nodes)

        graph = nx.Graph()
        graph.add_nodes_from(range(4))
        graph.add_edges_from([(0, 1), (0, 2)])
        top_nodes = [0, 3]
        msg = "without a perfect matching"
        with self.assertRaisesRegex(RuntimeError, msg):
            sccs = get_scc_of_projection(graph, top_nodes=top_nodes)


@unittest.skipUnless(networkx_available, "networkx is not available")
@unittest.skipUnless(scipy_available, "scipy is not available")
class TestTriangularize(unittest.TestCase):
    def test_low_rank_exception(self):
        N = 5
        row = list(range(N - 1))
        col = list(range(N - 1))
        data = [1 for _ in range(N - 1)]

        matrix = sps.coo_matrix((data, (row, col)), shape=(N, N))

        with self.assertRaises(RuntimeError) as exc:
            row_block_map, col_block_map = block_triangularize(matrix)
        self.assertIn('perfect matching', str(exc.exception))

    def test_non_square_exception(self):
        N = 5
        row = list(range(N - 1))
        col = list(range(N - 1))
        data = [1 for _ in range(N - 1)]

        matrix = sps.coo_matrix((data, (row, col)), shape=(N, N - 1))

        with self.assertRaises(ValueError) as exc:
            row_block_map, col_block_map = block_triangularize(matrix)
        self.assertIn('non-square matrices', str(exc.exception))

    def test_identity(self):
        N = 5
        matrix = sps.identity(N).tocoo()
        row_block_map, col_block_map = map_coords_to_block_triangular_indices(matrix)
        row_values = set(row_block_map.values())
        col_values = set(row_block_map.values())

        # For a (block) diagonal matrix, the order of diagonal
        # blocks is arbitrary, so we can't perform any strong
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
        col.extend(range(N - 1))
        data.extend(1 for _ in range(N - 1))

        matrix = sps.coo_matrix((data, (row, col)), shape=(N, N))

        row_block_map, col_block_map = map_coords_to_block_triangular_indices(matrix)
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
        row.extend(range(N - 1))
        col.extend(range(1, N))
        data.extend(1 for _ in range(N - 1))

        matrix = sps.coo_matrix((data, (row, col)), shape=(N, N))

        row_block_map, col_block_map = map_coords_to_block_triangular_indices(matrix)
        row_values = set(row_block_map.values())
        col_values = set(row_block_map.values())

        self.assertEqual(len(row_values), N)
        self.assertEqual(len(col_values), N)

        for i in range(N):
            # The block_triangularize function permutes
            # to lower triangular form, so rows and
            # columns are transposed to assemble the blocks.
            self.assertEqual(row_block_map[i], N - 1 - i)
            self.assertEqual(col_block_map[i], N - 1 - i)

    def test_bordered(self):
        """
        This matrix is non-decomposable
        |x       x|
        |  x     x|
        |    x   x|
        |      x x|
        |x x x x  |
        """
        N = 5
        row = []
        col = []
        data = []
        # Diagonal
        row.extend(range(N - 1))
        col.extend(range(N - 1))
        data.extend(1 for _ in range(N - 1))

        # Bottom row
        row.extend(N - 1 for _ in range(N - 1))
        col.extend(range(N - 1))
        data.extend(1 for _ in range(N - 1))

        # Right column
        row.extend(range(N - 1))
        col.extend(N - 1 for _ in range(N - 1))
        data.extend(1 for _ in range(N - 1))

        matrix = sps.coo_matrix((data, (row, col)), shape=(N, N))

        row_block_map, col_block_map = map_coords_to_block_triangular_indices(matrix)
        row_values = set(row_block_map.values())
        col_values = set(row_block_map.values())

        self.assertEqual(len(row_values), 1)
        self.assertEqual(len(col_values), 1)

        for i in range(N):
            self.assertEqual(row_block_map[i], 0)
            self.assertEqual(col_block_map[i], 0)

    def test_decomposable_bordered(self):
        """
        This matrix decomposes
        |x        |
        |  x      |
        |    x   x|
        |      x x|
        |x x x x  |
        """
        N = 5
        half = N // 2
        row = []
        col = []
        data = []

        # Diagonal
        row.extend(range(N - 1))
        col.extend(range(N - 1))
        data.extend(1 for _ in range(N - 1))

        # Bottom row
        row.extend(N - 1 for _ in range(N - 1))
        col.extend(range(N - 1))
        data.extend(1 for _ in range(N - 1))

        # Right column
        row.extend(range(half, N - 1))
        col.extend(N - 1 for _ in range(half, N - 1))
        data.extend(1 for _ in range(half, N - 1))

        matrix = sps.coo_matrix((data, (row, col)), shape=(N, N))

        row_block_map, col_block_map = map_coords_to_block_triangular_indices(matrix)
        row_values = set(row_block_map.values())
        col_values = set(row_block_map.values())

        self.assertEqual(len(row_values), half + 1)
        self.assertEqual(len(col_values), half + 1)

        first_half_set = set(range(half))
        for i in range(N):
            if i < half:
                # The first N//2 diagonal blocks are unordered
                self.assertIn(row_block_map[i], first_half_set)
                self.assertIn(col_block_map[i], first_half_set)
            else:
                self.assertEqual(row_block_map[i], half)
                self.assertEqual(col_block_map[i], half)

    def test_decomposable_tridiagonal(self):
        """
        This matrix decomposes into 2x2 blocks
        |x x      |
        |x x      |
        |  x x x  |
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
        col.extend(range(N - 1))
        data.extend(1 for _ in range(N - 1))

        # Above diagonal
        row.extend(i for i in range(N - 1) if not i % 2)
        col.extend(i + 1 for i in range(N - 1) if not i % 2)
        data.extend(1 for i in range(N - 1) if not i % 2)

        matrix = sps.coo_matrix((data, (row, col)), shape=(N, N))

        row_block_map, col_block_map = map_coords_to_block_triangular_indices(matrix)
        row_values = set(row_block_map.values())
        col_values = set(row_block_map.values())

        self.assertEqual(len(row_values), (N + 1) // 2)
        self.assertEqual(len(col_values), (N + 1) // 2)

        for i in range((N + 1) // 2):
            self.assertEqual(row_block_map[2 * i], i)
            self.assertEqual(col_block_map[2 * i], i)

            if 2 * i + 1 < N:
                self.assertEqual(row_block_map[2 * i + 1], i)
                self.assertEqual(col_block_map[2 * i + 1], i)

    def test_decomposable_tridiagonal_shuffled(self):
        """
        This matrix decomposes into 2x2 blocks
        |x x      |
        |x x      |
        |  x x x  |
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
        col.extend(range(N - 1))
        data.extend(1 for _ in range(N - 1))

        # Above diagonal
        row.extend(i for i in range(N - 1) if not i % 2)
        col.extend(i + 1 for i in range(N - 1) if not i % 2)
        data.extend(1 for i in range(N - 1) if not i % 2)

        # Same results hold after applying a random permutation.
        row_perm = list(range(N))
        col_perm = list(range(N))
        random.shuffle(row_perm)
        random.shuffle(col_perm)

        row = [row_perm[i] for i in row]
        col = [col_perm[j] for j in col]

        matrix = sps.coo_matrix((data, (row, col)), shape=(N, N))

        row_block_map, col_block_map = map_coords_to_block_triangular_indices(matrix)
        row_values = set(row_block_map.values())
        col_values = set(row_block_map.values())

        self.assertEqual(len(row_values), (N + 1) // 2)
        self.assertEqual(len(col_values), (N + 1) // 2)

        for i in range((N + 1) // 2):
            row_idx = row_perm[2 * i]
            col_idx = col_perm[2 * i]
            self.assertEqual(row_block_map[row_idx], i)
            self.assertEqual(col_block_map[col_idx], i)

            if 2 * i + 1 < N:
                row_idx = row_perm[2 * i + 1]
                col_idx = col_perm[2 * i + 1]
                self.assertEqual(row_block_map[row_idx], i)
                self.assertEqual(col_block_map[col_idx], i)

    def test_decomposable_tridiagonal_diagonal_blocks(self):
        """
        This matrix decomposes into 2x2 blocks
        |x x      |
        |x x      |
        |  x x x  |
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
        col.extend(range(N - 1))
        data.extend(1 for _ in range(N - 1))

        # Above diagonal
        row.extend(i for i in range(N - 1) if not i % 2)
        col.extend(i + 1 for i in range(N - 1) if not i % 2)
        data.extend(1 for i in range(N - 1) if not i % 2)

        matrix = sps.coo_matrix((data, (row, col)), shape=(N, N))

        row_blocks, col_blocks = get_diagonal_blocks(matrix)

        self.assertEqual(len(row_blocks), (N + 1) // 2)
        self.assertEqual(len(col_blocks), (N + 1) // 2)

        for i in range((N + 1) // 2):
            rows = row_blocks[i]
            cols = col_blocks[i]

            if 2 * i + 1 < N:
                self.assertEqual(set(rows), {2 * i, 2 * i + 1})
                self.assertEqual(set(cols), {2 * i, 2 * i + 1})
            else:
                self.assertEqual(set(rows), {2 * i})
                self.assertEqual(set(cols), {2 * i})


if __name__ == "__main__":
    unittest.main()
