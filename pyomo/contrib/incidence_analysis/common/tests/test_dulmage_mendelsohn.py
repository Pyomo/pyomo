import pyomo.common.unittest as unittest
import networkx as nx
from pyomo.contrib.incidence_analysis.common.dulmage_mendelsohn import (
        dulmage_mendelsohn,
        )


class TestPothenFanExample(unittest.TestCase):
    """
    Test on the first example graph from Pothen and Fan, 1990.
    """

    def _construct_graph(self):
        bg = nx.Graph()
        n_l = 12
        n_r = 11
        left_nodes = list(range(n_l))
        right_nodes = list(range(n_r))

        bg.add_nodes_from(left_nodes, bipartite=0)
        bg.add_nodes_from([n_l+i for i in right_nodes], bipartite=1)

        paper_edges = [
                (1, 1),
                (1, 2),
                (1, 3),
                (1, 4),
                (1, 5),
                (1, 6),
                (2, 4),
                (2, 5),
                (2, 7),
                (2, 8),
                (2, 10),
                (3, 1),
                (3, 3),
                (3, 5),
                (4, 6),
                (4, 7),
                (4, 11),
                (5, 6),
                (5, 7),
                (5, 9),
                (6, 8),
                (6, 9),
                (7, 8),
                (7, 9),
                (7, 10),
                (8, 10),
                (8, 11),
                (9, 11),
                (10, 10),
                (11, 10),
                (11, 11),
                (12, 11),
                ]

        edges = [(i-1, j-1+n_l) for i, j in paper_edges]
        bg.add_edges_from(edges)

        return bg

    def test_coarse_partition(self):
        bg = self._construct_graph()
        left_partition, right_partition = dulmage_mendelsohn(bg)

        # As partitioned in the paper

        # Potentially unmatched
        left_underconstrained = {7, 8, 9, 10, 11}
        # Matched with potentially unmatched left nodes
        right_overconstrained = {21, 22}

        right_underconstrained = {12, 13, 14, 15, 16}
        left_overconstrained = {0, 1, 2}

        left_square = {3, 4, 5, 6}
        right_square = {17, 18, 19, 20}

        nodes = left_partition[0] + left_partition[1]
        self.assertEqual(set(nodes), left_underconstrained)

        nodes = right_partition[2]
        self.assertEqual(set(nodes), right_overconstrained)

        nodes = right_partition[0] + right_partition[1]
        self.assertEqual(set(nodes), right_underconstrained)

        nodes = left_partition[2]
        self.assertEqual(set(nodes), left_overconstrained)

        nodes = left_partition[3]
        self.assertEqual(set(nodes), left_square)

        nodes = right_partition[3]
        self.assertEqual(set(nodes), right_square)


if __name__ == "__main__":
    unittest.main()
