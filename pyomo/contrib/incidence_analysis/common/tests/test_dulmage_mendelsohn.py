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

import pyomo.common.unittest as unittest
from pyomo.common.dependencies import networkx as nx, networkx_available
from pyomo.contrib.incidence_analysis.common.dulmage_mendelsohn import (
    dulmage_mendelsohn,
)


@unittest.skipUnless(networkx_available, "networkx is not available")
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
        bg.add_nodes_from([n_l + i for i in right_nodes], bipartite=1)

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

        edges = [(i - 1, j - 1 + n_l) for i, j in paper_edges]
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


@unittest.skipUnless(networkx_available, "networkx is not available")
class TestDisconnectedModel(unittest.TestCase):
    """
    An error is raised if top_nodes are not provided for a disconnected
    graph, which we test here.
    """

    def _construct_graph(self):
        """
        Graph with the following incidence matrix:
        |x x         x|
        |x   x        |
        |      x   x  |
        |        x x  |
        |      x x    |
        |            x|
        |            x|
        """
        N = 7
        top_nodes = list(range(N))
        bot_nodes = list(range(N, 2 * N))

        graph = nx.Graph()
        graph.add_nodes_from(top_nodes, bipartite=0)
        graph.add_nodes_from(bot_nodes, bipartite=1)

        edges = [
            (0, 0),
            (0, 1),
            (0, 6),
            (1, 0),
            (1, 2),
            (2, 3),
            (2, 5),
            (3, 4),
            (3, 5),
            (4, 3),
            (4, 4),
            (5, 6),
            (6, 6),
        ]
        edges = [(i, j + N) for i, j in edges]
        graph.add_edges_from(edges)
        return graph, top_nodes

    def test_graph_dm_partition(self):
        graph, top_nodes = self._construct_graph()

        with self.assertRaises(nx.AmbiguousSolution):
            top_dmp, bot_dmp = dulmage_mendelsohn(graph)

        top_dmp, bot_dmp = dulmage_mendelsohn(graph, top_nodes=top_nodes)

        self.assertFalse(nx.is_connected(graph))

        underconstrained_top = {0, 1}
        underconstrained_bot = {7, 8, 9}
        self.assertEqual(underconstrained_top, set(top_dmp[2]))
        self.assertEqual(underconstrained_bot, set(bot_dmp[0] + bot_dmp[1]))

        overconstrained_top = {5, 6}
        overconstrained_bot = {13}
        self.assertEqual(overconstrained_top, set(top_dmp[0] + top_dmp[1]))
        self.assertEqual(overconstrained_bot, set(bot_dmp[2]))

        wellconstrained_top = {2, 3, 4}
        wellconstrained_bot = {10, 11, 12}
        self.assertEqual(wellconstrained_top, set(top_dmp[3]))
        self.assertEqual(wellconstrained_bot, set(bot_dmp[3]))


if __name__ == "__main__":
    unittest.main()
