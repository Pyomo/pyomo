"""Community Detection Test File"""

# Structure for this file was adapted from:
#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from __future__ import division

import logging

import pyutilib.th as unittest
from six import StringIO

from pyomo.common.dependencies import networkx_available
from pyomo.common.log import LoggingIntercept
from pyomo.environ import ConcreteModel, Constraint, Objective, Var, Integers, minimize, RangeSet, Block, ConstraintList
from pyomo.contrib.community_detection.detection import detect_communities, CommunityMap, \
    community_louvain_available, matplotlib_available, community_louvain, matplotlib
from pyomo.contrib.community_detection.community_graph import generate_model_graph

from pyomo.solvers.tests.models.LP_unbounded import LP_unbounded
from pyomo.solvers.tests.models.QP_simple import QP_simple
from pyomo.solvers.tests.models.LP_inactive_index import LP_inactive_index
from pyomo.solvers.tests.models.SOS1_simple import SOS1_simple


@unittest.skipUnless(community_louvain_available, "'community' package from 'python-louvain' is not available.")
@unittest.skipUnless(networkx_available, "networkx is not available.")
class TestDecomposition(unittest.TestCase):

    def test_communities_1(self):
        m_class = LP_inactive_index()
        m_class._generate_model()
        model = m = m_class.model

        test_community_maps, test_partitions = _collect_community_maps(model)

        correct_partitions = [{3: 0, 4: 0, 5: 1, 6: 1, 7: 1, 8: 0, 9: 2, 10: 2, 11: 2},
                              {3: 0, 4: 1, 5: 0, 6: 2, 7: 2, 8: 0, 9: 0, 10: 0, 11: 2, 12: 1, 13: 1, 14: 1},
                              {3: 0, 4: 0, 5: 1, 6: 1, 7: 1, 8: 0, 9: 2, 10: 2, 11: 2},
                              {3: 0, 4: 1, 5: 0, 6: 0, 7: 0, 8: 2, 9: 2, 10: 2, 11: 0, 12: 1, 13: 1, 14: 1},
                              {3: 0, 4: 1, 5: 1, 6: 0, 7: 2}, {3: 0, 4: 1, 5: 0, 6: 0, 7: 1, 8: 0},
                              {3: 0, 4: 1, 5: 1, 6: 0, 7: 2}, {3: 0, 4: 1, 5: 0, 6: 0, 7: 1, 8: 0},
                              {0: 0, 1: 1, 2: 2}, {0: 0, 1: 0, 2: 0}, {0: 0, 1: 1, 2: 2}, {0: 0, 1: 0, 2: 0},
                              {0: 0, 1: 1, 2: 2}, {0: 0, 1: 0, 2: 0}, {0: 0, 1: 1, 2: 2}, {0: 0, 1: 0, 2: 0},
                              {0: 0, 1: 1, 2: 2, 3: 0, 4: 0, 5: 1, 6: 1, 7: 1, 8: 0, 9: 2, 10: 2, 11: 2},
                              {0: 0, 1: 1, 2: 2, 3: 1, 4: 2, 5: 0, 6: 0, 7: 0, 8: 1, 9: 1, 10: 1, 11: 0,
                               12: 2, 13: 2, 14: 2},
                              {0: 0, 1: 1, 2: 2, 3: 0, 4: 0, 5: 1, 6: 1, 7: 1, 8: 0, 9: 2, 10: 2, 11: 2},
                              {0: 0, 1: 1, 2: 2, 3: 1, 4: 2, 5: 0, 6: 0, 7: 0, 8: 1, 9: 1, 10: 1, 11: 0,
                               12: 2, 13: 2, 14: 2},
                              {0: 0, 1: 1, 2: 2, 3: 0, 4: 1, 5: 1, 6: 0, 7: 2},
                              {0: 0, 1: 1, 2: 2, 3: 2, 4: 0, 5: 1, 6: 1, 7: 0, 8: 2},
                              {0: 0, 1: 1, 2: 2, 3: 0, 4: 1, 5: 1, 6: 0, 7: 2},
                              {0: 0, 1: 1, 2: 2, 3: 2, 4: 0, 5: 1, 6: 1, 7: 0, 8: 2}]

        if correct_partitions == test_partitions:
            # Convert test_community_maps to a string because memory locations may vary by OS and PC
            str_test_community_maps = [str(community_map) for community_map in test_community_maps]

            correct_community_maps = [
                "{0: (['c1[1]', 'c1[2]', 'c2[2]'], ['x']), 1: (['c1[3]', 'c1[4]', 'c2[1]'], ['y']), "
                "2: (['b.c', 'B[1].c', 'B[2].c'], ['z'])}",
                "{0: (['obj[1]', 'OBJ', 'c1[3]', 'c1[4]', 'c2[1]'], ['x', 'y']), "
                "1: (['obj[2]', 'b.c', 'B[1].c', 'B[2].c'], ['x', 'y', 'z']), 2: (['c1[1]', 'c1[2]', 'c2[2]'], ['x'])}",
                "{0: (['c1[1]', 'c1[2]', 'c2[2]'], ['x']), 1: (['c1[3]', 'c1[4]', 'c2[1]'], ['y']), "
                "2: (['b.c', 'B[1].c', 'B[2].c'], ['z'])}",
                "{0: (['obj[1]', 'OBJ', 'c1[1]', 'c1[2]', 'c2[2]'], ['x', 'y']), "
                "1: (['obj[2]', 'b.c', 'B[1].c', 'B[2].c'], ['x', 'y', 'z']), 2: (['c1[3]', 'c1[4]', 'c2[1]'], ['y'])}",
                "{0: (['c1[2]', 'c2[2]'], ['x']), 1: (['c1[3]', 'c2[1]'], ['y']), 2: (['B[2].c'], ['z'])}",
                "{0: (['obj[2]', 'c1[3]', 'c2[1]', 'B[2].c'], ['x', 'y', 'z']), 1: (['c1[2]', 'c2[2]'], ['x'])}",
                "{0: (['c1[2]', 'c2[2]'], ['x']), 1: (['c1[3]', 'c2[1]'], ['y']), 2: (['B[2].c'], ['z'])}",
                "{0: (['obj[2]', 'c1[3]', 'c2[1]', 'B[2].c'], ['x', 'y', 'z']), 1: (['c1[2]', 'c2[2]'], ['x'])}",
                "{0: (['c1[1]', 'c1[2]', 'c2[2]'], ['x']), 1: (['c1[3]', 'c1[4]', 'c2[1]'], ['y']), "
                "2: (['b.c', 'B[1].c', 'B[2].c'], ['z'])}",
                "{0: (['obj[1]', 'obj[2]', 'OBJ', 'c1[1]', 'c1[2]', 'c1[3]', 'c1[4]',"
                " 'c2[1]', 'c2[2]', 'b.c', 'B[1].c', 'B[2].c'], ['x', 'y', 'z'])}",
                "{0: (['c1[1]', 'c1[2]', 'c2[2]'], ['x']), 1: (['c1[3]', 'c1[4]', 'c2[1]'], ['y']), "
                "2: (['b.c', 'B[1].c', 'B[2].c'], ['z'])}",
                "{0: (['obj[1]', 'obj[2]', 'OBJ', 'c1[1]', 'c1[2]', 'c1[3]', 'c1[4]', 'c2[1]', 'c2[2]', 'b.c',"
                " 'B[1].c', 'B[2].c'], ['x', 'y', 'z'])}",
                "{0: (['c1[2]', 'c2[2]'], ['x']), 1: (['c1[3]', 'c2[1]'], ['y']), 2: (['B[2].c'], ['z'])}",
                "{0: (['obj[2]', 'c1[2]', 'c1[3]', 'c2[1]', 'c2[2]', 'B[2].c'], ['x', 'y', 'z'])}",
                "{0: (['c1[2]', 'c2[2]'], ['x']), 1: (['c1[3]', 'c2[1]'], ['y']), 2: (['B[2].c'], ['z'])}",
                "{0: (['obj[2]', 'c1[2]', 'c1[3]', 'c2[1]', 'c2[2]', 'B[2].c'], ['x', 'y', 'z'])}",
                "{0: (['c1[1]', 'c1[2]', 'c2[2]'], ['x']), 1: (['c1[3]', 'c1[4]', 'c2[1]'], ['y']), "
                "2: (['b.c', 'B[1].c', 'B[2].c'], ['z'])}",
                "{0: (['OBJ', 'c1[1]', 'c1[2]', 'c2[2]'], ['x']), 1: (['obj[1]', 'c1[3]', 'c1[4]', 'c2[1]'], ['y']), "
                "2: (['obj[2]', 'b.c', 'B[1].c', 'B[2].c'], ['z'])}",
                "{0: (['c1[1]', 'c1[2]', 'c2[2]'], ['x']), 1: (['c1[3]', 'c1[4]', 'c2[1]'], ['y']), "
                "2: (['b.c', 'B[1].c', 'B[2].c'], ['z'])}",
                "{0: (['OBJ', 'c1[1]', 'c1[2]', 'c2[2]'], ['x']), 1: (['obj[1]', 'c1[3]', 'c1[4]', 'c2[1]'], ['y']), "
                "2: (['obj[2]', 'b.c', 'B[1].c', 'B[2].c'], ['z'])}",
                "{0: (['c1[2]', 'c2[2]'], ['x']), 1: (['c1[3]', 'c2[1]'], ['y']), 2: (['B[2].c'], ['z'])}",
                "{0: (['c1[2]', 'c2[2]'], ['x']), 1: (['c1[3]', 'c2[1]'], ['y']), 2: (['obj[2]', 'B[2].c'], ['z'])}",
                "{0: (['c1[2]', 'c2[2]'], ['x']), 1: (['c1[3]', 'c2[1]'], ['y']), 2: (['B[2].c'], ['z'])}",
                "{0: (['c1[2]', 'c2[2]'], ['x']), 1: (['c1[3]', 'c2[1]'], ['y']), 2: (['obj[2]', 'B[2].c'], ['z'])}"]

            self.assertEqual(correct_community_maps, str_test_community_maps)

        # Partition-based diagnostic test
        correct_num_communities, correct_num_nodes, test_num_communities, test_num_nodes = \
            _collect_partition_dependent_tests(test_community_maps, test_partitions)

        self.assertEqual(correct_num_communities, test_num_communities)
        self.assertEqual(correct_num_nodes, test_num_nodes)

    def test_communities_2(self):
        m_class = QP_simple()
        m_class._generate_model()
        model = m = m_class.model

        test_community_maps, test_partitions = _collect_community_maps(model)

        correct_partitions = [{2: 0, 3: 0}, {2: 0, 3: 0, 4: 0, 5: 0}, {2: 0, 3: 0}, {2: 0, 3: 0, 4: 0, 5: 0},
                              {2: 0, 3: 0}, {2: 0, 3: 0, 4: 0}, {2: 0, 3: 0}, {2: 0, 3: 0, 4: 0}, {0: 0, 1: 0},
                              {0: 0, 1: 0}, {0: 0, 1: 0}, {0: 0, 1: 0}, {0: 0, 1: 0}, {0: 0, 1: 0}, {0: 0, 1: 0},
                              {0: 0, 1: 0}, {0: 0, 1: 1, 2: 1, 3: 0}, {0: 0, 1: 1, 2: 1, 3: 0, 4: 1, 5: 0},
                              {0: 0, 1: 1, 2: 1, 3: 0}, {0: 0, 1: 1, 2: 1, 3: 0, 4: 1, 5: 0}, {0: 0, 1: 1, 2: 1, 3: 0},
                              {0: 0, 1: 1, 2: 1, 3: 1, 4: 0}, {0: 0, 1: 1, 2: 1, 3: 0}, {0: 0, 1: 1, 2: 1, 3: 1, 4: 0}]

        if correct_partitions == test_partitions:
            # Convert test_community_maps to a string because memory locations may vary by OS and PC
            str_test_community_maps = [str(community_map) for community_map in test_community_maps]

            correct_community_maps = ["{0: (['c1', 'c2'], ['x', 'y'])}",
                                      "{0: (['inactive_obj', 'obj', 'c1', 'c2'], ['x', 'y'])}",
                                      "{0: (['c1', 'c2'], ['x', 'y'])}",
                                      "{0: (['inactive_obj', 'obj', 'c1', 'c2'], ['x', 'y'])}",
                                      "{0: (['c1', 'c2'], ['x', 'y'])}", "{0: (['obj', 'c1', 'c2'], ['x', 'y'])}",
                                      "{0: (['c1', 'c2'], ['x', 'y'])}", "{0: (['obj', 'c1', 'c2'], ['x', 'y'])}",
                                      "{0: (['c1', 'c2'], ['x', 'y'])}",
                                      "{0: (['inactive_obj', 'obj', 'c1', 'c2'], ['x', 'y'])}",
                                      "{0: (['c1', 'c2'], ['x', 'y'])}",
                                      "{0: (['inactive_obj', 'obj', 'c1', 'c2'], ['x', 'y'])}",
                                      "{0: (['c1', 'c2'], ['x', 'y'])}", "{0: (['obj', 'c1', 'c2'], ['x', 'y'])}",
                                      "{0: (['c1', 'c2'], ['x', 'y'])}", "{0: (['obj', 'c1', 'c2'], ['x', 'y'])}",
                                      "{0: (['c2'], ['x']), 1: (['c1'], ['y'])}",
                                      "{0: (['obj', 'c2'], ['x']), 1: (['inactive_obj', 'c1'], ['y'])}",
                                      "{0: (['c2'], ['x']), 1: (['c1'], ['y'])}",
                                      "{0: (['obj', 'c2'], ['x']), 1: (['inactive_obj', 'c1'], ['y'])}",
                                      "{0: (['c2'], ['x']), 1: (['c1'], ['y'])}",
                                      "{0: (['c2'], ['x']), 1: (['obj', 'c1'], ['y'])}",
                                      "{0: (['c2'], ['x']), 1: (['c1'], ['y'])}",
                                      "{0: (['c2'], ['x']), 1: (['obj', 'c1'], ['y'])}"]

            self.assertEqual(correct_community_maps, str_test_community_maps)

        # Partition-based diagnostic test
        correct_num_communities, correct_num_nodes, test_num_communities, test_num_nodes = \
            _collect_partition_dependent_tests(test_community_maps, test_partitions)

        self.assertEqual(correct_num_communities, test_num_communities)
        self.assertEqual(correct_num_nodes, test_num_nodes)

    def test_communities_3(self):
        m_class = LP_unbounded()
        m_class._generate_model()
        model = m = m_class.model

        test_community_maps, test_partitions = _collect_community_maps(model)

        correct_partitions = [{}, {2: 0}, {}, {2: 0}, {}, {2: 0}, {}, {2: 0}, {0: 0, 1: 1}, {0: 0, 1: 0}, {0: 0, 1: 1},
                              {0: 0, 1: 0}, {0: 0, 1: 1}, {0: 0, 1: 0}, {0: 0, 1: 1}, {0: 0, 1: 0}, {0: 0, 1: 1},
                              {0: 0, 1: 0, 2: 0}, {0: 0, 1: 1}, {0: 0, 1: 0, 2: 0}, {0: 0, 1: 1}, {0: 0, 1: 0, 2: 0},
                              {0: 0, 1: 1}, {0: 0, 1: 0, 2: 0}]

        if correct_partitions == test_partitions:
            # Convert test_community_maps to a string because memory locations may vary by OS and PC
            str_test_community_maps = [str(community_map) for community_map in test_community_maps]

            correct_community_maps = ['{}', "{0: (['o'], ['x', 'y'])}", '{}', "{0: (['o'], ['x', 'y'])}", '{}',
                                      "{0: (['o'], ['x', 'y'])}", '{}', "{0: (['o'], ['x', 'y'])}",
                                      "{0: ([], ['x']), 1: ([], ['y'])}", "{0: (['o'], ['x', 'y'])}",
                                      "{0: ([], ['x']), 1: ([], ['y'])}", "{0: (['o'], ['x', 'y'])}",
                                      "{0: ([], ['x']), 1: ([], ['y'])}", "{0: (['o'], ['x', 'y'])}",
                                      "{0: ([], ['x']), 1: ([], ['y'])}", "{0: (['o'], ['x', 'y'])}",
                                      "{0: ([], ['x']), 1: ([], ['y'])}", "{0: (['o'], ['x', 'y'])}",
                                      "{0: ([], ['x']), 1: ([], ['y'])}", "{0: (['o'], ['x', 'y'])}",
                                      "{0: ([], ['x']), 1: ([], ['y'])}", "{0: (['o'], ['x', 'y'])}",
                                      "{0: ([], ['x']), 1: ([], ['y'])}", "{0: (['o'], ['x', 'y'])}"]

            self.assertEqual(correct_community_maps, str_test_community_maps)

        # Partition-based diagnostic test
        correct_num_communities, correct_num_nodes, test_num_communities, test_num_nodes = \
            _collect_partition_dependent_tests(test_community_maps, test_partitions)

        self.assertEqual(correct_num_communities, test_num_communities)
        self.assertEqual(correct_num_nodes, test_num_nodes)

    def test_communities_4(self):
        m_class = SOS1_simple()
        m_class._generate_model()
        model = m = m_class.model

        test_community_maps, test_partitions = _collect_community_maps(model)

        correct_partitions = [{3: 0, 4: 1, 5: 0}, {3: 0, 4: 1, 5: 0, 6: 1}, {3: 0, 4: 1, 5: 0},
                              {3: 0, 4: 0, 5: 0, 6: 0}, {3: 0, 4: 1, 5: 0},
                              {3: 0, 4: 1, 5: 0, 6: 1}, {3: 0, 4: 1, 5: 0}, {3: 0, 4: 0, 5: 0, 6: 0},
                              {0: 0, 1: 1, 2: 1}, {0: 0, 1: 0, 2: 0},
                              {0: 0, 1: 1, 2: 1}, {0: 0, 1: 0, 2: 0}, {0: 0, 1: 1, 2: 1}, {0: 0, 1: 0, 2: 0},
                              {0: 0, 1: 1, 2: 1},
                              {0: 0, 1: 0, 2: 0}, {0: 0, 1: 1, 2: 2, 3: 2, 4: 0, 5: 1},
                              {0: 0, 1: 1, 2: 2, 3: 0, 4: 2, 5: 0, 6: 1},
                              {0: 0, 1: 1, 2: 2, 3: 2, 4: 0, 5: 1}, {0: 0, 1: 1, 2: 2, 3: 0, 4: 2, 5: 0, 6: 1},
                              {0: 0, 1: 1, 2: 2, 3: 2, 4: 0, 5: 1}, {0: 0, 1: 1, 2: 2, 3: 0, 4: 2, 5: 0, 6: 1},
                              {0: 0, 1: 1, 2: 2, 3: 2, 4: 0, 5: 1}, {0: 0, 1: 1, 2: 2, 3: 0, 4: 2, 5: 0, 6: 1}]

        if correct_partitions == test_partitions:
            # Convert test_community_maps to a string because memory locations may vary by OS and PC
            str_test_community_maps = [str(community_map) for community_map in test_community_maps]

            correct_community_maps = ["{0: (['c1', 'c4'], ['y[1]', 'y[2]']), 1: (['c2'], ['x'])}",
                                      "{0: (['obj', 'c2'], ['x', 'y[1]', 'y[2]']), 1: (['c1', 'c4'], ['y[1]', 'y[2]'])}",
                                      "{0: (['c1', 'c4'], ['y[1]', 'y[2]']), 1: (['c2'], ['x'])}",
                                      "{0: (['obj', 'c1', 'c2', 'c4'], ['x', 'y[1]', 'y[2]'])}",
                                      "{0: (['c1', 'c4'], ['y[1]', 'y[2]']), 1: (['c2'], ['x'])}",
                                      "{0: (['obj', 'c2'], ['x', 'y[1]', 'y[2]']), 1: (['c1', 'c4'], ['y[1]', 'y[2]'])}",
                                      "{0: (['c1', 'c4'], ['y[1]', 'y[2]']), 1: (['c2'], ['x'])}",
                                      "{0: (['obj', 'c1', 'c2', 'c4'], ['x', 'y[1]', 'y[2]'])}",
                                      "{0: (['c2'], ['x']), 1: (['c1', 'c4'], ['y[1]', 'y[2]'])}",
                                      "{0: (['obj', 'c1', 'c2', 'c4'], ['x', 'y[1]', 'y[2]'])}",
                                      "{0: (['c2'], ['x']), 1: (['c1', 'c4'], ['y[1]', 'y[2]'])}",
                                      "{0: (['obj', 'c1', 'c2', 'c4'], ['x', 'y[1]', 'y[2]'])}",
                                      "{0: (['c2'], ['x']), 1: (['c1', 'c4'], ['y[1]', 'y[2]'])}",
                                      "{0: (['obj', 'c1', 'c2', 'c4'], ['x', 'y[1]', 'y[2]'])}",
                                      "{0: (['c2'], ['x']), 1: (['c1', 'c4'], ['y[1]', 'y[2]'])}",
                                      "{0: (['obj', 'c1', 'c2', 'c4'], ['x', 'y[1]', 'y[2]'])}",
                                      "{0: (['c2'], ['x']), 1: (['c4'], ['y[1]']), 2: (['c1'], ['y[2]'])}",
                                      "{0: (['obj', 'c2'], ['x']), 1: (['c4'], ['y[1]']), 2: (['c1'], ['y[2]'])}",
                                      "{0: (['c2'], ['x']), 1: (['c4'], ['y[1]']), 2: (['c1'], ['y[2]'])}",
                                      "{0: (['obj', 'c2'], ['x']), 1: (['c4'], ['y[1]']), 2: (['c1'], ['y[2]'])}",
                                      "{0: (['c2'], ['x']), 1: (['c4'], ['y[1]']), 2: (['c1'], ['y[2]'])}",
                                      "{0: (['obj', 'c2'], ['x']), 1: (['c4'], ['y[1]']), 2: (['c1'], ['y[2]'])}",
                                      "{0: (['c2'], ['x']), 1: (['c4'], ['y[1]']), 2: (['c1'], ['y[2]'])}",
                                      "{0: (['obj', 'c2'], ['x']), 1: (['c4'], ['y[1]']), 2: (['c1'], ['y[2]'])}"]

            self.assertEqual(correct_community_maps, str_test_community_maps)

        # Partition-based diagnostic test
        correct_num_communities, correct_num_nodes, test_num_communities, test_num_nodes = \
            _collect_partition_dependent_tests(test_community_maps, test_partitions)

        self.assertEqual(correct_num_communities, test_num_communities)
        self.assertEqual(correct_num_nodes, test_num_nodes)

    def test_communities_5(self):
        model = m = create_model_5()

        test_community_maps, test_partitions = _collect_community_maps(model)

        correct_partitions = [{6: 0, 7: 0, 8: 0, 9: 0, 10: 0}, {6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0},
                              {6: 0, 7: 0, 8: 0, 9: 0, 10: 0}, {6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0},
                              {6: 0, 7: 0, 8: 0, 9: 0, 10: 0}, {6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0},
                              {6: 0, 7: 0, 8: 0, 9: 0, 10: 0}, {6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0},
                              {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}, {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
                              {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}, {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
                              {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}, {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
                              {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}, {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
                              {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0},
                              {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 1, 7: 2, 8: 4, 9: 0, 10: 3, 11: 5},
                              {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0},
                              {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 1, 7: 2, 8: 4, 9: 0, 10: 3, 11: 5},
                              {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0},
                              {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 1, 7: 2, 8: 4, 9: 0, 10: 3, 11: 5},
                              {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0},
                              {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 1, 7: 2, 8: 4, 9: 0, 10: 3, 11: 5}]

        if correct_partitions == test_partitions:
            # Convert test_community_maps to a string because memory locations may vary by OS and PC
            str_test_community_maps = [str(community_map) for community_map in test_community_maps]

            correct_community_maps = [
                "{0: (['c1', 'c2', 'c3', 'c4', 'c5'], ['i1', 'i2', 'i3', 'i4', 'i5', 'i6'])}",
                "{0: (['obj', 'c1', 'c2', 'c3', 'c4', 'c5'], ['i1', 'i2', 'i3', 'i4', 'i5', 'i6'])}",
                "{0: (['c1', 'c2', 'c3', 'c4', 'c5'], ['i1', 'i2', 'i3', 'i4', 'i5', 'i6'])}",
                "{0: (['obj', 'c1', 'c2', 'c3', 'c4', 'c5'], ['i1', 'i2', 'i3', 'i4', 'i5', 'i6'])}",
                "{0: (['c1', 'c2', 'c3', 'c4', 'c5'], ['i1', 'i2', 'i3', 'i4', 'i5', 'i6'])}",
                "{0: (['obj', 'c1', 'c2', 'c3', 'c4', 'c5'], ['i1', 'i2', 'i3', 'i4', 'i5', 'i6'])}",
                "{0: (['c1', 'c2', 'c3', 'c4', 'c5'], ['i1', 'i2', 'i3', 'i4', 'i5', 'i6'])}",
                "{0: (['obj', 'c1', 'c2', 'c3', 'c4', 'c5'], ['i1', 'i2', 'i3', 'i4', 'i5', 'i6'])}",
                "{0: (['c1', 'c2', 'c3', 'c4', 'c5'], ['i1', 'i2', 'i3', 'i4', 'i5', 'i6'])}",
                "{0: (['obj', 'c1', 'c2', 'c3', 'c4', 'c5'], ['i1', 'i2', 'i3', 'i4', 'i5', 'i6'])}",
                "{0: (['c1', 'c2', 'c3', 'c4', 'c5'], ['i1', 'i2', 'i3', 'i4', 'i5', 'i6'])}",
                "{0: (['obj', 'c1', 'c2', 'c3', 'c4', 'c5'], ['i1', 'i2', 'i3', 'i4', 'i5', 'i6'])}",
                "{0: (['c1', 'c2', 'c3', 'c4', 'c5'], ['i1', 'i2', 'i3', 'i4', 'i5', 'i6'])}",
                "{0: (['obj', 'c1', 'c2', 'c3', 'c4', 'c5'], ['i1', 'i2', 'i3', 'i4', 'i5', 'i6'])}",
                "{0: (['c1', 'c2', 'c3', 'c4', 'c5'], ['i1', 'i2', 'i3', 'i4', 'i5', 'i6'])}",
                "{0: (['obj', 'c1', 'c2', 'c3', 'c4', 'c5'], ['i1', 'i2', 'i3', 'i4', 'i5', 'i6'])}",
                "{0: (['c1', 'c2', 'c3', 'c4', 'c5'], ['i1', 'i2', 'i3', 'i4', 'i5', 'i6'])}",
                "{0: (['c3'], ['i1']), 1: (['obj'], ['i2']), 2: (['c1'], ['i3']), "
                "3: (['c4'], ['i4']), 4: (['c2'], ['i5']), 5: (['c5'], ['i6'])}",
                "{0: (['c1', 'c2', 'c3', 'c4', 'c5'], ['i1', 'i2', 'i3', 'i4', 'i5', 'i6'])}",
                "{0: (['c3'], ['i1']), 1: (['obj'], ['i2']), 2: (['c1'], ['i3']), "
                "3: (['c4'], ['i4']), 4: (['c2'], ['i5']), 5: (['c5'], ['i6'])}",
                "{0: (['c1', 'c2', 'c3', 'c4', 'c5'], ['i1', 'i2', 'i3', 'i4', 'i5', 'i6'])}",
                "{0: (['c3'], ['i1']), 1: (['obj'], ['i2']), 2: (['c1'], ['i3']), "
                "3: (['c4'], ['i4']), 4: (['c2'], ['i5']), 5: (['c5'], ['i6'])}",
                "{0: (['c1', 'c2', 'c3', 'c4', 'c5'], ['i1', 'i2', 'i3', 'i4', 'i5', 'i6'])}",
                "{0: (['c3'], ['i1']), 1: (['obj'], ['i2']), 2: (['c1'], ['i3']), "
                "3: (['c4'], ['i4']), 4: (['c2'], ['i5']), 5: (['c5'], ['i6'])}"]

            self.assertEqual(correct_community_maps, str_test_community_maps)

        # Partition-based diagnostic test
        correct_num_communities, correct_num_nodes, test_num_communities, test_num_nodes = \
            _collect_partition_dependent_tests(test_community_maps, test_partitions)

        self.assertEqual(correct_num_communities, test_num_communities)
        self.assertEqual(correct_num_nodes, test_num_nodes)

    def test_communities_6(self):
        model = m = create_model_6()

        test_community_maps, test_partitions = _collect_community_maps(model)

        correct_partitions = [{4: 0, 5: 1}, {4: 0, 5: 0, 6: 1}, {4: 0, 5: 1}, {4: 0, 5: 0, 6: 1}, {4: 0, 5: 1},
                              {4: 0, 5: 0, 6: 1}, {4: 0, 5: 1}, {4: 0, 5: 0, 6: 1}, {0: 0, 1: 0, 2: 1, 3: 1},
                              {0: 0, 1: 0, 2: 1, 3: 1}, {0: 0, 1: 0, 2: 1, 3: 1}, {0: 0, 1: 0, 2: 1, 3: 1},
                              {0: 0, 1: 0, 2: 1, 3: 1}, {0: 0, 1: 0, 2: 1, 3: 1}, {0: 0, 1: 0, 2: 1, 3: 1},
                              {0: 0, 1: 0, 2: 1, 3: 1}, {0: 0, 1: 0, 2: 1, 3: 1, 4: 0, 5: 1},
                              {0: 0, 1: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 1}, {0: 0, 1: 0, 2: 1, 3: 1, 4: 0, 5: 1},
                              {0: 0, 1: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 1}, {0: 0, 1: 0, 2: 1, 3: 1, 4: 0, 5: 1},
                              {0: 0, 1: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 1}, {0: 0, 1: 0, 2: 1, 3: 1, 4: 0, 5: 1},
                              {0: 0, 1: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 1}]

        if correct_partitions == test_partitions:
            # Convert test_community_maps to a string because memory locations may vary by OS and PC
            str_test_community_maps = [str(community_map) for community_map in test_community_maps]

            correct_community_maps = ["{0: (['c1'], ['x1', 'x2']), 1: (['c2'], ['x3', 'x4'])}",
                                      "{0: (['obj', 'c1'], ['x1', 'x2']), 1: (['c2'], ['x3', 'x4'])}",
                                      "{0: (['c1'], ['x1', 'x2']), 1: (['c2'], ['x3', 'x4'])}",
                                      "{0: (['obj', 'c1'], ['x1', 'x2']), 1: (['c2'], ['x3', 'x4'])}",
                                      "{0: (['c1'], ['x1', 'x2']), 1: (['c2'], ['x3', 'x4'])}",
                                      "{0: (['obj', 'c1'], ['x1', 'x2']), 1: (['c2'], ['x3', 'x4'])}",
                                      "{0: (['c1'], ['x1', 'x2']), 1: (['c2'], ['x3', 'x4'])}",
                                      "{0: (['obj', 'c1'], ['x1', 'x2']), 1: (['c2'], ['x3', 'x4'])}",
                                      "{0: (['c1'], ['x1', 'x2']), 1: (['c2'], ['x3', 'x4'])}",
                                      "{0: (['obj', 'c1'], ['x1', 'x2']), 1: (['c2'], ['x3', 'x4'])}",
                                      "{0: (['c1'], ['x1', 'x2']), 1: (['c2'], ['x3', 'x4'])}",
                                      "{0: (['obj', 'c1'], ['x1', 'x2']), 1: (['c2'], ['x3', 'x4'])}",
                                      "{0: (['c1'], ['x1', 'x2']), 1: (['c2'], ['x3', 'x4'])}",
                                      "{0: (['obj', 'c1'], ['x1', 'x2']), 1: (['c2'], ['x3', 'x4'])}",
                                      "{0: (['c1'], ['x1', 'x2']), 1: (['c2'], ['x3', 'x4'])}",
                                      "{0: (['obj', 'c1'], ['x1', 'x2']), 1: (['c2'], ['x3', 'x4'])}",
                                      "{0: (['c1'], ['x1', 'x2']), 1: (['c2'], ['x3', 'x4'])}",
                                      "{0: (['obj', 'c1'], ['x1', 'x2']), 1: (['c2'], ['x3', 'x4'])}",
                                      "{0: (['c1'], ['x1', 'x2']), 1: (['c2'], ['x3', 'x4'])}",
                                      "{0: (['obj', 'c1'], ['x1', 'x2']), 1: (['c2'], ['x3', 'x4'])}",
                                      "{0: (['c1'], ['x1', 'x2']), 1: (['c2'], ['x3', 'x4'])}",
                                      "{0: (['obj', 'c1'], ['x1', 'x2']), 1: (['c2'], ['x3', 'x4'])}",
                                      "{0: (['c1'], ['x1', 'x2']), 1: (['c2'], ['x3', 'x4'])}",
                                      "{0: (['obj', 'c1'], ['x1', 'x2']), 1: (['c2'], ['x3', 'x4'])}"]

            self.assertEqual(correct_community_maps, str_test_community_maps)

        # Partition-based diagnostic test
        correct_num_communities, correct_num_nodes, test_num_communities, test_num_nodes = \
            _collect_partition_dependent_tests(test_community_maps, test_partitions)

        self.assertEqual(correct_num_communities, test_num_communities)
        self.assertEqual(correct_num_nodes, test_num_nodes)

    def test_communities_7(self):
        model = m = disconnected_model()

        test_community_maps, test_partitions = _collect_community_maps(model)

        correct_partitions = [{2: 0}, {2: 0, 3: 1, 4: 1}, {2: 0}, {2: 0, 3: 1, 4: 1}, {2: 0}, {2: 0, 3: 1, 4: 1},
                              {2: 0}, {2: 0, 3: 1, 4: 1}, {0: 0, 1: 1}, {0: 0, 1: 1}, {0: 0, 1: 1}, {0: 0, 1: 1},
                              {0: 0, 1: 1}, {0: 0, 1: 1}, {0: 0, 1: 1}, {0: 0, 1: 1}, {0: 0, 1: 1, 2: 0},
                              {0: 0, 1: 1, 2: 2, 3: 0, 4: 0}, {0: 0, 1: 1, 2: 0}, {0: 0, 1: 1, 2: 2, 3: 0, 4: 0},
                              {0: 0, 1: 1, 2: 0}, {0: 0, 1: 1, 2: 2, 3: 0, 4: 0}, {0: 0, 1: 1, 2: 0},
                              {0: 0, 1: 1, 2: 2, 3: 0, 4: 0}]

        if correct_partitions == test_partitions:
            # Convert test_community_maps to a string because memory locations may vary by OS and PC
            str_test_community_maps = [str(community_map) for community_map in test_community_maps]

            correct_community_maps = ["{0: (['c1'], ['x1'])}", "{0: (['OBJ'], []), 1: (['obj', 'c1'], ['x1'])}",
                                      "{0: (['c1'], ['x1'])}", "{0: (['OBJ'], []), 1: (['obj', 'c1'], ['x1'])}",
                                      "{0: (['c1'], ['x1'])}", "{0: (['OBJ'], []), 1: (['obj', 'c1'], ['x1'])}",
                                      "{0: (['c1'], ['x1'])}", "{0: (['OBJ'], []), 1: (['obj', 'c1'], ['x1'])}",
                                      "{0: (['c1'], ['x1']), 1: ([], ['x2'])}",
                                      "{0: (['obj', 'c1'], ['x1']), 1: ([], ['x2'])}",
                                      "{0: (['c1'], ['x1']), 1: ([], ['x2'])}",
                                      "{0: (['obj', 'c1'], ['x1']), 1: ([], ['x2'])}",
                                      "{0: (['c1'], ['x1']), 1: ([], ['x2'])}",
                                      "{0: (['obj', 'c1'], ['x1']), 1: ([], ['x2'])}",
                                      "{0: (['c1'], ['x1']), 1: ([], ['x2'])}",
                                      "{0: (['obj', 'c1'], ['x1']), 1: ([], ['x2'])}",
                                      "{0: (['c1'], ['x1']), 1: ([], ['x2'])}",
                                      "{0: (['obj', 'c1'], ['x1']), 1: ([], ['x2']), 2: (['OBJ'], [])}",
                                      "{0: (['c1'], ['x1']), 1: ([], ['x2'])}",
                                      "{0: (['obj', 'c1'], ['x1']), 1: ([], ['x2']), 2: (['OBJ'], [])}",
                                      "{0: (['c1'], ['x1']), 1: ([], ['x2'])}",
                                      "{0: (['obj', 'c1'], ['x1']), 1: ([], ['x2']), 2: (['OBJ'], [])}",
                                      "{0: (['c1'], ['x1']), 1: ([], ['x2'])}",
                                      "{0: (['obj', 'c1'], ['x1']), 1: ([], ['x2']), 2: (['OBJ'], [])}"]

            self.assertEqual(correct_community_maps, str_test_community_maps)

        # Partition-based diagnostic test
        correct_num_communities, correct_num_nodes, test_num_communities, test_num_nodes = \
            _collect_partition_dependent_tests(test_community_maps, test_partitions)

        self.assertEqual(correct_num_communities, test_num_communities)
        self.assertEqual(correct_num_nodes, test_num_nodes)

    def test_decode_1(self):
        model = m = decode_model_1()

        test_community_maps, test_partitions = _collect_community_maps(model)

        correct_partitions = [{4: 0, 5: 0, 6: 1, 7: 1, 8: 1}, {4: 0, 5: 0, 6: 1, 7: 1, 8: 1},
                              {4: 0, 5: 0, 6: 1, 7: 1, 8: 1}, {4: 0, 5: 0, 6: 1, 7: 1, 8: 1},
                              {4: 0, 5: 0, 6: 1, 7: 1, 8: 1}, {4: 0, 5: 0, 6: 1, 7: 1, 8: 1},
                              {4: 0, 5: 0, 6: 1, 7: 1, 8: 1}, {4: 0, 5: 0, 6: 1, 7: 1, 8: 1}, {0: 0, 1: 0, 2: 1, 3: 1},
                              {0: 0, 1: 0, 2: 1, 3: 1}, {0: 0, 1: 0, 2: 1, 3: 1}, {0: 0, 1: 0, 2: 1, 3: 1},
                              {0: 0, 1: 0, 2: 1, 3: 1}, {0: 0, 1: 0, 2: 1, 3: 1}, {0: 0, 1: 0, 2: 1, 3: 1},
                              {0: 0, 1: 0, 2: 1, 3: 1}, {0: 0, 1: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 1, 7: 1, 8: 1},
                              {0: 0, 1: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 1, 7: 1, 8: 1},
                              {0: 0, 1: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 1, 7: 1, 8: 1},
                              {0: 0, 1: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 1, 7: 1, 8: 1},
                              {0: 0, 1: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 1, 7: 1, 8: 1},
                              {0: 0, 1: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 1, 7: 1, 8: 1},
                              {0: 0, 1: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 1, 7: 1, 8: 1},
                              {0: 0, 1: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 1, 7: 1, 8: 1}]

        if correct_partitions == test_partitions:
            # Convert test_community_maps to a string because memory locations may vary by OS and PC
            str_test_community_maps = [str(community_map) for community_map in test_community_maps]

            correct_community_maps = ["{0: (['c1', 'c2'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x2', 'x3', 'x4'])}",
                                      "{0: (['c1', 'c2'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x2', 'x3', 'x4'])}",
                                      "{0: (['c1', 'c2'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x2', 'x3', 'x4'])}",
                                      "{0: (['c1', 'c2'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x2', 'x3', 'x4'])}",
                                      "{0: (['c1', 'c2'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x2', 'x3', 'x4'])}",
                                      "{0: (['c1', 'c2'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x2', 'x3', 'x4'])}",
                                      "{0: (['c1', 'c2'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x2', 'x3', 'x4'])}",
                                      "{0: (['c1', 'c2'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x2', 'x3', 'x4'])}",
                                      "{0: (['c1', 'c2', 'c3'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x3', 'x4'])}",
                                      "{0: (['c1', 'c2', 'c3'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x3', 'x4'])}",
                                      "{0: (['c1', 'c2', 'c3'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x3', 'x4'])}",
                                      "{0: (['c1', 'c2', 'c3'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x3', 'x4'])}",
                                      "{0: (['c1', 'c2', 'c3'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x3', 'x4'])}",
                                      "{0: (['c1', 'c2', 'c3'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x3', 'x4'])}",
                                      "{0: (['c1', 'c2', 'c3'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x3', 'x4'])}",
                                      "{0: (['c1', 'c2', 'c3'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x3', 'x4'])}",
                                      "{0: (['c1', 'c2'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x3', 'x4'])}",
                                      "{0: (['c1', 'c2'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x3', 'x4'])}",
                                      "{0: (['c1', 'c2'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x3', 'x4'])}",
                                      "{0: (['c1', 'c2'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x3', 'x4'])}",
                                      "{0: (['c1', 'c2'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x3', 'x4'])}",
                                      "{0: (['c1', 'c2'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x3', 'x4'])}",
                                      "{0: (['c1', 'c2'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x3', 'x4'])}",
                                      "{0: (['c1', 'c2'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x3', 'x4'])}"]

            self.assertEqual(correct_community_maps, str_test_community_maps)

        # Partition-based diagnostic test
        correct_num_communities, correct_num_nodes, test_num_communities, test_num_nodes = \
            _collect_partition_dependent_tests(test_community_maps, test_partitions)

        self.assertEqual(correct_num_communities, test_num_communities)
        self.assertEqual(correct_num_nodes, test_num_nodes)

    def test_decode_2(self):
        model = m = decode_model_2()

        test_community_maps, test_partitions = _collect_community_maps(model)

        correct_partitions = [{7: 0, 8: 0, 9: 0, 10: 1, 11: 1, 12: 1}, {7: 0, 8: 0, 9: 0, 10: 1, 11: 1, 12: 1},
                              {7: 0, 8: 0, 9: 1, 10: 1, 11: 1, 12: 1}, {7: 0, 8: 0, 9: 1, 10: 1, 11: 1, 12: 1},
                              {7: 0, 8: 0, 9: 0, 10: 1, 11: 1, 12: 1}, {7: 0, 8: 0, 9: 0, 10: 1, 11: 1, 12: 1},
                              {7: 0, 8: 0, 9: 1, 10: 1, 11: 1, 12: 1}, {7: 0, 8: 0, 9: 1, 10: 1, 11: 1, 12: 1},
                              {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1}, {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1},
                              {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1}, {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1},
                              {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1}, {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1},
                              {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1}, {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1},
                              {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 1, 10: 1, 11: 1, 12: 1},
                              {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 1, 10: 1, 11: 1, 12: 1},
                              {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 1, 10: 1, 11: 1, 12: 1},
                              {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 1, 10: 1, 11: 1, 12: 1},
                              {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 1, 10: 1, 11: 1, 12: 1},
                              {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 1, 10: 1, 11: 1, 12: 1},
                              {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 1, 10: 1, 11: 1, 12: 1},
                              {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 1, 10: 1, 11: 1, 12: 1}]

        if correct_partitions == test_partitions:
            # Convert test_community_maps to a string because memory locations may vary by OS and PC
            str_test_community_maps = [str(community_map) for community_map in test_community_maps]

            correct_community_maps = [
                "{0: (['c1', 'c2', 'c3'], ['x[1]', 'x[2]', 'x[3]', 'x[4]', 'x[5]']), "
                "1: (['c4', 'c5', 'c6'], ['x[4]', 'x[5]', 'x[6]', 'x[7]'])}",
                "{0: (['c1', 'c2', 'c3'], ['x[1]', 'x[2]', 'x[3]', 'x[4]', 'x[5]']), "
                "1: (['c4', 'c5', 'c6'], ['x[4]', 'x[5]', 'x[6]', 'x[7]'])}",
                "{0: (['c1', 'c2'], ['x[1]', 'x[2]', 'x[3]']), "
                "1: (['c3', 'c4', 'c5', 'c6'], ['x[3]', 'x[4]', 'x[5]', 'x[6]', 'x[7]'])}",
                "{0: (['c1', 'c2'], ['x[1]', 'x[2]', 'x[3]']), "
                "1: (['c3', 'c4', 'c5', 'c6'], ['x[3]', 'x[4]', 'x[5]', 'x[6]', 'x[7]'])}",
                "{0: (['c1', 'c2', 'c3'], ['x[1]', 'x[2]', 'x[3]', 'x[4]', 'x[5]']), "
                "1: (['c4', 'c5', 'c6'], ['x[4]', 'x[5]', 'x[6]', 'x[7]'])}",
                "{0: (['c1', 'c2', 'c3'], ['x[1]', 'x[2]', 'x[3]', 'x[4]', 'x[5]']), "
                "1: (['c4', 'c5', 'c6'], ['x[4]', 'x[5]', 'x[6]', 'x[7]'])}",
                "{0: (['c1', 'c2'], ['x[1]', 'x[2]', 'x[3]']), "
                "1: (['c3', 'c4', 'c5', 'c6'], ['x[3]', 'x[4]', 'x[5]', 'x[6]', 'x[7]'])}",
                "{0: (['c1', 'c2'], ['x[1]', 'x[2]', 'x[3]']), "
                "1: (['c3', 'c4', 'c5', 'c6'], ['x[3]', 'x[4]', 'x[5]', 'x[6]', 'x[7]'])}",
                "{0: (['c1', 'c2', 'c3'], ['x[1]', 'x[2]', 'x[3]']), "
                "1: (['c3', 'c4', 'c5', 'c6'], ['x[4]', 'x[5]', 'x[6]', 'x[7]'])}",
                "{0: (['c1', 'c2', 'c3'], ['x[1]', 'x[2]', 'x[3]']), "
                "1: (['c3', 'c4', 'c5', 'c6'], ['x[4]', 'x[5]', 'x[6]', 'x[7]'])}",
                "{0: (['c1', 'c2', 'c3'], ['x[1]', 'x[2]', 'x[3]']), "
                "1: (['c3', 'c4', 'c5', 'c6'], ['x[4]', 'x[5]', 'x[6]', 'x[7]'])}",
                "{0: (['c1', 'c2', 'c3'], ['x[1]', 'x[2]', 'x[3]']), "
                "1: (['c3', 'c4', 'c5', 'c6'], ['x[4]', 'x[5]', 'x[6]', 'x[7]'])}",
                "{0: (['c1', 'c2', 'c3'], ['x[1]', 'x[2]', 'x[3]']), "
                "1: (['c3', 'c4', 'c5', 'c6'], ['x[4]', 'x[5]', 'x[6]', 'x[7]'])}",
                "{0: (['c1', 'c2', 'c3'], ['x[1]', 'x[2]', 'x[3]']), "
                "1: (['c3', 'c4', 'c5', 'c6'], ['x[4]', 'x[5]', 'x[6]', 'x[7]'])}",
                "{0: (['c1', 'c2', 'c3'], ['x[1]', 'x[2]', 'x[3]']), "
                "1: (['c3', 'c4', 'c5', 'c6'], ['x[4]', 'x[5]', 'x[6]', 'x[7]'])}",
                "{0: (['c1', 'c2', 'c3'], ['x[1]', 'x[2]', 'x[3]']), "
                "1: (['c3', 'c4', 'c5', 'c6'], ['x[4]', 'x[5]', 'x[6]', 'x[7]'])}",
                "{0: (['c1', 'c2'], ['x[1]', 'x[2]', 'x[3]']), "
                "1: (['c3', 'c4', 'c5', 'c6'], ['x[4]', 'x[5]', 'x[6]', 'x[7]'])}",
                "{0: (['c1', 'c2'], ['x[1]', 'x[2]', 'x[3]']), "
                "1: (['c3', 'c4', 'c5', 'c6'], ['x[4]', 'x[5]', 'x[6]', 'x[7]'])}",
                "{0: (['c1', 'c2'], ['x[1]', 'x[2]', 'x[3]']), "
                "1: (['c3', 'c4', 'c5', 'c6'], ['x[4]', 'x[5]', 'x[6]', 'x[7]'])}",
                "{0: (['c1', 'c2'], ['x[1]', 'x[2]', 'x[3]']), "
                "1: (['c3', 'c4', 'c5', 'c6'], ['x[4]', 'x[5]', 'x[6]', 'x[7]'])}",
                "{0: (['c1', 'c2'], ['x[1]', 'x[2]', 'x[3]']), "
                "1: (['c3', 'c4', 'c5', 'c6'], ['x[4]', 'x[5]', 'x[6]', 'x[7]'])}",
                "{0: (['c1', 'c2'], ['x[1]', 'x[2]', 'x[3]']), "
                "1: (['c3', 'c4', 'c5', 'c6'], ['x[4]', 'x[5]', 'x[6]', 'x[7]'])}",
                "{0: (['c1', 'c2'], ['x[1]', 'x[2]', 'x[3]']), "
                "1: (['c3', 'c4', 'c5', 'c6'], ['x[4]', 'x[5]', 'x[6]', 'x[7]'])}",
                "{0: (['c1', 'c2'], ['x[1]', 'x[2]', 'x[3]']), "
                "1: (['c3', 'c4', 'c5', 'c6'], ['x[4]', 'x[5]', 'x[6]', 'x[7]'])}"]

            self.assertEqual(correct_community_maps, str_test_community_maps)

        # Partition-based diagnostic test
        correct_num_communities, correct_num_nodes, test_num_communities, test_num_nodes = \
            _collect_partition_dependent_tests(test_community_maps, test_partitions)

        self.assertEqual(correct_num_communities, test_num_communities)
        self.assertEqual(correct_num_nodes, test_num_nodes)

    def test_communities_8(self):
        output = StringIO()

        with LoggingIntercept(output, 'pyomo.contrib.community_detection', logging.ERROR):
            detect_communities(ConcreteModel())
        self.assertIn('in detect_communities: Empty community map was returned', output.getvalue())

        with LoggingIntercept(output, 'pyomo.contrib.community_detection', logging.WARNING):
            detect_communities(one_community_model())
        self.assertIn("Community detection found that with the given parameters, the model could not be decomposed - "
                      "only one community was found", output.getvalue())

        model = 'foo'
        with self.assertRaisesRegex(TypeError, "Invalid model: 'model=%s' - model must be an instance of "
                                               "ConcreteModel" % model):
            detect_communities(model)

        model = create_model_6()

        type_of_community_map = 'foo'
        with self.assertRaisesRegex(TypeError, "Invalid value for type_of_community_map: "
                                               "'type_of_community_map=%s' - Valid values: 'bipartite', "
                                               "'constraint', 'variable'"
                                               % type_of_community_map):
            detect_communities(model, type_of_community_map=type_of_community_map)

        with_objective = 'foo'
        with self.assertRaisesRegex(TypeError, "Invalid value for with_objective: 'with_objective=%s' - "
                                               "with_objective must be a Boolean" % with_objective):
            detect_communities(model, with_objective=with_objective)

        weighted_graph = 'foo'
        with self.assertRaisesRegex(TypeError, "Invalid value for weighted_graph: 'weighted_graph=%s' - "
                                               "weighted_graph must be a Boolean" % weighted_graph):
            detect_communities(model, weighted_graph=weighted_graph)

        random_seed = 'foo'
        with self.assertRaisesRegex(TypeError, "Invalid value for random_seed: 'random_seed=%s' - random_seed "
                                               "must be a non-negative integer" % random_seed):
            detect_communities(model, random_seed=random_seed)

        random_seed = -1
        with self.assertRaisesRegex(ValueError, "Invalid value for random_seed: 'random_seed=%s' - random_seed "
                                                "must be a non-negative integer" % random_seed):
            detect_communities(model, random_seed=random_seed)

        use_only_active_components = 'foo'
        with self.assertRaisesRegex(TypeError,
                                    "Invalid value for use_only_active_components: 'use_only_active_components=%s' "
                                    "- use_only_active_components must be True or None" % use_only_active_components):
            detect_communities(model, use_only_active_components=use_only_active_components)

    @unittest.skipUnless(matplotlib_available, "matplotlib is not available.")
    def test_visualize_model_graph_1(self):
        model = decode_model_1()
        community_map_object = detect_communities(model)

        fig, pos = community_map_object.visualize_model_graph(filename='test_visualize_model_graph_1')
        correct_pos_dict_length = 5

        self.assertTrue(isinstance(pos, dict))
        self.assertEqual(len(pos), correct_pos_dict_length)

    @unittest.skipUnless(matplotlib_available, "matplotlib is not available.")
    def test_visualize_model_graph_2(self):
        model = decode_model_2()
        community_map_object = detect_communities(model)

        fig, pos = community_map_object.visualize_model_graph(type_of_graph='bipartite',
                                                              filename='test_visualize_model_graph_2')
        correct_pos_dict_length = 13

        self.assertTrue(isinstance(pos, dict))
        self.assertEqual(len(pos), correct_pos_dict_length)

    def test_generate_structured_model_1(self):
        m_class = LP_inactive_index()
        m_class._generate_model()
        model = m = m_class.model

        community_map_object = cmo = detect_communities(model, random_seed=5)
        correct_partition = {3: 0, 4: 1, 5: 0, 6: 0, 7: 1, 8: 0}
        correct_components = {'b[0].B[2].c', 'b[0].c2[1]', 'b[0].c1[3]', 'equality_constraint_list[1]',
                              'b[1].c2[2]', 'b[1].x', 'b[0].x', 'b[0].y', 'b[0].z', 'b[0].obj[2]', 'b[1].c1[2]'}

        structured_model = cmo.generate_structured_model()
        self.assertIsInstance(structured_model, Block)

        all_components = set([str(component) for component in structured_model.component_data_objects(
            ctype=(Var, Constraint, Objective, ConstraintList), active=cmo.use_only_active_components,
            descend_into=True)])

        if cmo.graph_partition == correct_partition:
            # Test the number of blocks
            self.assertEqual(2, len(cmo.community_map),
                             len(list(structured_model.component_data_objects(ctype=Block, descend_into=True))))

            # Test what components have been created
            self.assertEqual(all_components, correct_components)

            # Basic test for the replacement of variables
            for objective in structured_model.component_data_objects(ctype=Objective, descend_into=True):
                objective_expr = str(objective.expr)  # This for loop should execute once (only one active objective)
            correct_objective_expr = '- b[0].x + b[0].y + b[0].z'
            self.assertEqual(correct_objective_expr, objective_expr)

        self.assertEqual(len(correct_partition), len(cmo.graph_partition))
        self.assertEqual(len(correct_components), len(all_components))

    def test_generate_structured_model_2(self):
        m_class = LP_inactive_index()
        m_class._generate_model()
        model = m = m_class.model

        community_map_object = cmo = detect_communities(model, with_objective=False, random_seed=5)
        correct_partition = {3: 0, 4: 1, 5: 1, 6: 0, 7: 2}
        correct_components = {'b[2].B[2].c', 'b[1].y', 'z', 'b[0].c1[2]', 'b[1].c1[3]', 'obj[2]',
                              'equality_constraint_list[3]', 'b[0].x', 'b[1].c2[1]', 'b[2].z', 'x',
                              'equality_constraint_list[1]', 'b[0].c2[2]', 'y', 'equality_constraint_list[2]'}

        structured_model = cmo.generate_structured_model()
        self.assertIsInstance(structured_model, Block)

        all_components = set([str(component) for component in structured_model.component_data_objects(
            ctype=(Var, Constraint, Objective, ConstraintList), active=cmo.use_only_active_components,
            descend_into=True)])

        if cmo.graph_partition == correct_partition:
            # Test the number of blocks
            self.assertEqual(3, len(cmo.community_map),
                             len(list(structured_model.component_data_objects(ctype=Block, descend_into=True))))

            # Test what components have been created
            self.assertEqual(correct_components, all_components)

            # Basic test for the replacement of variables
            for objective in structured_model.component_data_objects(ctype=Objective, descend_into=True):
                objective_expr = str(
                    objective.expr)  # This for loop should only execute once (only one active objective)
            correct_objective_expr = '- x + y + z'
            self.assertEqual(correct_objective_expr, objective_expr)

        self.assertEqual(len(correct_partition), len(cmo.graph_partition))
        self.assertEqual(len(correct_components), len(all_components))

    # The next 2 tests have been commented out because they were used to show that the Louvain community detection is
    # inconsistent on different Python versions, but we no longer want these tests to fail every time we push changes

    # # Test louvain community detection to see if Python version causes any inconsistencies in community detection
    # def test_louvain_decode_1(self):
    #     model = m = decode_model_1()
    #     model_graph, _, _ = generate_model_graph(model, type_of_graph='constraint', with_objective=False,
    #                                              weighted_graph=False, use_only_active_components=None)
    #     random_seed_value = 5
    #     partition_of_graph = community_louvain.best_partition(model_graph, random_state=random_seed_value)
    #
    #     correct_partition = {4: 0, 5: 0, 6: 1, 7: 1, 8: 1}
    #
    #     self.assertEqual(correct_partition, partition_of_graph)
    #
    # def test_louvain_communities_1(self):
    #     m_class = LP_inactive_index()
    #     m_class._generate_model()
    #     model = m = m_class.model
    #
    #     model_graph, _, _ = generate_model_graph(model, type_of_graph='constraint', with_objective=True,
    #                                              weighted_graph=True, use_only_active_components=None)
    #     random_seed_value = 5
    #     partition_of_graph = community_louvain.best_partition(model_graph, random_state=random_seed_value)
    #
    #     correct_partition = {3: 0, 4: 1, 5: 0, 6: 0, 7: 0, 8: 2, 9: 2, 10: 2, 11: 0, 12: 1, 13: 1, 14: 1}
    #
    #     self.assertEqual(correct_partition, partition_of_graph)


def _collect_community_maps(model):
    """
    This is the testing helper function, which collects all 24 possible community maps for a model and
    all 24 possible partitions of the respective model graphs (based on Louvain community detection done on
    the networkX graph of the model) - (24 is the total combinations possible assuming that we have
    provided a seed value for the community detection).

    This function generates all combinations of the parameters for detect_communities by looping through the types
    of community maps we can have and then looping through every combination of the other parameters (as seen below).
    The inner for loop is used to create 8 different combinations of parameter values for with_objective,
    weighted_graph, and use_only_active_components by counting from 0 to 7 and then interpreting this number as a
    binary value, which will then be used to assign a True/False value to each of the three parameters.

    Parameters
    ----------
    model: Block
        a Pyomo model or block to be used for community detection

    Returns
    -------
    list_of_community_maps:
        a list of 24 CommunityMap objects (the 24 community maps correspond to all the
        combinations of parameters that are possible for detect_communities).
        The parameters for creating the 24 community maps are:
        type_of_community_map = ['constraint', 'variable', 'bipartite']
        with_objective = [False, True]
        weighted_graph = [False, True]
        use_only_active_components = [None, True]

    list_of_partitions:
        a list of 24 partitions based on Louvain community detection done on the networkX graph of the model (the
        24 partitions correspond to all the combinations of parameters that are possible for detect_communities).
        The parameters for creating the 24 partitions are:
        type_of_community_map = ['constraint', 'variable', 'bipartite']
        with_objective = [False, True]
        weighted_graph = [False, True]
        use_only_active_components = [None, True]
    """
    random_seed_test = 5

    # Call the detect_communities with all possible parameter combinations for testing
    types_of_type_of_community_map = ['constraint', 'variable', 'bipartite']

    list_of_community_maps = []  # this will ultimately contain 24 community map objects
    list_of_partitions = []  # this will ultimately contain 24 dictionaries (partitions of networkX graphs)

    for community_map_type in types_of_type_of_community_map:
        for test_number in range(2 ** 3):  # raised to the third power because there are three boolean arguments
            argument_value_list = [0, 0, 0]
            index = 0

            # By effectively counting in binary (as done in the while loop below) up to 2**3, we will
            # loop through all True/False permutations for these three arguments
            while test_number != 0:
                argument_value_list[index] = test_number % 2
                test_number //= 2
                index += 1

            # Given a permutation, we will now assign values for arguments based on the 0 and 1 values
            with_objective, weighted_graph, use_only_active_components = [bool(val) for val in argument_value_list]

            # The 'active' parameter treats 'False' as logically ambiguous so we must change it to None if we do
            # not wish to include active components in this particular function call
            if not use_only_active_components:  # if we have given use_only_active_components a value of False
                use_only_active_components = None

            # Make the function call
            latest_community_map = detect_communities(model, type_of_community_map=community_map_type,
                                                      with_objective=with_objective, weighted_graph=weighted_graph,
                                                      random_seed=random_seed_test,
                                                      use_only_active_components=use_only_active_components)

            # Add this latest community map object and its partition to their respective lists
            list_of_community_maps.append(latest_community_map)
            list_of_partitions.append(latest_community_map.graph_partition)

    # Return the list of community maps for use in the testing functions
    return list_of_community_maps, list_of_partitions


def _collect_partition_dependent_tests(test_community_maps, test_partitions):
    expected_num_comm_list = []
    expected_num_members_list = []
    actual_num_comm_list = []
    actual_num_members_list = []

    for community_map, partition in zip(test_community_maps, test_partitions):
        expected_number_of_communities = len(set(partition.values()))
        expected_number_of_members = len(partition)

        # Now we will extract the lists within the community map that correspond to the
        # nodes of the networkX graph
        if community_map.type_of_community_map == 'constraint':
            list_of_node_lists = [community[0] for community in
                                  community_map.values()]

        elif community_map.type_of_community_map == 'variable':
            list_of_node_lists = [community[1] for community in
                                  community_map.values()]

        else:
            list_of_node_lists = [community[0] + community[1] for community in
                                  community_map.values()]

        # We have to flatten list_of_node_lists
        actual_number_of_members = len([node for node_list in list_of_node_lists for node in node_list])
        actual_number_of_communities = len(community_map)

        expected_num_comm_list.append(expected_number_of_communities)
        expected_num_members_list.append(expected_number_of_members)
        actual_num_comm_list.append(actual_number_of_communities)
        actual_num_members_list.append(actual_number_of_members)

    return expected_num_comm_list, expected_num_members_list, actual_num_comm_list, actual_num_members_list


def create_model_5():  # This model comes from a GAMS convert of instance st_test4.gms at minlplib.com
    model = m = ConcreteModel()
    m.i1 = Var(within=Integers, bounds=(0, 100), initialize=0)
    m.i2 = Var(within=Integers, bounds=(0, 100), initialize=0)
    m.i3 = Var(within=Integers, bounds=(0, 100), initialize=0)
    m.i4 = Var(within=Integers, bounds=(0, 1), initialize=0)
    m.i5 = Var(within=Integers, bounds=(0, 1), initialize=0)
    m.i6 = Var(within=Integers, bounds=(0, 2), initialize=0)
    m.obj = Objective(
        expr=0.5 * m.i1 * m.i1 + 6.5 * m.i1 + 7 * m.i6 * m.i6 - m.i6 - m.i2 - 2 * m.i3 + 3 * m.i4 - 2 * m.i5,
        sense=minimize)
    m.c1 = Constraint(expr=m.i1 + 2 * m.i2 + 8 * m.i3 + m.i4 + 3 * m.i5 + 5 * m.i6 <= 16)
    m.c2 = Constraint(expr=- 8 * m.i1 - 4 * m.i2 - 2 * m.i3 + 2 * m.i4 + 4 * m.i5 - m.i6 <= -1)
    m.c3 = Constraint(expr=2 * m.i1 + 0.5 * m.i2 + 0.2 * m.i3 - 3 * m.i4 - m.i5 - 4 * m.i6 <= 24)
    m.c4 = Constraint(expr=0.2 * m.i1 + 2 * m.i2 + 0.1 * m.i3 - 4 * m.i4 + 2 * m.i5 + 2 * m.i6 <= 12)
    m.c5 = Constraint(expr=- 0.1 * m.i1 - 0.5 * m.i2 + 2 * m.i3 + 5 * m.i4 - 5 * m.i5 + 3 * m.i6 <= 3)
    return model


def create_model_6():  # Toy model
    model = m = ConcreteModel()
    m.x1 = Var(bounds=(0, 1))
    m.x2 = Var(bounds=(0, 1))
    m.x3 = Var(bounds=(0, 1))
    m.x4 = Var(bounds=(0, 1))
    m.obj = Objective(expr=m.x1, sense=minimize)
    m.c1 = Constraint(expr=m.x1 + m.x2 >= 1)
    m.c2 = Constraint(expr=m.x3 + m.x4 >= 1)
    return model


def disconnected_model():
    model = m = ConcreteModel()
    m.x1 = Var(bounds=(0, 1))
    m.x2 = Var(bounds=(0, 1))
    m.OBJ = Objective(expr=1, sense=minimize)
    m.obj = Objective(expr=m.x1, sense=minimize)
    m.c1 = Constraint(expr=m.x1 >= 1)
    return model


def decode_model_1():
    model = m = ConcreteModel()
    m.x1 = Var(initialize=-3)
    m.x2 = Var(initialize=-1)
    m.x3 = Var(initialize=-3)
    m.x4 = Var(initialize=-1)
    m.c1 = Constraint(expr=m.x1 + m.x2 <= 0)
    m.c2 = Constraint(expr=m.x1 - 3 * m.x2 <= 0)
    m.c3 = Constraint(expr=m.x2 + m.x3 + 4 * m.x4 ** 2 == 0)
    m.c4 = Constraint(expr=m.x3 + m.x4 <= 0)
    m.c5 = Constraint(expr=m.x3 ** 2 + m.x4 ** 2 - 10 == 0)
    return model


def decode_model_2():
    model = m = ConcreteModel()
    m.x = Var(RangeSet(1, 7))
    m.c1 = Constraint(expr=m.x[1] + m.x[2] + m.x[3] <= 0)
    m.c2 = Constraint(expr=m.x[1] + 2 * m.x[2] + m.x[3] <= 0)
    m.c3 = Constraint(expr=m.x[3] + m.x[4] + m.x[5] <= 0)
    m.c4 = Constraint(expr=m.x[4] + m.x[5] + m.x[6] + m.x[7] <= 0)
    m.c5 = Constraint(expr=m.x[4] + 2 * m.x[5] + m.x[6] + 0.5 * m.x[7] <= 0)
    m.c6 = Constraint(expr=m.x[4] + m.x[5] + 3 * m.x[6] + m.x[7] <= 0)
    return model


def one_community_model():  # Toy model that cannot be decomposed; used to test logging messages
    model = m = ConcreteModel()
    m.x1 = Var(bounds=(0, 1))
    m.x2 = Var(bounds=(0, 1))
    m.obj = Objective(expr=m.x1, sense=minimize)
    m.c1 = Constraint(expr=m.x1 + m.x2 >= 1)
    return model


if __name__ == '__main__':
    unittest.main()
