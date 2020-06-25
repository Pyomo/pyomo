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
from pyomo.environ import ConcreteModel, Constraint, Objective, Var, Integers, minimize, RangeSet
from pyomo.contrib.community_detection.detection import detect_communities, stringify_community_map, \
    visualize_model_graph, community_louvain_available

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

        test_results = _collect_test_results(model)

        correct_community_maps = (
            {0: ([m.c1[2], m.c2[2]], [m.x]), 1: ([m.c1[3], m.c2[1]], [m.y]), 2: ([m.B[2].c], [m.z])},
            {0: ([m.c1[2], m.c2[2]], [m.x]), 1: ([m.c1[3], m.c2[1]], [m.y]), 2: ([m.B[2].c], [m.z])},
            {0: ([m.c1[2], m.c2[2]], [m.x]), 1: ([m.c1[3], m.c2[1]], [m.y]), 2: ([m.obj[2], m.B[2].c], [m.z])},
            {0: ([m.c1[2], m.c2[2]], [m.x]), 1: ([m.c1[3], m.c2[1]], [m.y]), 2: ([m.obj[2], m.B[2].c], [m.z])},
            {0: ([m.c1[2], m.c2[2]], [m.x]), 1: ([m.c1[3], m.c2[1]], [m.y]), 2: ([m.B[2].c], [m.z])},
            {0: ([m.c1[2], m.c2[2]], [m.x]), 1: ([m.c1[3], m.c2[1]], [m.y]), 2: ([m.B[2].c], [m.z])},
            {0: ([m.c1[2], m.c2[2]], [m.x]), 1: ([m.obj[2], m.c1[3], m.c2[1], m.B[2].c], [m.x, m.y, m.z])},
            {0: ([m.c1[2], m.c2[2]], [m.x]), 1: ([m.obj[2], m.c1[3], m.c2[1], m.B[2].c], [m.x, m.y, m.z])},
            {0: ([m.c1[2], m.c2[2]], [m.x]), 1: ([m.c1[3], m.c2[1]], [m.y]), 2: ([m.B[2].c], [m.z])},
            {0: ([m.c1[2], m.c2[2]], [m.x]), 1: ([m.c1[3], m.c2[1]], [m.y]), 2: ([m.B[2].c], [m.z])},
            {0: ([m.obj[2], m.c1[2], m.c1[3], m.c2[1], m.c2[2], m.B[2].c], [m.x, m.y, m.z])},
            {0: ([m.obj[2], m.c1[2], m.c1[3], m.c2[1], m.c2[2], m.B[2].c], [m.x, m.y, m.z])})

        self.assertEqual(correct_community_maps, test_results)

    def test_communities_2(self):
        m_class = QP_simple()
        m_class._generate_model()
        model = m = m_class.model

        test_results = _collect_test_results(model)

        correct_community_maps = ({0: ([m.c2], [m.x]), 1: ([m.c1], [m.y])}, {0: ([m.c2], [m.x]), 1: ([m.c1], [m.y])},
                                  {0: ([m.c2], [m.x]), 1: ([m.obj, m.c1], [m.y])},
                                  {0: ([m.c2], [m.x]), 1: ([m.obj, m.c1], [m.y])}, {0: ([m.c1, m.c2], [m.x, m.y])},
                                  {0: ([m.c1, m.c2], [m.x, m.y])}, {0: ([m.obj, m.c1, m.c2], [m.x, m.y])},
                                  {0: ([m.obj, m.c1, m.c2], [m.x, m.y])}, {0: ([m.c1, m.c2], [m.x, m.y])},
                                  {0: ([m.c1, m.c2], [m.x, m.y])}, {0: ([m.obj, m.c1, m.c2], [m.x, m.y])},
                                  {0: ([m.obj, m.c1, m.c2], [m.x, m.y])})

        self.assertEqual(correct_community_maps, test_results)

    def test_communities_3(self):
        m_class = LP_unbounded()
        m_class._generate_model()
        model = m = m_class.model

        test_results = _collect_test_results(model)

        correct_community_maps = (
            {0: ([], [m.x]), 1: ([], [m.y])}, {0: ([], [m.x]), 1: ([], [m.y])}, {0: ([m.o], [m.x, m.y])},
            {0: ([m.o], [m.x, m.y])}, {}, {}, {0: ([m.o], [m.x, m.y])}, {0: ([m.o], [m.x, m.y])},
            {0: ([], [m.x]), 1: ([], [m.y])}, {0: ([], [m.x]), 1: ([], [m.y])}, {0: ([m.o], [m.x, m.y])},
            {0: ([m.o], [m.x, m.y])})

        self.assertEqual(correct_community_maps, test_results)

    def test_communities_4(self):
        m_class = SOS1_simple()
        m_class._generate_model()
        model = m = m_class.model

        test_results = _collect_test_results(model)

        correct_community_maps = ({0: ([m.c2], [m.x]), 1: ([m.c4], [m.y[1]]), 2: ([m.c1], [m.y[2]])},
                                  {0: ([m.c2], [m.x]), 1: ([m.c4], [m.y[1]]), 2: ([m.c1], [m.y[2]])},
                                  {0: ([m.obj, m.c2], [m.x]), 1: ([m.c4], [m.y[1]]), 2: ([m.c1], [m.y[2]])},
                                  {0: ([m.obj, m.c2], [m.x]), 1: ([m.c4], [m.y[1]]), 2: ([m.c1], [m.y[2]])},
                                  {0: ([m.c1, m.c4], [m.y[1], m.y[2]]), 1: ([m.c2], [m.x])},
                                  {0: ([m.c1, m.c4], [m.y[1], m.y[2]]), 1: ([m.c2], [m.x])},
                                  {0: ([m.c1, m.c4], [m.y[1], m.y[2]]), 1: ([m.obj, m.c2], [m.x, m.y[1], m.y[2]])},
                                  {0: ([m.obj, m.c1, m.c2, m.c4], [m.x, m.y[1], m.y[2]])},
                                  {0: ([m.c2], [m.x]), 1: ([m.c1, m.c4], [m.y[1], m.y[2]])},
                                  {0: ([m.c2], [m.x]), 1: ([m.c1, m.c4], [m.y[1], m.y[2]])},
                                  {0: ([m.obj, m.c1, m.c2, m.c4], [m.x, m.y[1], m.y[2]])},
                                  {0: ([m.obj, m.c1, m.c2, m.c4], [m.x, m.y[1], m.y[2]])})

        self.assertEqual(correct_community_maps, test_results)

    def test_communities_5(self):
        model = m = create_model_5()

        test_results = _collect_test_results(model)

        correct_community_maps = ({0: ([m.c1, m.c2, m.c3, m.c4, m.c5], [m.i1, m.i2, m.i3, m.i4, m.i5, m.i6])},
                                  {0: ([m.c1, m.c2, m.c3, m.c4, m.c5], [m.i1, m.i2, m.i3, m.i4, m.i5, m.i6])},
                                  {0: ([m.c3], [m.i1]), 1: ([m.obj], [m.i2]), 2: ([m.c1], [m.i3]), 3: ([m.c4], [m.i4]),
                                   4: ([m.c2], [m.i5]), 5: ([m.c5], [m.i6])},
                                  {0: ([m.c3], [m.i1]), 1: ([m.obj], [m.i2]), 2: ([m.c1], [m.i3]), 3: ([m.c4], [m.i4]),
                                   4: ([m.c2], [m.i5]), 5: ([m.c5], [m.i6])},
                                  {0: ([m.c1, m.c2, m.c3, m.c4, m.c5], [m.i1, m.i2, m.i3, m.i4, m.i5, m.i6])},
                                  {0: ([m.c1, m.c2, m.c3, m.c4, m.c5], [m.i1, m.i2, m.i3, m.i4, m.i5, m.i6])},
                                  {0: ([m.obj, m.c1, m.c2, m.c3, m.c4, m.c5], [m.i1, m.i2, m.i3, m.i4, m.i5, m.i6])},
                                  {0: ([m.obj, m.c1, m.c2, m.c3, m.c4, m.c5], [m.i1, m.i2, m.i3, m.i4, m.i5, m.i6])},
                                  {0: ([m.c1, m.c2, m.c3, m.c4, m.c5], [m.i1, m.i2, m.i3, m.i4, m.i5, m.i6])},
                                  {0: ([m.c1, m.c2, m.c3, m.c4, m.c5], [m.i1, m.i2, m.i3, m.i4, m.i5, m.i6])},
                                  {0: ([m.obj, m.c1, m.c2, m.c3, m.c4, m.c5], [m.i1, m.i2, m.i3, m.i4, m.i5, m.i6])},
                                  {0: ([m.obj, m.c1, m.c2, m.c3, m.c4, m.c5], [m.i1, m.i2, m.i3, m.i4, m.i5, m.i6])})

        self.assertEqual(correct_community_maps, test_results)

    def test_communities_6(self):
        model = m = create_model_6()

        test_results = _collect_test_results(model, with_string_tests=True)

        correct_community_maps = (
            {0: ([m.c1], [m.x1, m.x2]), 1: ([m.c2], [m.x3, m.x4])},
            {0: ([m.c1], [m.x1, m.x2]), 1: ([m.c2], [m.x3, m.x4])},
            {0: ([m.obj, m.c1], [m.x1, m.x2]), 1: ([m.c2], [m.x3, m.x4])},
            {0: ([m.obj, m.c1], [m.x1, m.x2]), 1: ([m.c2], [m.x3, m.x4])},
            {0: ([m.c1], [m.x1, m.x2]), 1: ([m.c2], [m.x3, m.x4])},
            {0: ([m.c1], [m.x1, m.x2]), 1: ([m.c2], [m.x3, m.x4])},
            {0: ([m.obj, m.c1], [m.x1, m.x2]), 1: ([m.c2], [m.x3, m.x4])},
            {0: ([m.obj, m.c1], [m.x1, m.x2]), 1: ([m.c2], [m.x3, m.x4])},
            {0: ([m.c1], [m.x1, m.x2]), 1: ([m.c2], [m.x3, m.x4])},
            {0: ([m.c1], [m.x1, m.x2]), 1: ([m.c2], [m.x3, m.x4])},
            {0: ([m.obj, m.c1], [m.x1, m.x2]), 1: ([m.c2], [m.x3, m.x4])},
            {0: ([m.obj, m.c1], [m.x1, m.x2]), 1: ([m.c2], [m.x3, m.x4])},
            {0: (['c1'], ['x1', 'x2']), 1: (['c2'], ['x3', 'x4'])},
            {0: (['c1'], ['x1', 'x2']), 1: (['c2'], ['x3', 'x4'])},
            {0: (['obj', 'c1'], ['x1', 'x2']), 1: (['c2'], ['x3', 'x4'])},
            {0: (['obj', 'c1'], ['x1', 'x2']), 1: (['c2'], ['x3', 'x4'])},
            {0: (['c1'], ['x1', 'x2']), 1: (['c2'], ['x3', 'x4'])},
            {0: (['c1'], ['x1', 'x2']), 1: (['c2'], ['x3', 'x4'])},
            {0: (['obj', 'c1'], ['x1', 'x2']), 1: (['c2'], ['x3', 'x4'])},
            {0: (['obj', 'c1'], ['x1', 'x2']), 1: (['c2'], ['x3', 'x4'])},
            {0: (['c1'], ['x1', 'x2']), 1: (['c2'], ['x3', 'x4'])},
            {0: (['c1'], ['x1', 'x2']), 1: (['c2'], ['x3', 'x4'])},
            {0: (['obj', 'c1'], ['x1', 'x2']), 1: (['c2'], ['x3', 'x4'])},
            {0: (['obj', 'c1'], ['x1', 'x2']), 1: (['c2'], ['x3', 'x4'])})

        self.assertEqual(correct_community_maps, test_results)

    def test_communities_7(self):
        model = m = disconnected_model()

        test_results = _collect_test_results(model)

        correct_community_maps = ({0: ([m.c1], [m.x1]), 1: ([], [m.x2])}, {0: ([m.c1], [m.x1]), 1: ([], [m.x2])},
                                  {0: ([m.obj, m.c1], [m.x1]), 1: ([], [m.x2]), 2: ([m.OBJ], [])},
                                  {0: ([m.obj, m.c1], [m.x1]), 1: ([], [m.x2]), 2: ([m.OBJ], [])},
                                  {0: ([m.c1], [m.x1])}, {0: ([m.c1], [m.x1])},
                                  {0: ([m.obj, m.c1], [m.x1]), 1: ([m.OBJ], [])},
                                  {0: ([m.obj, m.c1], [m.x1]), 1: ([m.OBJ], [])},
                                  {0: ([m.c1], [m.x1]), 1: ([], [m.x2])}, {0: ([m.c1], [m.x1]), 1: ([], [m.x2])},
                                  {0: ([m.obj, m.c1], [m.x1]), 1: ([], [m.x2])},
                                  {0: ([m.obj, m.c1], [m.x1]), 1: ([], [m.x2])})

        self.assertEqual(correct_community_maps, test_results)

    def test_decode_1(self):
        model = m = decode_model_1()

        test_results = _collect_test_results(model)

        correct_community_maps = ({0: ([m.c1, m.c2], [m.x1, m.x2]), 1: ([m.c3, m.c4, m.c5], [m.x3, m.x4])},
                                  {0: ([m.c1, m.c2], [m.x1, m.x2]), 1: ([m.c3, m.c4, m.c5], [m.x3, m.x4])},
                                  {0: ([m.c1, m.c2], [m.x1, m.x2]), 1: ([m.c3, m.c4, m.c5], [m.x3, m.x4])},
                                  {0: ([m.c1, m.c2], [m.x1, m.x2]), 1: ([m.c3, m.c4, m.c5], [m.x3, m.x4])},
                                  {0: ([m.c1, m.c2], [m.x1, m.x2]), 1: ([m.c3, m.c4, m.c5], [m.x2, m.x3, m.x4])},
                                  {0: ([m.c1, m.c2], [m.x1, m.x2]), 1: ([m.c3, m.c4, m.c5], [m.x2, m.x3, m.x4])},
                                  {0: ([m.c1, m.c2], [m.x1, m.x2]), 1: ([m.c3, m.c4, m.c5], [m.x2, m.x3, m.x4])},
                                  {0: ([m.c1, m.c2], [m.x1, m.x2]), 1: ([m.c3, m.c4, m.c5], [m.x2, m.x3, m.x4])},
                                  {0: ([m.c1, m.c2, m.c3], [m.x1, m.x2]), 1: ([m.c3, m.c4, m.c5], [m.x3, m.x4])},
                                  {0: ([m.c1, m.c2, m.c3], [m.x1, m.x2]), 1: ([m.c3, m.c4, m.c5], [m.x3, m.x4])},
                                  {0: ([m.c1, m.c2, m.c3], [m.x1, m.x2]), 1: ([m.c3, m.c4, m.c5], [m.x3, m.x4])},
                                  {0: ([m.c1, m.c2, m.c3], [m.x1, m.x2]), 1: ([m.c3, m.c4, m.c5], [m.x3, m.x4])})

        self.assertEqual(correct_community_maps, test_results)

    def test_decode_2(self):
        model = m = decode_model_2()

        test_results = _collect_test_results(model)

        correct_community_maps = (
            {0: ([m.c1, m.c2], [m.x[1], m.x[2], m.x[3]]),
             1: ([m.c3, m.c4, m.c5, m.c6], [m.x[4], m.x[5], m.x[6], m.x[7]])},
            {0: ([m.c1, m.c2], [m.x[1], m.x[2], m.x[3]]),
             1: ([m.c3, m.c4, m.c5, m.c6], [m.x[4], m.x[5], m.x[6], m.x[7]])},
            {0: ([m.c1, m.c2], [m.x[1], m.x[2], m.x[3]]),
             1: ([m.c3, m.c4, m.c5, m.c6], [m.x[4], m.x[5], m.x[6], m.x[7]])},
            {0: ([m.c1, m.c2], [m.x[1], m.x[2], m.x[3]]),
             1: ([m.c3, m.c4, m.c5, m.c6], [m.x[4], m.x[5], m.x[6], m.x[7]])},
            {0: ([m.c1, m.c2, m.c3], [m.x[1], m.x[2], m.x[3], m.x[4], m.x[5]]),
             1: ([m.c4, m.c5, m.c6], [m.x[4], m.x[5], m.x[6], m.x[7]])},
            {0: ([m.c1, m.c2], [m.x[1], m.x[2], m.x[3]]), 1: (
                [m.c3, m.c4, m.c5, m.c6], [m.x[3], m.x[4], m.x[5], m.x[6], m.x[7]])},
            {0: ([m.c1, m.c2, m.c3], [m.x[1], m.x[2], m.x[3], m.x[4], m.x[5]]),
             1: ([m.c4, m.c5, m.c6], [m.x[4], m.x[5], m.x[6], m.x[7]])},
            {0: ([m.c1, m.c2], [m.x[1], m.x[2], m.x[3]]), 1: (
                [m.c3, m.c4, m.c5, m.c6], [m.x[3], m.x[4], m.x[5], m.x[6], m.x[7]])},
            {0: ([m.c1, m.c2, m.c3], [m.x[1], m.x[2], m.x[3]]),
             1: ([m.c3, m.c4, m.c5, m.c6], [m.x[4], m.x[5], m.x[6], m.x[7]])},
            {0: ([m.c1, m.c2, m.c3], [m.x[1], m.x[2], m.x[3]]),
             1: ([m.c3, m.c4, m.c5, m.c6], [m.x[4], m.x[5], m.x[6], m.x[7]])},
            {0: ([m.c1, m.c2, m.c3], [m.x[1], m.x[2], m.x[3]]),
             1: ([m.c3, m.c4, m.c5, m.c6], [m.x[4], m.x[5], m.x[6], m.x[7]])},
            {0: ([m.c1, m.c2, m.c3], [m.x[1], m.x[2], m.x[3]]),
             1: ([m.c3, m.c4, m.c5, m.c6], [m.x[4], m.x[5], m.x[6], m.x[7]])})

        self.assertEqual(correct_community_maps, test_results)

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
        with self.assertRaisesRegex(AssertionError, "Invalid model: 'model=%s' - model must be an instance of "
                                                    "ConcreteModel" % model):
            detect_communities(model)
        model = create_model_6()

        type_of_community_map = 'foo'
        with self.assertRaisesRegex(AssertionError, "Invalid value for type_of_community_map: "
                                                    "'type_of_community_map=%s' - Valid values: 'b', 'c', 'v'"
                                                    % type_of_community_map):
            detect_communities(model, type_of_community_map=type_of_community_map)

        with_objective = 'foo'
        with self.assertRaisesRegex(AssertionError, "Invalid value for with_objective: 'with_objective=%s' - "
                                                    "with_objective must be a Boolean" % with_objective):
            detect_communities(model, with_objective=with_objective)

        weighted_graph = 'foo'
        with self.assertRaisesRegex(AssertionError, "Invalid value for weighted_graph: 'weighted_graph=%s' - "
                                                    "weighted_graph must be a Boolean" % weighted_graph):
            detect_communities(model, weighted_graph=weighted_graph)

        random_seed = 'foo'
        with self.assertRaisesRegex(AssertionError, "Invalid value for random_seed: 'random_seed=%s' - random_seed "
                                                    "must be a non-negative integer" % random_seed):
            detect_communities(model, random_seed=random_seed)

    def test_visualize_model_graph_1(self):
        model = decode_model_1()
        seed_value = 5

        fig, pos = visualize_model_graph(model, random_seed=seed_value)
        correct_pos_dict_length = 5

        self.assertTrue(isinstance(pos, dict))
        self.assertEqual(len(pos), correct_pos_dict_length)

    def test_visualize_model_graph_2(self):
        model = decode_model_2()
        seed_value = 5

        community_map = detect_communities(model)

        fig, pos = visualize_model_graph(model, community_map=community_map, type_of_graph='b', random_seed=seed_value)
        correct_pos_dict_length = 13

        self.assertTrue(isinstance(pos, dict))
        self.assertEqual(len(pos), correct_pos_dict_length)


def _collect_test_results(model, with_string_tests=False):
    random_seed_test = 5

    community_map_b_unweighted_without = detect_communities(model, type_of_community_map='b', with_objective=False,
                                                            weighted_graph=False, random_seed=random_seed_test)
    community_map_b_weighted_without = detect_communities(model, type_of_community_map='b', with_objective=False,
                                                          weighted_graph=True, random_seed=random_seed_test)
    community_map_b_unweighted_with = detect_communities(model, type_of_community_map='b', with_objective=True,
                                                         weighted_graph=False, random_seed=random_seed_test)
    community_map_b_weighted_with = detect_communities(model, type_of_community_map='b', with_objective=True,
                                                       weighted_graph=True, random_seed=random_seed_test)
    community_map_c_unweighted_without = detect_communities(model, type_of_community_map='c', with_objective=False,
                                                            weighted_graph=False, random_seed=random_seed_test)
    community_map_c_weighted_without = detect_communities(model, type_of_community_map='c', with_objective=False,
                                                          weighted_graph=True, random_seed=random_seed_test)
    community_map_c_unweighted_with = detect_communities(model, type_of_community_map='c', with_objective=True,
                                                         weighted_graph=False, random_seed=random_seed_test)
    community_map_c_weighted_with = detect_communities(model, type_of_community_map='c', with_objective=True,
                                                       weighted_graph=True, random_seed=random_seed_test)
    community_map_v_unweighted_without = detect_communities(model, type_of_community_map='v', with_objective=False,
                                                            weighted_graph=False, random_seed=random_seed_test)
    community_map_v_weighted_without = detect_communities(model, type_of_community_map='v', with_objective=False,
                                                          weighted_graph=True, random_seed=random_seed_test)
    community_map_v_unweighted_with = detect_communities(model, type_of_community_map='v', with_objective=True,
                                                         weighted_graph=False, random_seed=random_seed_test)
    community_map_v_weighted_with = detect_communities(model, type_of_community_map='v', with_objective=True,
                                                       weighted_graph=True, random_seed=random_seed_test)

    test_results = (community_map_b_unweighted_without,
                    community_map_b_weighted_without,
                    community_map_b_unweighted_with,
                    community_map_b_weighted_with,
                    community_map_c_unweighted_without,
                    community_map_c_weighted_without,
                    community_map_c_unweighted_with,
                    community_map_c_weighted_with,
                    community_map_v_unweighted_without,
                    community_map_v_weighted_without,
                    community_map_v_unweighted_with,
                    community_map_v_weighted_with)

    if not with_string_tests:
        return test_results

    str_community_map_b_unweighted_without = stringify_community_map(model, type_of_community_map='b',
                                                                     with_objective=False,
                                                                     weighted_graph=False, random_seed=random_seed_test)
    str_community_map_b_weighted_without = stringify_community_map(model, type_of_community_map='b',
                                                                   with_objective=False,
                                                                   weighted_graph=True, random_seed=random_seed_test)
    str_community_map_b_unweighted_with = stringify_community_map(model, type_of_community_map='b', with_objective=True,
                                                                  weighted_graph=False, random_seed=random_seed_test)
    str_community_map_b_weighted_with = stringify_community_map(model, type_of_community_map='b', with_objective=True,
                                                                weighted_graph=True, random_seed=random_seed_test)
    str_community_map_c_unweighted_without = stringify_community_map(model, type_of_community_map='c',
                                                                     with_objective=False,
                                                                     weighted_graph=False, random_seed=random_seed_test)
    str_community_map_c_weighted_without = stringify_community_map(model, type_of_community_map='c',
                                                                   with_objective=False,
                                                                   weighted_graph=True, random_seed=random_seed_test)
    str_community_map_c_unweighted_with = stringify_community_map(model, type_of_community_map='c', with_objective=True,
                                                                  weighted_graph=False, random_seed=random_seed_test)
    str_community_map_c_weighted_with = stringify_community_map(model, type_of_community_map='c', with_objective=True,
                                                                weighted_graph=True, random_seed=random_seed_test)
    str_community_map_v_unweighted_without = stringify_community_map(model, type_of_community_map='v',
                                                                     with_objective=False,
                                                                     weighted_graph=False, random_seed=random_seed_test)
    str_community_map_v_weighted_without = stringify_community_map(model, type_of_community_map='v',
                                                                   with_objective=False,
                                                                   weighted_graph=True, random_seed=random_seed_test)
    str_community_map_v_unweighted_with = stringify_community_map(model, type_of_community_map='v', with_objective=True,
                                                                  weighted_graph=False, random_seed=random_seed_test)
    str_community_map_v_weighted_with = stringify_community_map(model, type_of_community_map='v', with_objective=True,
                                                                weighted_graph=True, random_seed=random_seed_test)

    str_test_results = (str_community_map_b_unweighted_without,
                        str_community_map_b_weighted_without,
                        str_community_map_b_unweighted_with,
                        str_community_map_b_weighted_with,
                        str_community_map_c_unweighted_without,
                        str_community_map_c_weighted_without,
                        str_community_map_c_unweighted_with,
                        str_community_map_c_weighted_with,
                        str_community_map_v_unweighted_without,
                        str_community_map_v_weighted_without,
                        str_community_map_v_unweighted_with,
                        str_community_map_v_weighted_with)

    return test_results + str_test_results


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
