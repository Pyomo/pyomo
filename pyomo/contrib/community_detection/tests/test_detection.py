"""Community Detection Test File - Rahul Joglekar"""

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
from pyomo.environ import *
from pyomo.contrib.community_detection.detection import *
from pyomo.core import ConcreteModel
from pyomo.solvers.tests.models.LP_unbounded import LP_unbounded
from pyomo.solvers.tests.models.QP_simple import QP_simple
from pyomo.solvers.tests.models.LP_inactive_index import LP_inactive_index
from pyomo.solvers.tests.models.SOS1_simple import SOS1_simple
import pyutilib.th as unittest


class TestDecomposition(unittest.TestCase):

    def test_communities_1(self):
        m_class = LP_unbounded()
        m_class._generate_model()
        model = m_class.model

        community_map_v_unweighted_without = detect_communities(model, node_type='v', with_objective=False,
                                                                weighted_graph=False, random_seed=5)
        community_map_v_weighted_without = detect_communities(model, node_type='v', with_objective=False,
                                                              weighted_graph=True, random_seed=5)
        community_map_v_unweighted_with = detect_communities(model, node_type='v', with_objective=True,
                                                             weighted_graph=False, random_seed=5)
        community_map_v_weighted_with = detect_communities(model, node_type='v', with_objective=True,
                                                           weighted_graph=True, random_seed=5)
        community_map_c_unweighted_without = detect_communities(model, node_type='c', with_objective=False,
                                                                weighted_graph=False, random_seed=5)
        community_map_c_weighted_without = detect_communities(model, node_type='c', with_objective=False,
                                                              weighted_graph=True, random_seed=5)
        community_map_c_unweighted_with = detect_communities(model, node_type='c', with_objective=True,
                                                             weighted_graph=False, random_seed=5)
        community_map_c_weighted_with = detect_communities(model, node_type='c', with_objective=True,
                                                           weighted_graph=True, random_seed=5)

        test_results = (community_map_v_unweighted_without,
                        community_map_v_weighted_without,
                        community_map_v_unweighted_with,
                        community_map_v_weighted_with,
                        community_map_c_unweighted_without,
                        community_map_c_weighted_without,
                        community_map_c_unweighted_with,
                        community_map_c_weighted_with)

        correct_community_maps = ({0: ['x'], 1: ['y']}, {0: ['x'], 1: ['y']}, {0: ['x', 'y']}, {0: ['x', 'y']},
                                  {}, {}, {0: ['o']}, {0: ['o']})

        self.assertEqual(correct_community_maps, test_results)

    def test_communities_2(self):
        m_class = QP_simple()
        m_class._generate_model()
        model = m_class.model

        community_map_v_unweighted_without = detect_communities(model, node_type='v', with_objective=False,
                                                                weighted_graph=False, random_seed=5)
        community_map_v_weighted_without = detect_communities(model, node_type='v', with_objective=False,
                                                              weighted_graph=True, random_seed=5)
        community_map_v_unweighted_with = detect_communities(model, node_type='v', with_objective=True,
                                                             weighted_graph=False, random_seed=5)
        community_map_v_weighted_with = detect_communities(model, node_type='v', with_objective=True,
                                                           weighted_graph=True, random_seed=5)
        community_map_c_unweighted_without = detect_communities(model, node_type='c', with_objective=False,
                                                                weighted_graph=False, random_seed=5)
        community_map_c_weighted_without = detect_communities(model, node_type='c', with_objective=False,
                                                              weighted_graph=True, random_seed=5)
        community_map_c_unweighted_with = detect_communities(model, node_type='c', with_objective=True,
                                                             weighted_graph=False, random_seed=5)
        community_map_c_weighted_with = detect_communities(model, node_type='c', with_objective=True,
                                                           weighted_graph=True, random_seed=5)

        test_results = (community_map_v_unweighted_without,
                        community_map_v_weighted_without,
                        community_map_v_unweighted_with,
                        community_map_v_weighted_with,
                        community_map_c_unweighted_without,
                        community_map_c_weighted_without,
                        community_map_c_unweighted_with,
                        community_map_c_weighted_with)

        correct_community_maps = ({0: ['x', 'y']}, {0: ['x', 'y']}, {0: ['x', 'y']}, {0: ['x', 'y']}, {0: ['c1', 'c2']},
                                  {0: ['c1', 'c2']}, {0: ['c1', 'c2', 'inactive_obj', 'obj']},
                                  {0: ['c1', 'c2', 'inactive_obj', 'obj']})

        self.assertEqual(correct_community_maps, test_results)

    def test_communities_3(self):
        m_class = LP_inactive_index()
        m_class._generate_model()
        model = m_class.model

        community_map_v_unweighted_without = detect_communities(model, node_type='v', with_objective=False,
                                                                weighted_graph=False, random_seed=5)
        community_map_v_weighted_without = detect_communities(model, node_type='v', with_objective=False,
                                                              weighted_graph=True, random_seed=5)
        community_map_v_unweighted_with = detect_communities(model, node_type='v', with_objective=True,
                                                             weighted_graph=False, random_seed=5)
        community_map_v_weighted_with = detect_communities(model, node_type='v', with_objective=True,
                                                           weighted_graph=True, random_seed=5)
        community_map_c_unweighted_without = detect_communities(model, node_type='c', with_objective=False,
                                                                weighted_graph=False, random_seed=5)
        community_map_c_weighted_without = detect_communities(model, node_type='c', with_objective=False,
                                                              weighted_graph=True, random_seed=5)
        community_map_c_unweighted_with = detect_communities(model, node_type='c', with_objective=True,
                                                             weighted_graph=False, random_seed=5)
        community_map_c_weighted_with = detect_communities(model, node_type='c', with_objective=True,
                                                           weighted_graph=True, random_seed=5)

        test_results = (community_map_v_unweighted_without,
                        community_map_v_weighted_without,
                        community_map_v_unweighted_with,
                        community_map_v_weighted_with,
                        community_map_c_unweighted_without,
                        community_map_c_weighted_without,
                        community_map_c_unweighted_with,
                        community_map_c_weighted_with)

        correct_community_maps = ({0: ['x'], 1: ['y'], 2: ['z']}, {0: ['x'], 1: ['y'], 2: ['z']},
                                  {0: ['x', 'y', 'z']}, {0: ['x', 'y', 'z']},
                                  {0: ['c1[1]', 'c1[2]', 'c2[2]'], 1: ['c1[3]', 'c1[4]', 'c2[1]'],
                                   2: ['b.c', 'B[1].c', 'B[2].c']},
                                  {0: ['c1[1]', 'c1[2]', 'c2[2]'], 1: ['c1[3]', 'c1[4]', 'c2[1]'],
                                   2: ['b.c', 'B[1].c', 'B[2].c']},
                                  {0: ['c1[1]', 'c1[2]', 'c2[2]', 'obj[1]', 'OBJ'], 1: ['c1[3]', 'c1[4]', 'c2[1]'],
                                   2: ['b.c', 'B[1].c', 'B[2].c', 'obj[2]']},
                                  {0: ['c1[1]', 'c1[2]', 'c2[2]', 'obj[1]', 'OBJ'], 1: ['c1[3]', 'c1[4]', 'c2[1]'],
                                   2: ['b.c', 'B[1].c', 'B[2].c', 'obj[2]']})

        self.assertEqual(correct_community_maps, test_results)

    def test_communities_4(self):
        m_class = SOS1_simple()
        m_class._generate_model()
        model = m_class.model

        community_map_v_unweighted_without = detect_communities(model, node_type='v', with_objective=False,
                                                                weighted_graph=False, random_seed=5)
        community_map_v_weighted_without = detect_communities(model, node_type='v', with_objective=False,
                                                              weighted_graph=True, random_seed=5)
        community_map_v_unweighted_with = detect_communities(model, node_type='v', with_objective=True,
                                                             weighted_graph=False, random_seed=5)
        community_map_v_weighted_with = detect_communities(model, node_type='v', with_objective=True,
                                                           weighted_graph=True, random_seed=5)
        community_map_c_unweighted_without = detect_communities(model, node_type='c', with_objective=False,
                                                                weighted_graph=False, random_seed=5)
        community_map_c_weighted_without = detect_communities(model, node_type='c', with_objective=False,
                                                              weighted_graph=True, random_seed=5)
        community_map_c_unweighted_with = detect_communities(model, node_type='c', with_objective=True,
                                                             weighted_graph=False, random_seed=5)
        community_map_c_weighted_with = detect_communities(model, node_type='c', with_objective=True,
                                                           weighted_graph=True, random_seed=5)

        test_results = (community_map_v_unweighted_without,
                        community_map_v_weighted_without,
                        community_map_v_unweighted_with,
                        community_map_v_weighted_with,
                        community_map_c_unweighted_without,
                        community_map_c_weighted_without,
                        community_map_c_unweighted_with,
                        community_map_c_weighted_with)

        correct_community_maps = ({0: ['x'], 1: ['y[1]', 'y[2]']}, {0: ['x'], 1: ['y[1]', 'y[2]']},
                                  {0: ['x', 'y[1]', 'y[2]']}, {0: ['x', 'y[1]', 'y[2]']}, {0: ['c1', 'c4'], 1: ['c2']},
                                  {0: ['c1', 'c4'], 1: ['c2']}, {0: ['c1', 'c4'], 1: ['c2', 'obj']},
                                  {0: ['c1', 'c2', 'c4', 'obj']})

        self.assertEqual(correct_community_maps, test_results)

    def test_communities_5(self):
        model = create_model_5()

        community_map_v_unweighted_without = detect_communities(model, node_type='v', with_objective=False,
                                                                weighted_graph=False, random_seed=5)
        community_map_v_weighted_without = detect_communities(model, node_type='v', with_objective=False,
                                                              weighted_graph=True, random_seed=5)
        community_map_v_unweighted_with = detect_communities(model, node_type='v', with_objective=True,
                                                             weighted_graph=False, random_seed=5)
        community_map_v_weighted_with = detect_communities(model, node_type='v', with_objective=True,
                                                           weighted_graph=True, random_seed=5)
        community_map_c_unweighted_without = detect_communities(model, node_type='c', with_objective=False,
                                                                weighted_graph=False, random_seed=5)
        community_map_c_weighted_without = detect_communities(model, node_type='c', with_objective=False,
                                                              weighted_graph=True, random_seed=5)
        community_map_c_unweighted_with = detect_communities(model, node_type='c', with_objective=True,
                                                             weighted_graph=False, random_seed=5)
        community_map_c_weighted_with = detect_communities(model, node_type='c', with_objective=True,
                                                           weighted_graph=True, random_seed=5)

        test_results = (community_map_v_unweighted_without,
                        community_map_v_weighted_without,
                        community_map_v_unweighted_with,
                        community_map_v_weighted_with,
                        community_map_c_unweighted_without,
                        community_map_c_weighted_without,
                        community_map_c_unweighted_with,
                        community_map_c_weighted_with)

        correct_community_maps = ({0: ['i1', 'i2', 'i3', 'i4', 'i5', 'i6']}, {0: ['i1', 'i2', 'i3', 'i4', 'i5', 'i6']},
                                  {0: ['i1', 'i2', 'i3', 'i4', 'i5', 'i6']}, {0: ['i1', 'i2', 'i3', 'i4', 'i5', 'i6']},
                                  {0: ['c1', 'c2', 'c3', 'c4', 'c5']}, {0: ['c1', 'c2', 'c3', 'c4', 'c5']},
                                  {0: ['c1', 'c2', 'c3', 'c4', 'c5', 'obj']},
                                  {0: ['c1', 'c2', 'c3', 'c4', 'c5', 'obj']})

        self.assertEqual(correct_community_maps, test_results)

    def test_communities_6(self):
        model = create_model_6()

        community_map_v_unweighted_without = detect_communities(model, node_type='v', with_objective=False,
                                                                weighted_graph=False, random_seed=5)
        community_map_v_weighted_without = detect_communities(model, node_type='v', with_objective=False,
                                                              weighted_graph=True, random_seed=5)
        community_map_v_unweighted_with = detect_communities(model, node_type='v', with_objective=True,
                                                             weighted_graph=False, random_seed=5)
        community_map_v_weighted_with = detect_communities(model, node_type='v', with_objective=True,
                                                           weighted_graph=True, random_seed=5)
        community_map_c_unweighted_without = detect_communities(model, node_type='c', with_objective=False,
                                                                weighted_graph=False, random_seed=5)
        community_map_c_weighted_without = detect_communities(model, node_type='c', with_objective=False,
                                                              weighted_graph=True, random_seed=5)
        community_map_c_unweighted_with = detect_communities(model, node_type='c', with_objective=True,
                                                             weighted_graph=False, random_seed=5)
        community_map_c_weighted_with = detect_communities(model, node_type='c', with_objective=True,
                                                           weighted_graph=True, random_seed=5)

        test_results = (community_map_v_unweighted_without,
                        community_map_v_weighted_without,
                        community_map_v_unweighted_with,
                        community_map_v_weighted_with,
                        community_map_c_unweighted_without,
                        community_map_c_weighted_without,
                        community_map_c_unweighted_with,
                        community_map_c_weighted_with)

        correct_community_maps = ({0: ['x1', 'x2'], 1: ['x3', 'x4']}, {0: ['x1', 'x2'], 1: ['x3', 'x4']},
                                  {0: ['x1', 'x2'], 1: ['x3', 'x4']}, {0: ['x1', 'x2'], 1: ['x3', 'x4']},
                                  {0: ['c1'], 1: ['c2']}, {0: ['c1'], 1: ['c2']}, {0: ['c1', 'obj'], 1: ['c2']},
                                  {0: ['c1', 'obj'], 1: ['c2']})

        self.assertEqual(correct_community_maps, test_results)


def create_model_5():  # MINLP written by GAMS Convert at 05/10/19 14:22:56
    #
    #  Equation counts
    #      Total        E        G        L        N        X        C        B
    #          6        1        0        5        0        0        0        0
    #
    #  Variable counts
    #                   x        b        i      s1s      s2s       sc       si
    #      Total     cont   binary  integer     sos1     sos2    scont     sint
    #          7        1        0        6        0        0        0        0
    #  FX      0        0        0        0        0        0        0        0
    #
    #  Nonzero counts
    #      Total    const       NL      DLL
    #         37       35        2        0
    #
    #  Reformulation has removed 1 variable and 1 equation
    model = m = ConcreteModel()
    m.i1 = Var(within=Integers, bounds=(0, 1E15), initialize=0)
    m.i2 = Var(within=Integers, bounds=(0, 1E15), initialize=0)
    m.i3 = Var(within=Integers, bounds=(0, 1E15), initialize=0)
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


def create_model_6():
    model = m = ConcreteModel()
    m.x1 = Var(bounds=(0, 1))
    m.x2 = Var(bounds=(0, 1))
    m.x3 = Var(bounds=(0, 1))
    m.x4 = Var(bounds=(0, 1))
    m.obj = Objective(expr=m.x1, sense=minimize)
    m.c1 = Constraint(expr=m.x1 + m.x2 >= 1)
    m.c2 = Constraint(expr=m.x3 + m.x4 >= 1)
    return model


if __name__ == '__main__':
    unittest.main()
