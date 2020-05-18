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
import pyutilib.th as unittest
from pyutilib.misc import import_file

from pyomo.environ import *
from pyomo.contrib.community_detection.detection import *
from pyomo.core import ConcreteModel
from pyomo.solvers.tests.models.LP_unbounded import LP_unbounded
from pyomo.solvers.tests.models.QCP_simple import QCP_simple
from pyomo.solvers.tests.models.MIQCP_simple import MIQCP_simple
from pyomo.solvers.tests.models.MILP_simple import MILP_simple
from pyomo.solvers.tests.models.SOS1_simple import SOS1_simple
from pyomo.core.expr.current import identify_variables


class TestDecomposition(unittest.TestCase):

    """
        def test_communities_1(self):
            model = LP_unbounded()._generate_model

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

            results = (community_map_v_unweighted_without,
                       community_map_v_weighted_without,
                       community_map_v_unweighted_with,
                       community_map_v_weighted_with,
                       community_map_c_unweighted_without,
                       community_map_c_weighted_without,
                       community_map_c_unweighted_with,
                       community_map_c_weighted_with)

            correct_community_maps =
    """

    def test_communities_1(self):
        # First test case
        models_location = 'D:\College\Sophomore Year\PSE Research\Current Work\Relevant\Rewritten Models'
        file = 'ball_mk2_10.py'
        exfile = import_file(models_location + '\\' + file)
        model = exfile.create_model()

        results_with_obj = (detect_communities(model, 'v'), detect_communities(model, 'c'))
        correct_dicts_with_obj = ({0: ['i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8', 'i9', 'i10', 'i11']},
                                  {0: ['c2', 'obj']})
        self.assertEqual(correct_dicts_with_obj, results_with_obj)

        results_without_obj = (detect_communities(model, 'v', False), detect_communities(model, 'c', False))
        correct_dicts_without_obj = ({0: ['i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8', 'i9', 'i10', 'i11']}, {0: ['c2']})
        self.assertEqual(correct_dicts_without_obj, results_without_obj)


    def test_communities_2(self):
        # Fourth test case
        models_location = 'D:\College\Sophomore Year\PSE Research\Current Work\Relevant\Rewritten Models'
        file = 'gbd.py'
        exfile = import_file(models_location + '\\' + file)
        model = exfile.create_model()

        results_with_obj = (detect_communities(model, 'v'), detect_communities(model, 'c'))
        correct_dicts_with_obj = ({0: ['x2', 'b3', 'b4', 'b5']}, {0: ['c2', 'c3', 'c4', 'c5', 'obj']})
        self.assertEqual(correct_dicts_with_obj, results_with_obj)

        results_without_obj = (detect_communities(model, 'v', False), detect_communities(model, 'c', False))
        correct_dicts_without_obj = ({0: ['x2', 'b3', 'b4', 'b5']}, {0: ['c2', 'c3', 'c4', 'c5']})
        self.assertEqual(correct_dicts_without_obj, results_without_obj)

    def test_communities_3(self):
        # Third test case
        models_location = 'D:\College\Sophomore Year\PSE Research\Current Work\Relevant\Rewritten Models'
        file = 'ball_mk3_30.py'
        exfile = import_file(models_location + '\\' + file)
        model = exfile.create_model()

        results_with_obj = (detect_communities(model, 'v'), detect_communities(model, 'c'))
        correct_dicts_with_obj = ({0: ['i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8', 'i9', 'i10', 'i11', 'i12', 'i13',
                                       'i14', 'i15', 'i16', 'i17', 'i18', 'i19', 'i20', 'i21', 'i22', 'i23', 'i24',
                                       'i25', 'i26', 'i27', 'i28', 'i29', 'i30', 'i31']}, {0: ['c2', 'obj']})
        self.assertEqual(correct_dicts_with_obj, results_with_obj)

        results_without_obj = (detect_communities(model, 'v', False), detect_communities(model, 'c', False))
        correct_dicts_without_obj = ({0: ['i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8', 'i9', 'i10', 'i11', 'i12', 'i13',
                                          'i14', 'i15', 'i16', 'i17', 'i18', 'i19', 'i20', 'i21', 'i22', 'i23', 'i24',
                                          'i25', 'i26', 'i27', 'i28', 'i29', 'i30', 'i31']}, {0: ['c2']})
        self.assertEqual(correct_dicts_without_obj, results_without_obj)

    def test_communities_4(self):
        # Second test case
        model = create_model_4()

        results_with_obj = (detect_communities(model, 'v'), detect_communities(model, 'c'))
        correct_dicts_with_obj = ({0: ['i1', 'i2', 'i3', 'i4', 'i5', 'i6']}, {0: ['c1', 'c2', 'c3', 'c4', 'c5', 'obj']})
        self.assertEqual(correct_dicts_with_obj, results_with_obj)

        results_without_obj = (detect_communities(model, 'v', False), detect_communities(model, 'c', False))
        correct_dicts_without_obj = ({0: ['i1', 'i2', 'i3', 'i4', 'i5', 'i6']}, {0: ['c1', 'c2', 'c3', 'c4', 'c5']})
        self.assertEqual(correct_dicts_without_obj, results_without_obj)

    def test_communities_5(self):
        # Fifth test case
        model = create_model_5()
        results_with_obj = (detect_communities(model, 'v'), detect_communities(model, 'c'))
        correct_dicts_with_obj = ({0: ['x1', 'x2'], 1: ['x3', 'x4']}, {0: ['c1', 'obj'], 1: ['c2']})
        self.assertEqual(correct_dicts_with_obj, results_with_obj)

        results_without_obj = (detect_communities(model, 'v', False), detect_communities(model, 'c', False))
        correct_dicts_without_obj = ({0: ['x1', 'x2'], 1: ['x3', 'x4']}, {0: ['c1'], 1: ['c2']})
        self.assertEqual(correct_dicts_without_obj, results_without_obj)


def create_model_4():  # MINLP written by GAMS Convert at 05/10/19 14:22:56
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


def create_model_5():
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
