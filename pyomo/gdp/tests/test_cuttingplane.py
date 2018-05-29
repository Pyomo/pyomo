#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyutilib.th as unittest

from pyomo.environ import *
from pyomo.gdp import *
from pyomo.core.base import expr_common, expr as EXPR
import pyomo.opt
import pyomo.gdp.tests.models as models

import random
from six import StringIO

from nose.tools import set_trace

solver = 'gurobi'
solvers = pyomo.opt.check_available_solvers('gurobi')

# TODO:
#     - test that deactivated objectives on the model don't get used by the
#       transformation

class OneVarDisj(unittest.TestCase):
    @staticmethod
    def makeModel():
        m = ConcreteModel()
        m.x = Var(bounds=(0, 10))
        m.disj1 = Disjunct()
        m.disj1.xTrue = Constraint(expr=m.x==1)
        m.disj2 = Disjunct()
        m.disj2.xFalse = Constraint(expr=m.x==0)
        m.disjunction = Disjunction(expr=[m.disj1, m.disj2])
        m.obj = Objective(expr=m.x)
        return m

    # there are no useful cuts here, and so we don't add them!
    def test_no_cuts_added(self):
        m = self.makeModel()

        TransformationFactory('gdp.cuttingplane').apply_to(m)
        cuts = m._pyomo_gdp_cuttingplane_relaxation.cuts
        self.assertEqual(len(cuts), 0)

class TwoTermDisj(unittest.TestCase):
    def setUp(self):
        EXPR.set_expression_tree_format(expr_common.Mode.coopr3_trees)
        # set seed so we can test name collisions predictably
        random.seed(666)

    def tearDown(self):
        EXPR.set_expression_tree_format(expr_common._default_mode)

    @unittest.skipIf('gurobi' not in solvers, "Gurobi solver not available")
    def test_transformation_block(self):
        m = models.makeTwoTermDisj_boxes()
        TransformationFactory('gdp.cuttingplane').apply_to(m)

        # we created the block
        transBlock = m._pyomo_gdp_cuttingplane_relaxation
        self.assertIsInstance(transBlock, Block)
        # the cuts are on it
        cuts = transBlock.cuts
        self.assertIsInstance(cuts, Constraint)
        # this one adds 2 cuts
        # TODO: you could test number of cuts here when you are sure.
        #self.assertEqual(len(cuts), 2)

    # TODO: you could rewrite this, but I think it might be better to just test
    # that the cuts are all valid. I don't really care what the coefficients are
    # as long as they aren't hurting anything...
    # @unittest.skipIf('gurobi' not in solvers, "Gurobi solver not available")
    # def test_cut_constraint(self):
    #     m = models.makeTwoTermDisj_boxes()
    #     TransformationFactory('gdp.cuttingplane').apply_to(m)

    #     cut = m._pyomo_gdp_cuttingplane_relaxation.cuts[0]
    #     self.assertEqual(cut.lower, 0)
    #     self.assertIsNone(cut.upper)

    #     # test body
    #     self.assertEqual(len(cut.body._coef), 4)
    #     self.assertEqual(len(cut.body._args), 4)
    #     self.assertEqual(cut.body._const, 0)
        
    #     coefs = {
    #         0: 0.45,
    #         1: 0.55,
    #         2: 0.1,
    #         3: -0.1
    #     }

    #     xhat = {
    #         0: 2.7,
    #         1: 1.3,
    #         2: 0.85,
    #         3: 0.15
    #     }

    #     variables = {
    #         0: m.x,
    #         1: m.y,
    #         2: m.d[0].indicator_var,
    #         3: m.d[1].indicator_var
    #     }

    #     for i in range(4):
    #         self.assertAlmostEqual(cut.body._coef[i], coefs[i])
    #         self.assertEqual(len(cut.body._args[i]._coef), 1)
    #         self.assertEqual(len(cut.body._args[i]._args), 1)
    #         self.assertAlmostEqual(cut.body._args[i]._const, -1*xhat[i])
    #         self.assertEqual(cut.body._args[i]._coef[0], 1)
    #         self.assertIs(cut.body._args[i]._args[0], variables[i])

    @unittest.skipIf('gurobi' not in solvers, "Gurobi solver not available")
    def test_create_using(self):
        m = models.makeTwoTermDisj_boxes()

        # TODO: this is duplicate code with other transformation tests
        modelcopy = TransformationFactory('gdp.cuttingplane').create_using(m)
        modelcopy_buf = StringIO()
        modelcopy.pprint(ostream=modelcopy_buf)
        modelcopy_output = modelcopy_buf.getvalue()

        TransformationFactory('gdp.cuttingplane').apply_to(m)
        model_buf = StringIO()
        m.pprint(ostream=model_buf)
        model_output = model_buf.getvalue()
        self.maxDiff = None
        self.assertMultiLineEqual(modelcopy_output, model_output)

    @unittest.skipIf('gurobi' not in solvers, "Gurobi solver not available")
    def test_active_objective_err(self):
        m = models.makeTwoTermDisj_boxes()
        m.obj.deactivate()
        self.assertRaisesRegexp(
            GDP_Error,
            "Cannot apply cutting planes transformation without an active "
            "objective in the model*",
            TransformationFactory('gdp.cuttingplane').apply_to,
            m
        )

class Grossmann_TestCases(unittest.TestCase):
    @unittest.skipIf('gurobi' not in solvers, "Gurobi solver not available")
    def test_correct_soln(self):
        m = models.grossmann_oneDisj()
        TransformationFactory('gdp.cuttingplane').apply_to(m)

        # TODO: probably don't want to be solving here in the long term?
        # checking if we get the optimal solution.
        SolverFactory(solver).solve(m)
        self.assertAlmostEqual(m.x.value, 2)
        self.assertAlmostEqual(m.y.value, 10)

    @unittest.skipIf('gurobi' not in solvers, "Gurobi solver not available")
    def test_cuts_valid(self):
        m = models.grossmann_oneDisj()
        TransformationFactory('gdp.cuttingplane').apply_to(m)

        # Constraint 1
        cuts = m._pyomo_gdp_cuttingplane_relaxation.cuts
        # TODO: I'm actually not sure what I expect here right now:
        self.assertEqual(len(cuts), 4)

        cut0 = cuts[0]
        self.assertEqual(cut0.upper, 0)
        self.assertIsNone(cut0.lower)
        cut0_expr = cut0.body
        # we first check that the first cut is tight at the upper righthand
        # corners of the two regions:
        m.x.fix(2)
        m.y.fix(10)
        m.disjunct1.indicator_var.fix(1)
        m.disjunct2.indicator_var.fix(0)
        # As long as this is within MIP tolerance, we are happy:
        self.assertAlmostEqual(value(cut0_expr), 0)

        m.x.fix(10)
        m.y.fix(3)
        m.disjunct1.indicator_var.fix(0)
        m.disjunct2.indicator_var.fix(1)
        self.assertAlmostEqual(value(cut0_expr), 0)

        # Constraint 2
        # now we check that the second cut is tight for the top region:
        cut1 = cuts[1]
        self.assertEqual(cut1.upper, 0)
        self.assertIsNone(cut1.lower)
        cut1_expr = cut1.body
        m.x.fix(2)
        m.y.fix(10)
        m.disjunct1.indicator_var.fix(1)
        m.disjunct2.indicator_var.fix(0)
        self.assertLessEqual(value(cut1_expr), 0)

        m.x.fix(0)
        self.assertLessEqual(value(cut1_expr), 0)

    @unittest.skipIf('gurobi' not in solvers, "Gurobi solver not available")
    def test_cuts_dont_cut_off_optimal(self):
        m = models.to_break_constraint_tolerances()

        TransformationFactory('gdp.cuttingplane').apply_to(m)

        SolverFactory(solver).solve(m)
        self.assertAlmostEqual(m.x.value, 2)
        self.assertAlmostEqual(m.y.value, 127)

        m.x.fix(2)
        m.y.fix(127)

        cuts = m._pyomo_gdp_cuttingplane_relaxation.cuts
        self.assertEqual(len(cuts), 1)
        cut1_expr = cuts[0].body

        self.assertGreaterEqual(0, value(cut1_expr))

    @unittest.skipIf('gurobi' not in solvers, "Gurobi solver not available")
    def test_2disj_cuts_valid_for_optimal(self):
        m = models.grossmann_twoDisj()
        
        TransformationFactory('gdp.cuttingplane').apply_to(m)

        cuts = m._pyomo_gdp_cuttingplane_relaxation.cuts
        self.assertEqual(len(cuts), 1)

        m.x.fix(2)
        m.y.fix(8)
        m.disjunct1.indicator_var.fix(1)
        m.disjunct3.indicator_var.fix(1)
        m.disjunct2.indicator_var.fix(0)
        m.disjunct4.indicator_var.fix(0)

        cut = cuts[0].body
        self.assertGreaterEqual(0, value(cut))

    @unittest.skipIf('gurobi' not in solvers, "Gurobi solver not available")
    def test_2disj_cuts_valid_elsewhere(self):
        # I'm doing this test to see if it is cutting into the the feasible
        # region somewhere other than the optimal value... That is, if the angle
        # is off enough to cause problems.
        m = models.grossmann_twoDisj()
        
        TransformationFactory('gdp.cuttingplane').apply_to(m)

        cuts = m._pyomo_gdp_cuttingplane_relaxation.cuts
        self.assertEqual(len(cuts), 1)

        m.x.fix(10)
        m.y.fix(3)
        m.disjunct1.indicator_var.fix(0)
        m.disjunct3.indicator_var.fix(0)
        m.disjunct2.indicator_var.fix(1)
        m.disjunct4.indicator_var.fix(1)

        cut = cuts[0].body
        self.assertGreaterEqual(0, value(cut))
