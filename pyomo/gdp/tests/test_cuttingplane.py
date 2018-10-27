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

import pyomo.opt
import pyomo.gdp.tests.models as models
from pyomo.repn import generate_standard_repn

import random
from six import StringIO

from nose.tools import set_trace, raises

solvers = pyomo.opt.check_available_solvers('ipopt')

# TODO:
#     - test that deactivated objectives on the model don't get used by the
#       transformation

def check_linear_coef(self, repn, var, coef):
    var_id = None
    for i,v in enumerate(repn.linear_vars):
        if v is var:
            var_id = i
    self.assertIsNotNone(var_id)
    self.assertAlmostEqual(repn.linear_coefs[var_id], coef)

class OneVarDisj(unittest.TestCase):
    # there are no useful cuts here, and so we don't add them!
    def test_no_cuts_added(self):
        m = models.oneVarDisj()

        TransformationFactory('gdp.cuttingplane').apply_to(m)
        cuts = m._pyomo_gdp_cuttingplane_relaxation.cuts
        self.assertEqual(len(cuts), 0)

class TwoTermDisj(unittest.TestCase):
    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    # check that we have a transformation block and that the cuts are on it.
    def test_transformation_block(self):
        m = models.makeTwoTermDisj_boxes()
        TransformationFactory('gdp.cuttingplane').apply_to(m)

        # we created the block
        transBlock = m._pyomo_gdp_cuttingplane_relaxation
        self.assertIsInstance(transBlock, Block)
        # the cuts are on it
        cuts = transBlock.cuts
        self.assertIsInstance(cuts, Constraint)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cut_constraint(self):
        m = models.makeTwoTermDisj_boxes()
        TransformationFactory('gdp.cuttingplane').apply_to(m)

        # we don't get any cuts from this
        self.assertEqual(len(m._pyomo_gdp_cuttingplane_relaxation.cuts), 0)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_create_using(self):
        m = models.makeTwoTermDisj_boxes()

        # TODO: this is duplicate code with other transformation tests. Someday
        # it would be nice to centralize it...
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

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
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
    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_correct_soln(self):
        m = models.grossmann_oneDisj()
        TransformationFactory('gdp.cuttingplane').apply_to(m)

        # TODO: probably don't want to be solving here in the long term?
        # checking if we get the optimal solution.
        SolverFactory('gurobi').solve(m)
        self.assertAlmostEqual(m.x.value, 2)
        self.assertAlmostEqual(m.y.value, 10)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cut_valid(self):
        m = models.grossmann_oneDisj()
        TransformationFactory('gdp.cuttingplane').apply_to(m)

        cuts = m._pyomo_gdp_cuttingplane_relaxation.cuts
        # we expect one cut
        self.assertEqual(len(cuts), 1)

        cut0 = cuts[0]
        self.assertEqual(cut0.upper, 0)
        self.assertIsNone(cut0.lower)
        cut0_expr = cut0.body
        # we check that the cut is tight at the upper righthand corners of the
        # two regions (within a tolerance (in either direction))
        m.x.fix(2)
        m.y.fix(10)
        m.disjunct1.indicator_var.fix(1)
        m.disjunct2.indicator_var.fix(0)
        self.assertAlmostEqual(value(cut0_expr), 0)

        m.x.fix(10)
        m.y.fix(3)
        m.disjunct1.indicator_var.fix(0)
        m.disjunct2.indicator_var.fix(1)
        self.assertAlmostEqual(value(cut0_expr), 0)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cuts_dont_cut_off_optimal(self):
        m = models.to_break_constraint_tolerances()

        TransformationFactory('gdp.cuttingplane').apply_to(m)

        SolverFactory('gurobi').solve(m)
        self.assertAlmostEqual(m.x.value, 2)
        self.assertAlmostEqual(m.y.value, 127)

        m.x.fix(2)
        m.y.fix(127)

        cuts = m._pyomo_gdp_cuttingplane_relaxation.cuts
        self.assertEqual(len(cuts), 1)
        cut1_expr = cuts[0].body
        # cut tight, but within tolerance
        self.assertAlmostEqual(0, value(cut1_expr))

        # check that the cut is valid for the other upper RH corner
        m.x.fix(120)
        m.y.fix(3)
        self.assertGreaterEqual(0, value(cut1_expr))

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_2disj_cuts_valid_for_optimal(self):
        m = models.grossmann_twoDisj()
        
        TransformationFactory('gdp.cuttingplane').apply_to(m)

        cuts = m._pyomo_gdp_cuttingplane_relaxation.cuts
        self.assertEqual(len(cuts), 1)

        # fix to optimal solution
        m.x.fix(2)
        m.y.fix(8)
        m.disjunct1.indicator_var.fix(1)
        m.disjunct3.indicator_var.fix(1)
        m.disjunct2.indicator_var.fix(0)
        m.disjunct4.indicator_var.fix(0)

        cut = cuts[0].body
        self.assertGreaterEqual(0, value(cut))

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_2disj_cuts_valid_elsewhere(self):
        # Check to see if it is cutting into the the feasible region somewhere
        # other than the optimal value... That is, if the angle is off enough to
        # cause problems.
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
        
        m.y.fix(2)
        self.assertGreaterEqual(0, value(cut))

        m.x.fix(9)
        self.assertGreaterEqual(0, value(cut))

        m.x.fix(1)
        m.y.fix(8)
        self.assertGreaterEqual(0, value(cut))

        m.y.fix(7)
        self.assertGreaterEqual(0, value(cut))

class NonlinearConvex(unittest.TestCase):
    # This is a really weak test because it is not actually *this* apply_to call
    # that raises the exception. But that exception is tested in bigm, so maybe
    # it doesn't matter much.
    @raises(GDP_Error)
    def test_complain_if_no_bigm(self):
        m = models.twoDisj_nonlin_convex()
        TransformationFactory('gdp.cuttingplane').apply_to(m)

    def test_cuts_with_tighter_m(self):
        m = models.twoDisj_nonlin_convex()

        # DEBUG
        # sanity check
        #sanity = TransformationFactory('gdp.bigm').create_using(m, bigM=68)
        #results = SolverFactory('baron').solve(sanity, tee=True)
        #print results

        # I think this is the issue with this problem. The optimal solution is
        # nasty... I might do better to a construct a problem with an integer
        # optimal solution. Because then I could have some faith in this test...

        # x = 5.89442707753, y = 5.44721382244
        #set_trace()

        # This bigM value comes from the fact that both disjunts are contained
        # in the box defined by 0 <= x <= 6, 0 <= y <= 6.
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=68)
        cuts = m._pyomo_gdp_cuttingplane_relaxation.cuts

        self.assertEqual(len(cuts), 1)

        m.x.fix(5.928)
        m.y.fix(5.372)
        m.circle1.indicator_var.fix(0)
        m.circle2.indicator_var.fix(1)

        # cuts off optimal solution by about 0.5
        self.assertGreaterEqual(0, value(cuts[0].body))

    def test_cuts_with_weak_m(self):
        m = models.twoDisj_nonlin_convex()

        # the optimal solution is approximately x = 5.928, y = 5.372

        # TODO: An additional bug is that this whole thing crashes when I use
        # couenne...?
        TransformationFactory('gdp.cuttingplane').apply_to(m, solver='baron',
                                                           bigM=537)
        cuts = m._pyomo_gdp_cuttingplane_relaxation.cuts
        self.assertEqual(len(cuts), 5)

        m.x.fix(5.928)
        m.y.fix(5.372)
        m.circle1.indicator_var.fix(0)
        m.circle2.indicator_var.fix(1)

        self.assertGreaterEqual(0, value(cuts[0].body))
        
        
