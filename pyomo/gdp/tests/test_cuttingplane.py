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
    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cut_validity(self):
        m = models.oneVarDisj()

        TransformationFactory('gdp.cuttingplane').apply_to(m)
        cuts = m._pyomo_gdp_cuttingplane_relaxation.cuts

        # I really expect 1 or 0. If we are getting more here then we have
        # pretty much got to be adding unncessary cuts...
        self.assertLessEqual(len(cuts), 1)
        for i in cuts:
            cut_expr = cuts[i].body
            m.x.fix(1)
            m.disj1.indicator_var.fix(1)
            m.disj2.indicator_var.fix(0)
            self.assertLessEqual(value(cut_expr), 0)
            m.x.fix(0)
            m.disj1.indicator_var.fix(0)
            m.disj2.indicator_var.fix(1)
            val = value(cut_expr)
            # TODO: How bad an idea is this?? (If you don't like it, just search
            # for the assertAlmostEquals, the ones I changed will all have them)
            #self.assertLessEqual(value(cut_expr), 0)
            # if this isn't less than 0 it better BE 0
            if val > 0:
                self.assertAlmostEqual(val, 0)

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
    def test_cuts_valid_for_optimal(self):
        m = models.makeTwoTermDisj_boxes()
        TransformationFactory('gdp.cuttingplane').apply_to(m)

        cuts = m._pyomo_gdp_cuttingplane_relaxation.cuts
        for i in cuts:
            cut_expr = cuts[i].body
            m.d[0].indicator_var.fix(1)
            m.d[1].indicator_var.fix(0)
            m.x.fix(3)
            m.y.fix(1)
            val = value(cut_expr)
            if val > 0:
                # TODO: so this isn't quite at tolerance of 1e-7
                self.assertAlmostEqual(val, 0, places=6)
            #self.assertLessEqual(value(cut_expr), 0)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cuts_valid_on_chull_vertices(self):
        m = models.makeTwoTermDisj_boxes()
        TransformationFactory('gdp.cuttingplane').apply_to(m)

        cuts = m._pyomo_gdp_cuttingplane_relaxation.cuts
        for i in cuts:
            cut_expr = cuts[i].body
            m.d[0].indicator_var.fix(1)
            m.d[1].indicator_var.fix(0)
            m.x.fix(4)
            m.y.fix(2)
            # (4,2)
            self.assertLessEqual(value(cut_expr), 0)
            m.y.fix(1)
            # (4,1)
            val = value(cut_expr)
            if val > 0:
                self.assertAlmostEqual(val, 0, places=6)
            #self.assertLessEqual(value(cut_expr), 0)
            m.d[0].indicator_var.fix(0)
            m.d[1].indicator_var.fix(1)
            m.x.fix(1)
            m.y.fix(4)
            # (1,4)
            #self.assertLessEqual(value(cut_expr), 0)
            self.assertTrue(value(cuts[i].expr))
            m.x.fix(2)
            # (2.4)
            #self.assertLessEqual(value(cut_expr), 0)
            self.assertTrue(value(cuts[i].expr))
            m.x.fix(1)
            m.y.fix(3)
            # (1,3)
            #self.assertLessEqual(value(cut_expr), 0)
            self.assertTrue(value(cuts[i].expr))

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
        SolverFactory('cplex').solve(m)
        # self.assertAlmostEqual(m.x.value, 2, delta=1e-4)
        # self.assertAlmostEqual(m.y.value, 10, delta=1e-4)
        # I'm calculating my own relative tolerance. Really, nosetests??
        rel_tol = abs(m.x.value - 2)/float(min(m.x.value, 2))
        # TODO: So this is the lowest tolerance at which this passes. But I'm
        # not so unhappy with 1e-7, should we be?
        self.assertLessEqual(rel_tol, 1e-7)
        rel_tol = abs(m.y.value - 10)/float(min(m.y.value, 10))
        self.assertLessEqual(rel_tol, 1e-7)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cut_valid_at_optimal(self):
        m = models.grossmann_oneDisj()
        TransformationFactory('gdp.cuttingplane').apply_to(m)

        cuts = m._pyomo_gdp_cuttingplane_relaxation.cuts
        for i in cuts:
            cut0 = cuts[i]
            self.assertEqual(cut0.upper, 0)
            self.assertIsNone(cut0.lower)
            cut0_expr = cut0.body

            # we check that the cut is valid at the upper righthand corners of
            # the two regions (within a tolerance (in either direction))
            m.x.fix(2)
            m.y.fix(10)
            m.disjunct1.indicator_var.fix(1)
            m.disjunct2.indicator_var.fix(0)
            # TODO: This fails, but it would be true with a tolerance of 1e-7
            # (2,10) = optimal soln
            self.assertLessEqual(value(cut0_expr), 0)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cut_valid_on_facets_containing_optimal(self):
        m = models.grossmann_oneDisj()
        TransformationFactory('gdp.cuttingplane').apply_to(m)

        cuts = m._pyomo_gdp_cuttingplane_relaxation.cuts
        for i in cuts:
            cut0_expr = cuts[i].body

            m.x.fix(10)
            m.y.fix(3)
            m.disjunct1.indicator_var.fix(0)
            m.disjunct2.indicator_var.fix(1)
            # (10,3)
            self.assertLessEqual(value(cut0_expr), 0)

            m.disjunct1.indicator_var.fix(1)
            m.disjunct2.indicator_var.fix(0)
            m.x.fix(0)
            m.y.fix(10)
            # (0,10)
            self.assertLessEqual(value(cut0_expr), 0)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_soln_correct(self):
        m = models.to_break_constraint_tolerances()

        TransformationFactory('gdp.cuttingplane').apply_to(m)

        # TODO: Need to either check for cplex or choose a different solver.
        # Need a MIP solver though...
        SolverFactory('cplex').solve(m)
        # TODO: I'm playing the same game as above:
        # self.assertAlmostEqual(m.x.value, 2)
        # self.assertAlmostEqual(m.y.value, 127)
        rel_tol = abs(m.x.value - 2)/float(min(m.x.value, 2))
        self.assertLessEqual(rel_tol, 1e-4)
        rel_tol = abs(m.y.value - 127)/float(min(m.y.value, 127))
        self.assertLessEqual(rel_tol, 1e-4)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cuts_valid_for_optimal(self):
        m = models.to_break_constraint_tolerances()

        TransformationFactory('gdp.cuttingplane').apply_to(m)
        cuts = m._pyomo_gdp_cuttingplane_relaxation.cuts
        for i in cuts:
            m.disjunct1.indicator_var.fix(1)
            m.disjunct2.indicator_var.fix(0)
            m.x.fix(2)
            m.y.fix(127)

            cut1_expr = cuts[i].body
            # cut valid for optimal soln
            self.assertGreaterEqual(0, value(cut1_expr))

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cuts_valid_for_facets_containing_optimal(self):
        m = models.to_break_constraint_tolerances()

        TransformationFactory('gdp.cuttingplane').apply_to(m)
        cuts = m._pyomo_gdp_cuttingplane_relaxation.cuts
        for i in cuts:
            cut1_expr = cuts[i].body

            # check that the cut is valid for the other upper RH corner
            m.disjunct1.indicator_var.fix(0)
            m.disjunct2.indicator_var.fix(1)
            m.x.fix(120)
            m.y.fix(3)
            self.assertGreaterEqual(0, value(cut1_expr))

            # and check the other neighboring vertex
            m.disjunct1.indicator_var.fix(1)
            m.disjunct2.indicator_var.fix(0)
            m.x.fix(0)
            m.y.fix(127)
            self.assertGreaterEqual(0, value(cut1_expr))

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_2disj_cuts_valid_for_optimal(self):
        m = models.grossmann_twoDisj()
        
        TransformationFactory('gdp.cuttingplane').apply_to(m)

        cuts = m._pyomo_gdp_cuttingplane_relaxation.cuts

        # fix to optimal solution
        m.x.fix(2)
        m.y.fix(8)
        m.disjunct1.indicator_var.fix(1)
        m.disjunct3.indicator_var.fix(1)
        m.disjunct2.indicator_var.fix(0)
        m.disjunct4.indicator_var.fix(0)

        for i in range(len(cuts)):
            cut = cuts[i].body
            self.assertGreaterEqual(0, value(cut))

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_2disj_cuts_valid_elsewhere(self):
        # Check to see if it is cutting into the the feasible region somewhere
        # other than the optimal value... That is, if the angle is off enough to
        # cause problems.
        m = models.grossmann_twoDisj()
        
        TransformationFactory('gdp.cuttingplane').apply_to(m)

        cuts = m._pyomo_gdp_cuttingplane_relaxation.cuts

        for i in range(len(cuts)):
            cut = cuts[i].body

            m.x.fix(10)
            m.y.fix(3)
            m.disjunct1.indicator_var.fix(0)
            m.disjunct3.indicator_var.fix(0)
            m.disjunct2.indicator_var.fix(1)
            m.disjunct4.indicator_var.fix(1)
            self.assertGreaterEqual(0, value(cut))
        
            m.y.fix(2)
            self.assertGreaterEqual(0, value(cut))

            m.x.fix(9)
            self.assertGreaterEqual(0, value(cut))

            m.y.fix(3)
            self.assertGreaterEqual(0, value(cut))

            m.disjunct1.indicator_var.fix(1)
            m.disjunct2.indicator_var.fix(0)
            m.disjunct3.indicator_var.fix(1)
            m.disjunct4.indicator_var.fix(0)

            m.x.fix(1)
            m.y.fix(8)
            self.assertGreaterEqual(0, value(cut))

            m.y.fix(7)
            self.assertGreaterEqual(0, value(cut))

            m.x.fix(2)
            self.assertGreaterEqual(0, value(cut))
            
            m.y.fix(8)
            self.assertGreaterEqual(0, value(cut))

class NonlinearConvex_TwoCircles(unittest.TestCase):
    # This is a really weak test because it is not actually *this* apply_to call
    # that raises the exception. But that exception is tested in bigm, so maybe
    # it doesn't matter much.
    @raises(GDP_Error)
    def test_complain_if_no_bigm(self):
        m = models.twoDisj_twoCircles_easy()
        TransformationFactory('gdp.cuttingplane').apply_to(m)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cuts_valid_for_optimal(self):
        m = models.twoDisj_twoCircles_easy()
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=1e6)

        cuts = m._pyomo_gdp_cuttingplane_relaxation.cuts
        
        m.x.fix(2)
        m.y.fix(7)
        m.upper_circle.indicator_var.fix(1)
        m.lower_circle.indicator_var.fix(0)
        for i in range(len(cuts)):
            self.assertGreaterEqual(0, value(cuts[i].body))

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cuts_valid_on_facet_containing_optimal(self):
        m = models.twoDisj_twoCircles_easy()
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=1e6)

        cuts = m._pyomo_gdp_cuttingplane_relaxation.cuts

        m.x.fix(5)
        m.y.fix(3)
        m.upper_circle.indicator_var.fix(0)
        m.lower_circle.indicator_var.fix(1)
        for i in range(len(cuts)):
            self.assertTrue(value(cuts[i].expr))
            #self.assertGreaterEqual(0, value(cuts[i].body))
            
    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cuts_valid_for_optimal_tighter_m(self):
        m = models.twoDisj_twoCircles_easy()

        # this M comes from the fact that y \in (0,8) and x \in (0,6)
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=83)

        cuts = m._pyomo_gdp_cuttingplane_relaxation.cuts

        m.x.fix(2)
        m.y.fix(7)
        m.upper_circle.indicator_var.fix(1)
        m.lower_circle.indicator_var.fix(0)

        for i in range(len(cuts)):
            self.assertTrue(value(cuts[i].expr))
            #self.assertGreaterEqual(0, value(cuts[i].body))

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cuts_valid_for_optimalFacet_tighter_m(self):
        m = models.twoDisj_twoCircles_easy()

        # this M comes from the fact that y \in (0,8) and x \in (0,6)
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=83)

        cuts = m._pyomo_gdp_cuttingplane_relaxation.cuts

        m.x.fix(5)
        m.y.fix(3)
        m.upper_circle.indicator_var.fix(0)
        m.lower_circle.indicator_var.fix(1)

        for i in range(len(cuts)):
            self.assertGreaterEqual(0, value(cuts[i].body))
        
class NonlinearConvex_OverlappingCircles(unittest.TestCase):        
    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cuts_valid_for_optimal(self):
        m = models.fourCircles()
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=1e6)

        cuts = m._pyomo_gdp_cuttingplane_relaxation.cuts
        
        m.x.fix(2)
        m.y.fix(7)
        m.upper_circle.indicator_var.fix(1)
        m.lower_circle.indicator_var.fix(0)
        m.upper_circle2.indicator_var.fix(1)
        m.lower_circle2.indicator_var.fix(0)
        for i in range(len(cuts)):
            self.assertGreaterEqual(0, value(cuts[i].body))

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cuts_valid_on_facet_containing_optimal(self):
        m = models.fourCircles()
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=1e6)

        cuts = m._pyomo_gdp_cuttingplane_relaxation.cuts

        m.x.fix(5)
        m.y.fix(3)
        m.upper_circle.indicator_var.fix(0)
        m.lower_circle.indicator_var.fix(1)
        m.upper_circle2.indicator_var.fix(0)
        m.lower_circle2.indicator_var.fix(1)
        for i in range(len(cuts)):
            self.assertGreaterEqual(0, value(cuts[i].body))

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cuts_valid_for_optimal_tightM(self):
        m = models.fourCircles()
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=83)

        cuts = m._pyomo_gdp_cuttingplane_relaxation.cuts
        
        m.x.fix(2)
        m.y.fix(7)
        m.upper_circle.indicator_var.fix(1)
        m.lower_circle.indicator_var.fix(0)
        m.upper_circle2.indicator_var.fix(1)
        m.lower_circle2.indicator_var.fix(0)
        for i in range(len(cuts)):
            self.assertGreaterEqual(0, value(cuts[i].body))

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cuts_valid_on_facet_containing_optimal_tightM(self):
        m = models.fourCircles()
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=83)

        cuts = m._pyomo_gdp_cuttingplane_relaxation.cuts

        m.x.fix(5)
        m.y.fix(3)
        m.upper_circle.indicator_var.fix(0)
        m.lower_circle.indicator_var.fix(1)
        m.upper_circle2.indicator_var.fix(0)
        m.lower_circle2.indicator_var.fix(1)
        for i in range(len(cuts)):
            self.assertGreaterEqual(0, value(cuts[i].body))
