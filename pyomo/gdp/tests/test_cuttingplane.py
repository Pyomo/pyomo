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
from pyomo.gdp.tests.common_tests import diff_apply_to_and_create_using

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

def check_validity(self, body, lower, upper):
    if lower is not None:
        self.assertGreaterEqual(value(body), value(lower))
    if upper is not None:
        self.assertLessEqual(value(body), value(upper))

class OneVarDisj(unittest.TestCase):
    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_no_cuts_for_optimal_m(self):
        m = models.oneVarDisj_2pts()

        TransformationFactory('gdp.cuttingplane').apply_to(m)
        cuts = m._pyomo_gdp_cuttingplane_transformation.cuts

        # I really expect 1 or 0. If we are getting more here then we have
        # pretty much got to be adding unncessary cuts...

        # Big-m is actually tight for the optimal M value, so I should have no
        # cuts. If I have any then we are wasting our time.
        self.assertEqual(len(cuts), 0)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_expected_two_segment_cut(self):
        m = models.twoSegments_SawayaGrossmann()
        # have to make M big for the bigm relaxation to be the box 0 <= x <= 3,
        # 0 <= Y <= 1 (in the limit)
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=1e6)

        # I actually know exactly the cut I am expecting in this case (I think I
        # get it twice, which is a bummer, but I am just going to make sure I
        # get it by testing that it is tight at two points)
        cuts = m._pyomo_gdp_cuttingplane_transformation.cuts

        # I should get one cut because I made bigM really bad, but I only need
        # one facet of the convex hull for this problem to be done.
        cuts.pprint()
        self.assertEqual(len(cuts), 1)

        cut_expr = cuts[0].body
        # should be 2Y_2 <= x
        m.x.fix(0)
        m.disj2.indicator_var.fix(0)
        # The almost equal here is OK because we are going to check that it is
        # actually valid in the next test. I just wanted to make sure it is the
        # line I am expecting, so I want to know that it is tight here.
        self.assertAlmostEqual(value(cut_expr), 0)

        m.x.fix(2)
        m.disj2.indicator_var.fix(1)
        self.assertAlmostEqual(value(cut_expr), 0)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_two_segment_cuts_valid(self):
        m = models.twoSegments_SawayaGrossmann()
        # have to make M big for the bigm relaxation to be the box 0 <= x <= 3,
        # 0 <= Y <= 1 (in the limit)
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=1e6)

        cuts = m._pyomo_gdp_cuttingplane_transformation.cuts

        # check that all the cuts are valid everywhere
        for cut in cuts.values():
            cut_expr = cut.body
            cut_lower = cut.lower
            cut_upper = cut.upper
            # there are just 4 extreme points of convex hull, we'll test all of
            # them

            # (0,0)
            m.x.fix(0)
            m.disj2.indicator_var.fix(0)
            check_validity(self, cut_expr, cut_lower, cut_upper)

            # (1,0)
            m.x.fix(1)
            check_validity(self, cut_expr, cut_lower, cut_upper)

            # (2, 1)
            m.x.fix(2)
            m.disj2.indicator_var.fix(1)
            check_validity(self, cut_expr, cut_lower, cut_upper)
            
            # (3, 1)
            m.x.fix(3)
            check_validity(self, cut_expr, cut_lower, cut_upper)
        

class TwoTermDisj(unittest.TestCase):
    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    # check that we have a transformation block and that the cuts are on it.
    def test_transformation_block(self):
        m = models.makeTwoTermDisj_boxes()
        TransformationFactory('gdp.cuttingplane').apply_to(m)

        # we created the block
        transBlock = m._pyomo_gdp_cuttingplane_transformation
        self.assertIsInstance(transBlock, Block)
        # the cuts are on it
        cuts = transBlock.cuts
        self.assertIsInstance(cuts, Constraint)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cuts_valid_for_optimal(self):
        m = models.makeTwoTermDisj_boxes()
        TransformationFactory('gdp.cuttingplane').apply_to(m)

        cuts = m._pyomo_gdp_cuttingplane_transformation.cuts
        for cut in cuts.values():
            cut_expr = cut.body
            lower = cut.lower
            upper = cut.upper
            m.d[0].indicator_var.fix(1)
            m.d[1].indicator_var.fix(0)
            m.x.fix(3)
            m.y.fix(1)
            check_validity(self, cut_expr, lower, upper)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cuts_valid_on_chull_vertices(self):
        m = models.makeTwoTermDisj_boxes()
        TransformationFactory('gdp.cuttingplane').apply_to(m)

        extreme_pts = [
            (1,0,4,1),
            (1,0,4,2),
            (1,0,3,1),
            (1,0,3,2),
            (0,1,1,3),
            (0,1,1,4),
            (0,1,2,3),
            (0,1,2,4)
        ]

        cuts = m._pyomo_gdp_cuttingplane_transformation.cuts
        for cut in cuts.values():
            cut_expr = cut.body
            lower = cut.lower
            upper = cut.upper
            # now there are 8 extreme points and we can test all of them
            for pt in extreme_pts:
                m.d[0].indicator_var.fix(pt[0])
                m.d[1].indicator_var.fix(pt[1])
                m.x.fix(pt[2])
                m.y.fix(pt[3])
                check_validity(self, cut_expr, lower, upper)

    # [ESJ 15 Mar 19] This is kind of a miracle that this passes... But it is
    # true and deserves brownie points. But I'm not sure that makes it a good
    # test? My other idea is below... Both are a little scary because I am
    # allowing NO numerical error. But I also don't have any!
    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cut_is_correct_facet(self):
        m = models.makeTwoTermDisj_boxes()
        TransformationFactory('gdp.cuttingplane').apply_to(m)

        facet_extreme_pts = [
            (1,0,3,1),
            (1,0,3,2),
            (0,1,1,3),
            (0,1,1,4)
        ]
        
        cuts = m._pyomo_gdp_cuttingplane_transformation.cuts
        cut = cuts[0]
        #set_trace()
        cut_expr = cut.body
        lower = cut.lower
        upper = cut.upper
        tight = 0
        for pt in facet_extreme_pts:
            m.d[0].indicator_var.fix(pt[0])
            m.d[1].indicator_var.fix(pt[1])
            m.x.fix(pt[2])
            m.y.fix(pt[3])
            if lower is not None:
                self.assertEqual(value(lower), value(cut_expr))
            if upper is not None:
                self.assertEqual(value(upper), value(cut_expr))


    # [ESJ 15 Mar 19]: This is a slightly weird idea, but I think we've kind of
    # messed up if each of our cuts isn't tight for at least one extreme point of
    # the convex hull (in the original space). I don't think this is a very
    # robust test yet because I am asking for exact equality. But on the other
    # hand, exact means numerically life is very good.
    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cuts_tight_somewhere(self):
        m = models.makeTwoTermDisj_boxes()
        TransformationFactory('gdp.cuttingplane').apply_to(m)

        # TODO: this is redundant code, I should probably centralize this
        extreme_pts = [
            (1,0,4,1),
            (1,0,4,2),
            (1,0,3,1),
            (1,0,3,2),
            (0,1,1,3),
            (0,1,1,4),
            (0,1,2,3),
            (0,1,2,4)
        ]

        cuts = m._pyomo_gdp_cuttingplane_transformation.cuts
        for cut in cuts.values():
            cut_expr = cut.body
            lower = value(cut.lower)
            upper = value(cut.upper)
            tight = 0
            for pt in extreme_pts:
                m.d[0].indicator_var.fix(pt[0])
                m.d[1].indicator_var.fix(pt[1])
                m.x.fix(pt[2])
                m.y.fix(pt[3])
                if lower is not None:
                    if value(cut_expr) == lower:
                        tight += 1
                if upper is not None:
                    if value(cut_expr) == upper:
                        tight += 1
            self.assertGreaterEqual(tight, 1)
   
    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_create_using(self):
        m = models.makeTwoTermDisj_boxes()
        # TODO: I think doesn't pass because of inconsistent ordering of terms
        # within expressions (based on an old note to myself, but I need to
        # check)
        diff_apply_to_and_create_using(self, m, 'gdp.cuttingplane')

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
    def test_cuts_valid_at_extreme_pts(self):
        m = models.grossmann_oneDisj()
        TransformationFactory('gdp.cuttingplane').apply_to(m)

        extreme_points = [
            (1,0,2,10),
            (1,0,0,10),
            (1,0,0,7),
            (1,0,2,7),
            (0,1,8,0),
            (0,1,8,3),
            (0,1,10,0),
            (0,1,10,3)
        ]

        cuts = m._pyomo_gdp_cuttingplane_transformation.cuts
        for cut in cuts.values():
            cut_expr = cut.body
            lower = cut.lower
            upper = cut.upper

            # we check that the cut is valid for all of the above points
            for pt in extreme_points:
                m.x.fix(pt[2])
                m.y.fix(pt[3])
                m.disjunct1.indicator_var.fix(pt[0])
                m.disjunct2.indicator_var.fix(pt[1])
                check_validity(self, cut_expr, lower, upper)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cuts_valid_at_extreme_pts_rescaled(self):
        m = models.to_break_constraint_tolerances()
        TransformationFactory('gdp.cuttingplane').apply_to(m)
        extreme_points = [
            (1,0,2,127),
            (1,0,0,127),
            (1,0,0,117),
            (1,0,2,117),
            (0,1,118,0),
            (0,1,118,3),
            (0,1,120,0),
            (0,1,120,3)
        ]

        cuts = m._pyomo_gdp_cuttingplane_transformation.cuts
        for cut in cuts.values():
            cut_expr = cut.body
            lower = cut.lower
            upper = cut.upper

            # we check that the cut is valid for all of the above points
            for pt in extreme_points:
                m.x.fix(pt[2])
                m.y.fix(pt[3])
                m.disjunct1.indicator_var.fix(pt[0])
                m.disjunct2.indicator_var.fix(pt[1])
                check_validity(self, cut_expr, lower, upper)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_2disj_cuts_valid_for_extreme_pts(self):
        m = models.grossmann_twoDisj()
        TransformationFactory('gdp.cuttingplane').apply_to(m)

        extreme_points = [
            (1,0,1,0,1,7),
            (1,0,1,0,1,8),
            (1,0,1,0,2,7),
            (1,0,1,0,2,8),
            (0,1,0,1,9,2),
            (0,1,0,1,9,3),
            (0,1,0,1,11,2),
            (0,1,0,1,11,3)
        ]

        cuts = m._pyomo_gdp_cuttingplane_transformation.cuts
        for cut in cuts.values():
            cut_expr = cut.body
            lower = cut.lower
            upper = cut.upper

            for pt in extreme_points:
                m.x.fix(pt[4])
                m.y.fix(pt[5])
                m.disjunct1.indicator_var.fix(pt[0])
                m.disjunct2.indicator_var.fix(pt[1])
                m.disjunct3.indicator_var.fix(pt[2])
                m.disjunct4.indicator_var.fix(pt[3])
                check_validity(self, cut_expr, lower, upper)

class NonlinearConvex_TwoCircles(unittest.TestCase):
    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cuts_valid_for_optimal(self):
        m = models.twoDisj_twoCircles_easy()
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=1e6)

        cuts = m._pyomo_gdp_cuttingplane_transformation.cuts
        
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

        cuts = m._pyomo_gdp_cuttingplane_transformation.cuts

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

        cuts = m._pyomo_gdp_cuttingplane_transformation.cuts

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

        cuts = m._pyomo_gdp_cuttingplane_transformation.cuts

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

        cuts = m._pyomo_gdp_cuttingplane_transformation.cuts
        
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

        cuts = m._pyomo_gdp_cuttingplane_transformation.cuts

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

        cuts = m._pyomo_gdp_cuttingplane_transformation.cuts
        
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

        cuts = m._pyomo_gdp_cuttingplane_transformation.cuts

        m.x.fix(5)
        m.y.fix(3)
        m.upper_circle.indicator_var.fix(0)
        m.lower_circle.indicator_var.fix(1)
        m.upper_circle2.indicator_var.fix(0)
        m.lower_circle2.indicator_var.fix(1)
        for i in range(len(cuts)):
            self.assertGreaterEqual(0, value(cuts[i].body))
