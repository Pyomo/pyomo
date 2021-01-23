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

from pyomo.environ import (ConcreteModel, Var, Constraint, Objective, Block,
                           TransformationFactory, value, maximize, Suffix)
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.plugins.cuttingplane import create_cuts_fme 

import pyomo.opt
import pyomo.gdp.tests.models as models
from pyomo.repn import generate_standard_repn
from pyomo.gdp.tests.common_tests import diff_apply_to_and_create_using

from six import StringIO

solvers = pyomo.opt.check_available_solvers('ipopt', 'gurobi')

def check_validity(self, body, lower, upper, TOL=0):
    if lower is not None:
        self.assertGreaterEqual(value(body), value(lower) - TOL)
    if upper is not None:
        self.assertLessEqual(value(body), value(upper) + TOL)

class OneVarDisj(unittest.TestCase):
    def check_no_cuts_for_optimal_m(self, m):
        cuts = m._pyomo_gdp_cuttingplane_transformation.cuts

        # Big-m is actually tight for the optimal M value, so I should have no
        # cuts. If I have any then we are wasting our time.
        self.assertEqual(len(cuts), 0)

    def check_expected_two_segment_cut(self, m):
        # I am expecting exactly one cut
        cuts = m._pyomo_gdp_cuttingplane_transformation.cuts

        # I should get one cut because I made bigM really bad, but I only need
        # one facet of the convex hull for this problem to be done.
        self.assertEqual(len(cuts), 1)

        cut_expr = cuts[0].body
        # should be 2Y_2 <= x
        m.x.fix(0)
        m.disj1.indicator_var.fix(1)
        m.disj2.indicator_var.fix(0)
        # The almost equal here is OK because we are going to check that it is
        # actually valid in the next test. I just wanted to make sure it is the
        # line I am expecting, so I want to know that it is tight here...
        self.assertAlmostEqual(value(cut_expr), 0)

        # ...and that it is tight here
        m.x.fix(2)
        m.disj2.indicator_var.fix(1)
        m.disj1.indicator_var.fix(0)
        self.assertAlmostEqual(value(cut_expr), 0)

    def check_two_segment_cuts_valid(self, m):
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
            check_validity(self, cut_expr, cut_lower, cut_upper, TOL=1e-8)

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

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_no_cuts_for_optimal_m(self):
        m = models.oneVarDisj_2pts()

        TransformationFactory('gdp.cuttingplane').apply_to(m)
        self.check_no_cuts_for_optimal_m(m)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_no_cuts_for_optimal_m_fme(self):
        m = models.oneVarDisj_2pts()

        TransformationFactory('gdp.cuttingplane').apply_to(
            m, 
            create_cuts=create_cuts_fme,
            post_process_cut=None
        )
        self.check_no_cuts_for_optimal_m(m)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_no_cuts_for_optimal_m_inf_norm(self):
        m = models.oneVarDisj_2pts()

        TransformationFactory('gdp.cuttingplane').apply_to(
            m, 
            norm=float('inf'),
            post_process_cut=None
        )
        self.check_no_cuts_for_optimal_m(m)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_expected_two_segment_cut(self):
        m = models.twoSegments_SawayaGrossmann()
        # have to make M big for the bigm relaxation to be the box 0 <= x <= 3,
        # 0 <= Y <= 1 (in the limit)
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=1e6,
                                                           verbose=True)
        self.check_expected_two_segment_cut(m)
    
    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_expected_two_segment_cut_fme(self):
        m = models.twoSegments_SawayaGrossmann()
        # have to make M big for the bigm relaxation to be the box 0 <= x <= 3,
        # 0 <= Y <= 1 (in the limit)
        TransformationFactory('gdp.cuttingplane').apply_to(
            m, bigM=1e6, create_cuts=create_cuts_fme,
            post_process_cut=None)
        self.check_expected_two_segment_cut(m)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_expected_two_segment_cut_inf_norm(self):
        m = models.twoSegments_SawayaGrossmann()
        
        # make sure this is fine if dual Suffix is already on model
        m.dual = Suffix(direction=Suffix.IMPORT)

        # have to make M big for the bigm relaxation to be the box 0 <= x <= 3,
        # 0 <= Y <= 1 (in the limit)
        TransformationFactory('gdp.cuttingplane').apply_to(
            m, bigM=1e6, norm=float('inf'),
            post_process_cut=None)
        self.check_expected_two_segment_cut(m)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_expected_two_segment_cut_inf_norm_fme(self):
        m = models.twoSegments_SawayaGrossmann()
        # have to make M big for the bigm relaxation to be the box 0 <= x <= 3,
        # 0 <= Y <= 1 (in the limit)
        TransformationFactory('gdp.cuttingplane').apply_to(
            m, bigM=1e6, norm=float('inf'), create_cuts=create_cuts_fme,
            post_process_cut=None, verbose=True)
        self.check_expected_two_segment_cut(m)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_deactivated_objectives_ignored(self):
        m = models.twoSegments_SawayaGrossmann()
        # add an opposite direction objective, but deactivate it
        m.another_obj = Objective(expr=m.x - m.disj2.indicator_var,
                                  sense=maximize)
        m.another_obj.deactivate()

        # have to make M big for the bigm relaxation to be the box 0 <= x <= 3,
        # 0 <= Y <= 1 (in the limit)
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=1e6,
                                                           verbose=True)
        self.check_expected_two_segment_cut(m)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_two_segment_cuts_valid(self):
        m = models.twoSegments_SawayaGrossmann()
        # add stupid variable to make sure that we don't require everything in
        # rBigM to be useful.
        m.will_be_stale = Var()

        # have to make M big for the bigm relaxation to be the box 0 <= x <= 3,
        # 0 <= Y <= 1 (in the limit)
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=1e6)

        self.check_two_segment_cuts_valid(m)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_two_segment_cuts_valid_fme(self):
        m = models.twoSegments_SawayaGrossmann()
        # have to make M big for the bigm relaxation to be the box 0 <= x <= 3,
        # 0 <= Y <= 1 (in the limit)
        TransformationFactory('gdp.cuttingplane').apply_to(
            m, bigM=1e6, create_cuts=create_cuts_fme,
            post_process_cut=None)

        self.check_two_segment_cuts_valid(m)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_two_segment_cuts_valid_inf_norm(self):
        m = models.twoSegments_SawayaGrossmann()

        # make sure this is fine if there is a random component called dual:
        # Note that this not only tests that we handle the creating of the dual
        # Suffix correctly, but also that we handle a stupid variable in rBigM
        # that will come back stale and really shouldn't be in the separation
        # problem.
        m.dual = Var()

        # have to make M big for the bigm relaxation to be the box 0 <= x <= 3,
        # 0 <= Y <= 1 (in the limit)
        # This one has to post process, but it is correct with the default
        # settings.
        TransformationFactory('gdp.cuttingplane').apply_to(
            m, bigM=1e6, norm=float('inf'))

        self.check_two_segment_cuts_valid(m)

    def check_expected_two_segment_cut_exact(self, cuts):
        m = cuts.model()
        # I should get one cut because I made bigM really bad, but I only need
        # one facet of the convex hull for this problem to be done.
        self.assertEqual(len(cuts), 1)

        # check that the cut is exact.
        cut_expr = cuts[0].body
        # should be 2Y_2 <= x
        m.x.fix(0)
        m.disj1.indicator_var.fix(1)
        m.disj2.indicator_var.fix(0)
        # The almost equal here is OK because we are going to check that it is
        # actually valid in the next test. I just wanted to make sure it is the
        # line I am expecting, so I want to know that it is tight here...
        self.assertEqual(value(cut_expr), 0)

        # ...and that it is tight here
        m.x.fix(2)
        m.disj2.indicator_var.fix(1)
        m.disj1.indicator_var.fix(0)
        self.assertEqual(value(cut_expr), 0)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_integer_arithmetic_cuts_valid_l2(self):
        m = models.twoSegments_SawayaGrossmann()
        # have to make M big for the bigm relaxation to be the box 0 <= x <= 3,
        # 0 <= Y <= 1 (in the limit)
        TransformationFactory('gdp.cuttingplane').apply_to(
            m, bigM=1e6, create_cuts=create_cuts_fme,
            post_process_cut=None, do_integer_arithmetic=True)
        cuts = m._pyomo_gdp_cuttingplane_transformation.cuts

        self.check_expected_two_segment_cut_exact(cuts)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_integer_arithmetic_cuts_valid_inf_norm(self):
        m = models.twoSegments_SawayaGrossmann()
        # have to make M big for the bigm relaxation to be the box 0 <= x <= 3,
        # 0 <= Y <= 1 (in the limit)
        TransformationFactory('gdp.cuttingplane').apply_to(
            m, bigM=1e6, create_cuts=create_cuts_fme, norm=float('inf'),
            post_process_cut=None, do_integer_arithmetic=True)
        cuts = m._pyomo_gdp_cuttingplane_transformation.cuts

        self.check_expected_two_segment_cut_exact(cuts)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_maximization(self):
        m = models.twoSegments_SawayaGrossmann()
        m.obj.expr = -m.obj.expr
        m.obj.sense = maximize

        # have to make M big for the bigm relaxation to be the box 0 <= x <= 3,
        # 0 <= Y <= 1 (in the limit)
        TransformationFactory('gdp.cuttingplane').apply_to(
            m, bigM=1e6, create_cuts=create_cuts_fme,
            post_process_cut=None, do_integer_arithmetic=True)
        cuts = m._pyomo_gdp_cuttingplane_transformation.cuts

        self.check_expected_two_segment_cut_exact(cuts)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cuts_named_correctly(self):
        m = models.twoSegments_SawayaGrossmann()
        # have to make M big for the bigm relaxation to be the box 0 <= x <= 3,
        # 0 <= Y <= 1 (in the limit)
        TransformationFactory('gdp.cuttingplane').apply_to(
            m, bigM=1e6, create_cuts=create_cuts_fme, cuts_name="perfect_cuts",
            post_process_cut=None, do_integer_arithmetic=True)
        cuts = m.component("perfect_cuts")
        self.assertIsInstance(cuts, Constraint)
        self.assertIsNone(
            m._pyomo_gdp_cuttingplane_transformation.component("cuts"))

        self.check_expected_two_segment_cut_exact(cuts)

    def test_non_unique_cut_name_error(self):
        m = models.twoSegments_SawayaGrossmann()

        self.assertRaisesRegexp(
            GDP_Error,
            "cuts_name was specified as 'disj1', but this is "
            "already a component on the instance! Please "
            "specify a unique name.",
            TransformationFactory('gdp.cuttingplane').apply_to,
            m,
            cuts_name="disj1")
   
class TwoTermDisj(unittest.TestCase):
    extreme_points = [
        (1,0,4,1),
        (1,0,4,2),
        (1,0,3,1),
        (1,0,3,2),
        (0,1,1,3),
        (0,1,1,4),
        (0,1,2,3),
        (0,1,2,4)
    ]

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

    def check_cuts_valid_for_optimal(self, m, TOL):
        cuts = m._pyomo_gdp_cuttingplane_transformation.cuts
        for cut in cuts.values():
            cut_expr = cut.body
            lower = cut.lower
            upper = cut.upper
            m.d[0].indicator_var.fix(1)
            m.d[1].indicator_var.fix(0)
            m.x.fix(3)
            m.y.fix(1)
            check_validity(self, cut_expr, lower, upper, TOL)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cuts_valid_for_optimal_fme(self):
        m = models.makeTwoTermDisj_boxes()
        TransformationFactory('gdp.cuttingplane').apply_to(
            m, create_cuts=create_cuts_fme, post_process_cut=None)

        self.check_cuts_valid_for_optimal(m, TOL=0)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cuts_valid_for_optimal_with_tolerance(self):
        m = models.makeTwoTermDisj_boxes()
        TransformationFactory('gdp.cuttingplane').apply_to(
            m, back_off_problem_tolerance=1e-7)

        self.check_cuts_valid_for_optimal(m, TOL=1e-8)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cuts_valid_for_optimal_inf_norm(self):
        m = models.makeTwoTermDisj_boxes()
        TransformationFactory('gdp.cuttingplane').apply_to( m,
                                                            norm=float('inf'))
        # same tolerance as the l-2 norm version:
        self.check_cuts_valid_for_optimal(m, TOL=1e-8)

    def check_cuts_valid_on_hull_vertices(self, m, TOL=0):
        cuts = m._pyomo_gdp_cuttingplane_transformation.cuts
        for cut in cuts.values():
            cut_expr = cut.body
            lower = cut.lower
            upper = cut.upper
            # now there are 8 extreme points and we can test all of them
            for pt in self.extreme_points:
                m.d[0].indicator_var.fix(pt[0])
                m.d[1].indicator_var.fix(pt[1])
                m.x.fix(pt[2])
                m.y.fix(pt[3])
                check_validity(self, cut_expr, lower, upper, TOL)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cuts_valid_on_hull_vertices_fme(self):
        m = models.makeTwoTermDisj_boxes()
        TransformationFactory('gdp.cuttingplane').apply_to(
            m, create_cuts=create_cuts_fme, post_process_cut=None)

        self.check_cuts_valid_on_hull_vertices(m, TOL=0)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cuts_valid_on_hull_vertices_with_tolerance(self):
        m = models.makeTwoTermDisj_boxes()
        TransformationFactory('gdp.cuttingplane').apply_to(
            m, back_off_problem_tolerance=2e-8, verbose=True)

        self.check_cuts_valid_on_hull_vertices(m, TOL=1e-8)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cuts_valid_on_hull_vertices_inf_norm(self):
        m = models.makeTwoTermDisj_boxes()
        # we actually don't have to adjust the back-off problem tolerance for
        # this norm.
        TransformationFactory('gdp.cuttingplane').apply_to(
            m, norm=float('inf'), verbose=True)

        self.check_cuts_valid_on_hull_vertices(m, TOL=1e-8)
        
    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cuts_are_correct_facets_fme(self):
        m = models.makeTwoTermDisj_boxes()
        TransformationFactory('gdp.cuttingplane').apply_to(
            m, create_cuts=create_cuts_fme, post_process_cut=None, 
            zero_tolerance=0)
        # This would also be a valid cut, it just doesn't happen to be what we
        # choose.
        # facet_extreme_pts = [
        #     (1,0,3,1),
        #     (1,0,3,2),
        #     (0,1,1,3),
        #     (0,1,1,4)
        # ]
        facet_extreme_pts = [
            (0,1,1,3),
            (0,1,2,3),
            (1,0,3,1),
            (1,0,4,1)
        ]
        
        cuts = m._pyomo_gdp_cuttingplane_transformation.cuts
        # Here, we get just one facet
        self.assertEqual(len(cuts), 1)
        cut = cuts[0]
        cut_expr = cut.body
        lower = cut.lower
        upper = cut.upper
        for pt in facet_extreme_pts:
            m.d[0].indicator_var.fix(pt[0])
            m.d[1].indicator_var.fix(pt[1])
            m.x.fix(pt[2])
            m.y.fix(pt[3])
            if lower is not None:
                self.assertEqual(value(lower), value(cut_expr))
            if upper is not None:
                self.assertEqual(value(upper), value(cut_expr))

    def check_cuts_are_correct_facets(self, m):
        cut1_tight_pts = [
            (1,0,3,1),
            (0,1,1,3)
        ]
        facet2_extreme_pts = [
            (1,0,3,1),
            (1,0,4,1)
        ]
        cuts = m._pyomo_gdp_cuttingplane_transformation.cuts
        # ESJ: In this version, we don't get the facets, but we still get two
        # cuts, and we check they are tight at points on the relevant facets.
        self.assertEqual(len(cuts), 2)
        cut = cuts[0]
        cut_expr = cut.body
        lower = cut.lower
        upper = cut.upper
        for pt in cut1_tight_pts:
            m.d[0].indicator_var.fix(pt[0])
            m.d[1].indicator_var.fix(pt[1])
            m.x.fix(pt[2])
            m.y.fix(pt[3])
            if lower is not None:
                # ESJ: I had to increase the tolerance here, but I guess I don't
                # care as long as the error is in the right direction... (Which
                # we do test when we check the cuts are valid on the hull
                # vertices)
                self.assertAlmostEqual(value(lower), value(cut_expr), places=6)
            if upper is not None:
                self.assertAlmostEqual(value(upper), value(cut_expr))

        cut = cuts[1]
        cut_expr = cut.body
        lower = cut.lower
        upper = cut.upper
        for pt in facet2_extreme_pts:
            m.d[0].indicator_var.fix(pt[0])
            m.d[1].indicator_var.fix(pt[1])
            m.x.fix(pt[2])
            m.y.fix(pt[3])
            if lower is not None:
                self.assertAlmostEqual(value(lower), value(cut_expr))
            if upper is not None:
                self.assertAlmostEqual(value(upper), value(cut_expr))

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cuts_are_correct_facets_with_tolerance(self):
        m = models.makeTwoTermDisj_boxes()
        TransformationFactory('gdp.cuttingplane').apply_to(m)
        self.check_cuts_are_correct_facets(m)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cuts_are_correct_facets_inf_norm(self):
        m = models.makeTwoTermDisj_boxes()
        TransformationFactory('gdp.cuttingplane').apply_to(m, norm=float('inf'))
        self.check_cuts_are_correct_facets(m)
   
    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_create_using(self):
        m = models.makeTwoTermDisj_boxes()
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

    # I'm doing this test with Gurobi because ipopt doesn't really catch this
    # problem as constraints are never *exactly* satisfied. (The problem being
    # that we need to consider exactly satisfied equalities interesting for
    # FME.)
    @unittest.skipIf('gurobi' not in solvers, "Gurobi solver not available")
    def test_equality_constraints_on_disjuncts_with_fme(self):
        m = models.oneVarDisj_2pts()
        m.obj.expr = m.x + m.disj1.indicator_var
        m.obj.sense = maximize

        TransformationFactory('gdp.cuttingplane').apply_to(
            m,
            create_cuts=create_cuts_fme,
            post_process_cut=None, verbose=True, solver='gurobi',
            # don't actually need this, but taking the excuse to set solver
            # options
            solver_options={'FeasibilityTol': 1e-8},
            cuts_name="cuts", bigM=5)

        # rBigM first iteration solve will give (x = 3, Y = 0.6). If we don't
        # catch equality constraints, we don't get a cut. But we need to get 
        # x + Y <= 1. (Where Y is the indicator that x = 0).
        self.assertEqual(len(m.cuts), 1)
        cut = m.cuts[0]
        self.assertEqual(cut.lower, 0)
        self.assertIsNone(cut.upper)
        repn = generate_standard_repn(cut.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 2)
        self.assertIs(repn.linear_vars[0], m.disj1.indicator_var)
        self.assertEqual(repn.linear_coefs[0], 1)
        self.assertIs(repn.linear_vars[1], m.x)
        self.assertEqual(repn.linear_coefs[1], -1)

class Grossmann_TestCases(unittest.TestCase):
    def check_cuts_valid_at_extreme_pts(self, m):
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
    def test_cut_valid_at_extreme_pts_fme(self):
        m = models.grossmann_oneDisj()
        TransformationFactory('gdp.cuttingplane').apply_to(
            m, create_cuts=create_cuts_fme, post_process_cut=None)

        self.check_cuts_valid_at_extreme_pts(m)

    # ESJ: This is a bit inconsistent because this one passes without a
    # tolerance... But since it does I'm tempted to leave it this way?
    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cut_valid_at_extreme_pts_projection(self):
        m = models.grossmann_oneDisj()
        TransformationFactory('gdp.cuttingplane').apply_to(m)

        self.check_cuts_valid_at_extreme_pts(m)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cut_valid_at_extreme_pts_inf_norm(self):
        m = models.grossmann_oneDisj()
        TransformationFactory('gdp.cuttingplane').apply_to(m, norm=float('inf'))

        self.check_cuts_valid_at_extreme_pts(m)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cut_is_correct_facet_fme(self):
        m = models.grossmann_oneDisj()
        TransformationFactory('gdp.cuttingplane').apply_to(
            m, create_cuts=create_cuts_fme, post_process_cut=None)
        cuts = m._pyomo_gdp_cuttingplane_transformation.cuts
        # ESJ: Again, for FME, we don't mind getting both the possible facets,
        # as long as they are beautiful.
        self.assertEqual(len(cuts), 2)
        # similar to the two boxes example, this is on the line where two facets
        # intersect
        facet2_extreme_points = [
            (1,0,2,10),
            (1,0,2,7),
            (0,1,10,0),
            (0,1,10,3)
        ]
        facet_extreme_points = [
            (1,0,2,10),
            (1,0,0,10),
            (0,1,8,3),
            (0,1,10,3)
        ]

        for pt in facet_extreme_points:
            m.x.fix(pt[2])
            m.y.fix(pt[3])
            m.disjunct1.indicator_var.fix(pt[0])
            m.disjunct2.indicator_var.fix(pt[1])
            self.assertEqual(value(cuts[0].lower), value(cuts[0].body))
        for pt in facet2_extreme_points:
            m.x.fix(pt[2])
            m.y.fix(pt[3])
            m.disjunct1.indicator_var.fix(pt[0])
            m.disjunct2.indicator_var.fix(pt[1])
            self.assertEqual(value(cuts[1].lower), value(cuts[1].body))

    def check_cut_is_correct_facet(self, m):
        cuts = m._pyomo_gdp_cuttingplane_transformation.cuts
        # ESJ: We do get to cuts, but they aren't the facets like FME
        self.assertEqual(len(cuts), 2)
        # similar to the two boxes example, this is on the line where two facets
        # intersect, we get cuts which intersect the two facets from FME. This
        # makes sense because these are angled.
        cut1_tight_points = [
            (1,0,2,10),
            (0,1,10,3)
        ]
        cut2_tight_points = [
            (1,0,2,10),
            (1,0,0,10)
        ]

        for pt in cut1_tight_points:
            m.x.fix(pt[2])
            m.y.fix(pt[3])
            m.disjunct1.indicator_var.fix(pt[0])
            m.disjunct2.indicator_var.fix(pt[1])
            self.assertAlmostEqual(value(cuts[0].lower), value(cuts[0].body),
                                   places=6)
        for pt in cut2_tight_points:
            m.x.fix(pt[2])
            m.y.fix(pt[3])
            m.disjunct1.indicator_var.fix(pt[0])
            m.disjunct2.indicator_var.fix(pt[1])
            self.assertAlmostEqual(value(cuts[1].lower), value(cuts[1].body),
                                   places=6)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cut_is_correct_facet_projection(self):
        m = models.grossmann_oneDisj()
        TransformationFactory('gdp.cuttingplane').apply_to(m)
        self.check_cut_is_correct_facet(m)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cut_is_correct_facet_inf_norm(self):
        m = models.grossmann_oneDisj()
        # without the increase of cut_filtering_threshold, we get a third cut,
        # whcih is also tight where cut 2 is. It doesn't improve the objective
        # by much at all, so it's redundant.
        TransformationFactory('gdp.cuttingplane').apply_to(
            m, norm=float('inf'), cut_filtering_threshold=0.2)
        self.check_cut_is_correct_facet(m)

    def check_cuts_valid_at_extreme_pts_rescaled(self, m):
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
    def test_cuts_valid_at_extreme_pts_rescaled_fme(self):
        m = models.to_break_constraint_tolerances()
        TransformationFactory('gdp.cuttingplane').apply_to(
            m, create_cuts=create_cuts_fme, post_process_cut=None)
        self.check_cuts_valid_at_extreme_pts_rescaled(m)

    # Again, this actually passes without tolerance, so leaving it for now...
    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cuts_valid_at_extreme_pts_rescaled_projection(self):
        m = models.to_break_constraint_tolerances()
        TransformationFactory('gdp.cuttingplane').apply_to(m)
        self.check_cuts_valid_at_extreme_pts_rescaled(m)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cuts_valid_at_extreme_pts_rescaled_inf_norm(self):
        m = models.to_break_constraint_tolerances()
        # this cuts off by a little more than 1e-8 without the adjusted back-off
        # problem tolerance
        TransformationFactory('gdp.cuttingplane').apply_to(
            m, norm=float('inf'), back_off_problem_tolerance=1e-7, verbose=True)
        self.check_cuts_valid_at_extreme_pts_rescaled(m)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cut_is_correct_facet_rescaled_fme(self):
        m = models.to_break_constraint_tolerances()
        TransformationFactory('gdp.cuttingplane').apply_to(
            m, create_cuts=create_cuts_fme, post_process_cut=None)

        cuts = m._pyomo_gdp_cuttingplane_transformation.cuts
        self.assertEqual(len(cuts), 1)
        
        # we don't get a whole facet. We get 0 <= 129y_1 + 123y_2 - x - y, which
        # is the sum of two facets: 
        # 0 <= 2y_1 + 120y_2 - x and 
        # 0 <= 127y_1 + 3y_2 - y
        # But this is valid and the only cut needed, so we won't complain.
        cut_extreme_points = [
            (1,0,2,127),
            (0,1,120,3)
        ]

        for pt in cut_extreme_points:
            m.x.fix(pt[2])
            m.y.fix(pt[3])
            m.disjunct1.indicator_var.fix(pt[0])
            m.disjunct2.indicator_var.fix(pt[1])
            # tiny bit of numerical error
            self.assertAlmostEqual(value(cuts[0].lower), value(cuts[0].body))
            self.assertLessEqual(value(cuts[0].lower), value(cuts[0].body))

    def check_cut_is_correct_facet_rescaled(self, m):
        cuts = m._pyomo_gdp_cuttingplane_transformation.cuts
        self.assertEqual(len(cuts), 1)
        
        cut_tight_points = [
            (1,0,2,127),
            (0,1,120,3)
        ]

        for pt in cut_tight_points:
            m.x.fix(pt[2])
            m.y.fix(pt[3])
            m.disjunct1.indicator_var.fix(pt[0])
            m.disjunct2.indicator_var.fix(pt[1])
            # ESJ: 5 places is not ideal... But it's in the direction of valid,
            # so I think that's just the price we pay. This test still seems
            # useful to me as a sanity check that the cut is where it should be.
            self.assertAlmostEqual(value(cuts[0].lower), value(cuts[0].body),
                                   places=5)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cut_is_correct_facet_rescaled_projection(self):
        m = models.to_break_constraint_tolerances()
        TransformationFactory('gdp.cuttingplane').apply_to(m)
        self.check_cut_is_correct_facet_rescaled(m)        

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cut_is_correct_facet_rescaled_inf_norm(self):
        m = models.to_break_constraint_tolerances()
        # This would give two cuts, the second improving by about 0.05, without
        # the tighter threshold.
        TransformationFactory('gdp.cuttingplane').apply_to(
            m, norm=float('inf'), cut_filtering_threshold=0.1)
        self.check_cut_is_correct_facet_rescaled(m)

    def check_2disj_cuts_valid_for_extreme_pts(self, m):
        extreme_points = [
            (1,0,1,0,1,7),
            (1,0,1,0,1,8),
            (1,0,1,0,2,7),
            (1,0,1,0,2,8),
            (0,1,0,1,9,2),
            (0,1,0,1,9,3),
            (0,1,0,1,10,2),
            (0,1,0,1,10,3)
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

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_2disj_cuts_valid_for_extreme_pts_fme(self):
        m = models.grossmann_twoDisj()
        TransformationFactory('gdp.cuttingplane').apply_to(
            m, create_cuts=create_cuts_fme, post_process_cut=None)

        self.check_2disj_cuts_valid_for_extreme_pts(m)

    # This passes without a tolerance.
    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_2disj_cuts_valid_for_extreme_pts_projection(self):
        m = models.grossmann_twoDisj()
        TransformationFactory('gdp.cuttingplane').apply_to(m)

        self.check_2disj_cuts_valid_for_extreme_pts(m)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_2disj_cuts_valid_for_extreme_pts_inf_norm(self):
        m = models.grossmann_twoDisj()
        TransformationFactory('gdp.cuttingplane').apply_to(m, norm=float('inf'))

        self.check_2disj_cuts_valid_for_extreme_pts(m)

class NonlinearConvex_TwoCircles(unittest.TestCase):
    def check_cuts_valid_for_optimal(self, m):
        cuts = m._pyomo_gdp_cuttingplane_transformation.cuts
        self.assertGreaterEqual(len(cuts), 1) # we should get at least one.

        m.x.fix(2)
        m.y.fix(7)
        m.upper_circle.indicator_var.fix(1)
        m.lower_circle.indicator_var.fix(0)
        for i in range(len(cuts)):
            self.assertGreaterEqual(value(cuts[i].body), 0)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cuts_valid_for_optimal(self):
        m = models.twoDisj_twoCircles_easy()
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=1e6)

        self.check_cuts_valid_for_optimal(m)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cuts_valid_for_optimal_fme(self):
        m = models.twoDisj_twoCircles_easy()
        TransformationFactory('gdp.cuttingplane').apply_to(
            m, bigM=1e6, create_cuts=create_cuts_fme, verbose=True)

        self.check_cuts_valid_for_optimal(m)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cuts_valid_for_optimal_inf_norm(self):
        m = models.twoDisj_twoCircles_easy()
        TransformationFactory('gdp.cuttingplane').apply_to(
            m, bigM=1e6, norm=float('inf'), verbose=True)

        self.check_cuts_valid_for_optimal(m)

    def check_cuts_valid_on_facet_containing_optimal(self, m):
        cuts = m._pyomo_gdp_cuttingplane_transformation.cuts
        self.assertGreaterEqual(len(cuts), 1) # we should get at least one.

        m.x.fix(5)
        m.y.fix(3)
        m.upper_circle.indicator_var.fix(0)
        m.lower_circle.indicator_var.fix(1)
        for i in range(len(cuts)):
            self.assertTrue(value(cuts[i].expr))

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cuts_valid_on_facet_containing_optimal(self):
        m = models.twoDisj_twoCircles_easy()
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=1e6)
        self.check_cuts_valid_on_facet_containing_optimal(m)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cuts_valid_on_facet_containing_optimal_fme(self):
        m = models.twoDisj_twoCircles_easy()
        TransformationFactory('gdp.cuttingplane').apply_to(
            m, bigM=1e6, create_cuts=create_cuts_fme, verbose=True)
        self.check_cuts_valid_on_facet_containing_optimal(m)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cuts_valid_on_facet_containing_optimal_inf_norm(self):
        m = models.twoDisj_twoCircles_easy()
        TransformationFactory('gdp.cuttingplane').apply_to(
            m, bigM=1e6, norm=float('inf'), verbose=True)
        self.check_cuts_valid_on_facet_containing_optimal(m)

    def check_cuts_valid_for_other_extreme_points(self, m):
        cuts = m._pyomo_gdp_cuttingplane_transformation.cuts
        self.assertGreaterEqual(len(cuts), 1) # we should get at least one.

        m.x.fix(3)
        m.y.fix(1)
        m.upper_circle.indicator_var.fix(1)
        m.lower_circle.indicator_var.fix(0)
        for i in range(len(cuts)):
            self.assertGreaterEqual(value(cuts[i].body), 0)

        m.x.fix(0)
        m.y.fix(5)
        m.upper_circle.indicator_var.fix(0)
        m.lower_circle.indicator_var.fix(1)
        for i in range(len(cuts)):
            self.assertGreaterEqual(value(cuts[i].body), 0)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cuts_valid_for_other_extreme_points(self):
        # testing that we don't cut off anything on "the other side" (of the R^2
        # picture). There's little reason we should, but this is also a sanity
        # check that the cuts are in the correct direction. (Which one can lose
        # confidence about in the case of numerical difficulties...)
        m = models.twoDisj_twoCircles_easy()
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=1e6)
        self.check_cuts_valid_for_other_extreme_points(m)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cuts_valid_for_other_extreme_points_fme(self):
        # testing that we don't cut off anything on "the other side" (of the R^2
        # picture). There's little reason we should, but this is also a sanity
        # check that the cuts are in the correct direction. (Which one can lose
        # confidence about in the case of numerical difficulties...)
        m = models.twoDisj_twoCircles_easy()
        TransformationFactory('gdp.cuttingplane').apply_to(
            m, bigM=1e6, create_cuts=create_cuts_fme, verbose=True)
        self.check_cuts_valid_for_other_extreme_points(m)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cuts_valid_for_other_extreme_points_inf_norm(self):
        # testing that we don't cut off anything on "the other side" (of the R^2
        # picture). There's little reason we should, but this is also a sanity
        # check that the cuts are in the correct direction. (Which one can lose
        # confidence about in the case of numerical difficulties...)
        m = models.twoDisj_twoCircles_easy()
        TransformationFactory('gdp.cuttingplane').apply_to(
            m, bigM=1e6, norm=float('inf'), cut_filtering_threshold=0.5)
        self.check_cuts_valid_for_other_extreme_points(m)
            
    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cuts_valid_for_optimal_tighter_m(self):
        m = models.twoDisj_twoCircles_easy()

        # this M comes from the fact that y \in (0,8) and x \in (0,6)
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=83)
        self.check_cuts_valid_for_optimal(m)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cuts_valid_for_optimal_tighter_m_inf_norm(self):
        m = models.twoDisj_twoCircles_easy()

        # this M comes from the fact that y \in (0,8) and x \in (0,6)
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=83,
                                                           norm=float('inf'))
        self.check_cuts_valid_for_optimal(m)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cuts_valid_for_optimal_tighter_m_fme(self):
        m = models.twoDisj_twoCircles_easy()

        # this M comes from the fact that y \in (0,8) and x \in (0,6)
        TransformationFactory('gdp.cuttingplane').apply_to(
            m, bigM=83, create_cuts=create_cuts_fme)
        self.check_cuts_valid_for_optimal(m)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cuts_valid_for_optimalFacet_tighter_m(self):
        m = models.twoDisj_twoCircles_easy()

        # this M comes from the fact that y \in (0,8) and x \in (0,6)
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=83)
        self.check_cuts_valid_on_facet_containing_optimal(m)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cuts_valid_for_optimalFacet_tighter_m_fme(self):
        m = models.twoDisj_twoCircles_easy()

        # this M comes from the fact that y \in (0,8) and x \in (0,6)
        TransformationFactory('gdp.cuttingplane').apply_to(
            m, bigM=83, create_cuts=create_cuts_fme)
        self.check_cuts_valid_on_facet_containing_optimal(m)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cuts_valid_for_optimalFacet_tighter_m_inf_norm(self):
        m = models.twoDisj_twoCircles_easy()

        # this M comes from the fact that y \in (0,8) and x \in (0,6)
        TransformationFactory('gdp.cuttingplane').apply_to(
            m, bigM=83, norm=float('inf'))
        self.check_cuts_valid_on_facet_containing_optimal(m)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cuts_valid_for_other_extreme_points_tighter_m(self):
        m = models.twoDisj_twoCircles_easy()
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=83)
        self.check_cuts_valid_for_other_extreme_points(m)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cuts_valid_for_other_extreme_points_tighter_m_fme(self):
        m = models.twoDisj_twoCircles_easy()
        TransformationFactory('gdp.cuttingplane').apply_to(
            m, bigM=83, create_cuts=create_cuts_fme)
        self.check_cuts_valid_for_other_extreme_points(m)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cuts_valid_for_other_extreme_points_tighter_m_inf_norm(self):
        m = models.twoDisj_twoCircles_easy()
        TransformationFactory('gdp.cuttingplane').apply_to(
            m, bigM=83, norm=float('inf'), cut_filtering_threshold=0.5)
        self.check_cuts_valid_for_other_extreme_points(m)
        
class NonlinearConvex_OverlappingCircles(unittest.TestCase):  
    def check_cuts_valid_for_optimal(self, m):
        cuts = m._pyomo_gdp_cuttingplane_transformation.cuts
        self.assertGreaterEqual(len(cuts), 1) # we should get at least one.
        
        m.x.fix(2)
        m.y.fix(7)
        m.upper_circle.indicator_var.fix(1)
        m.lower_circle.indicator_var.fix(0)
        m.upper_circle2.indicator_var.fix(1)
        m.lower_circle2.indicator_var.fix(0)
        for i in range(len(cuts)):
            self.assertGreaterEqual(value(cuts[i].body), 0)
      
    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cuts_valid_for_optimal(self):
        m = models.fourCircles()
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=1e6)
        self.check_cuts_valid_for_optimal(m)
        
    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cuts_valid_for_optimal_fme(self):
        m = models.fourCircles()
        TransformationFactory('gdp.cuttingplane').apply_to(
            m, bigM=1e6, create_cuts=create_cuts_fme)
        self.check_cuts_valid_for_optimal(m)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cuts_valid_for_optimal_inf_norm(self):
        m = models.fourCircles()
        TransformationFactory('gdp.cuttingplane').apply_to(
            m, bigM=1e6, norm=float('inf'))
        self.check_cuts_valid_for_optimal(m)

    def check_cuts_valid_on_facet_containing_optimal(self, m):
        cuts = m._pyomo_gdp_cuttingplane_transformation.cuts
        self.assertGreaterEqual(len(cuts), 1) # we should get at least one.

        m.x.fix(5)
        m.y.fix(3)
        m.upper_circle.indicator_var.fix(0)
        m.lower_circle.indicator_var.fix(1)
        m.upper_circle2.indicator_var.fix(0)
        m.lower_circle2.indicator_var.fix(1)
        for i in range(len(cuts)):
            self.assertGreaterEqual(value(cuts[i].body), 0)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cuts_valid_on_facet_containing_optimal(self):
        m = models.fourCircles()
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=1e6)
        self.check_cuts_valid_on_facet_containing_optimal(m)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cuts_valid_on_facet_containing_optimal_fme(self):
        m = models.fourCircles()
        TransformationFactory('gdp.cuttingplane').apply_to(
            m, bigM=1e6,create_cuts=create_cuts_fme)
        self.check_cuts_valid_on_facet_containing_optimal(m)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cuts_valid_on_facet_containing_optimal_inf_norm(self):
        m = models.fourCircles()
        TransformationFactory('gdp.cuttingplane').apply_to(
            m, bigM=1e6, norm=float('inf'))
        self.check_cuts_valid_on_facet_containing_optimal(m)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cuts_valid_for_optimal_tightM(self):
        m = models.fourCircles()
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=83)
        self.check_cuts_valid_for_optimal(m)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cuts_valid_for_optimal_tightM_fme(self):
        m = models.fourCircles()
        TransformationFactory('gdp.cuttingplane').apply_to(
            m, bigM=1e6,create_cuts=create_cuts_fme)
        self.check_cuts_valid_for_optimal(m)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cuts_valid_for_optimal_tightM_inf_norm(self):
        m = models.fourCircles()
        TransformationFactory('gdp.cuttingplane').apply_to(
            m, bigM=1e6, norm=float('inf'))
        self.check_cuts_valid_for_optimal(m)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cuts_valid_on_facet_containing_optimal_tightM(self):
        m = models.fourCircles()
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=83)
        self.check_cuts_valid_on_facet_containing_optimal(m)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cuts_valid_on_facet_containing_optimal_tightM_fme(self):
        m = models.fourCircles()
        TransformationFactory('gdp.cuttingplane').apply_to(
            m, bigM=1e6,create_cuts=create_cuts_fme)
        self.check_cuts_valid_on_facet_containing_optimal(m)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_cuts_valid_on_facet_containing_optimal_tightM_inf_norm(self):
        m = models.fourCircles()
        TransformationFactory('gdp.cuttingplane').apply_to(
            m, bigM=1e6, norm=float('inf'))
        self.check_cuts_valid_on_facet_containing_optimal(m)
