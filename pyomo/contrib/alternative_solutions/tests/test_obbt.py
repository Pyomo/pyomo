#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from numpy.testing import assert_array_almost_equal

import pyomo.environ as pe
import pyomo.common.unittest as unittest

from pyomo.contrib.alternative_solutions.obbt import obbt_analysis
import pyomo.contrib.alternative_solutions.tests.test_cases as tc
import pdb

mip_solver = "gurobi_appsi"
# mip_solver = 'gurobi'


class TestOBBTUnit(unittest.TestCase):

    # TODO: Add more test cases
    """
    So far I have added test cases for the feasibility problems, we should test cases
    where we put objective constraints in as well based on the absolute and relative difference.

    Add a case where bounds are only found for a subset of variables.

    Try cases where refine_discrete_bounds is set to true to ensure that new constraints are
    added to refine the bounds. I created the problem get_implied_bound_ip to facilitate this

    Check to see that warm starting works for a MIP and MILP case

    We should also check that warmstarting and refining bounds works for gurobi and appsi_gurobi

    We should pass at least one solver_options to ensure this work (e.g. time limit)

    I only looked at linear cases here, so you think others are worth testing, some simple non-linear (convex) cases?

    """

    def test_obbt_continuous(self):
        """Check that the correct bounds are found for a continuous problem."""
        m = tc.get_2d_diamond_problem()
        results = obbt_analysis(m, solver=mip_solver)
        self.assertEqual(results.keys(), m.continuous_bounds.keys())
        for var, bounds in results.items():
            assert_array_almost_equal(bounds, m.continuous_bounds[var])

    def test_mip_rel_objective(self):
        """Check that relative mip gap constraints are added for a mip with indexed vars and constraints"""
        m = tc.get_indexed_pentagonal_pyramid_mip()
        results = obbt_analysis(m, rel_opt_gap=0.5)
        self.assertAlmostEqual(m._obbt.optimality_tol_rel.lb, 2.5)

    def test_mip_abs_objective(self):
        """Check that absolute mip gap constraints are added"""
        m = tc.get_pentagonal_pyramid_mip()
        results = obbt_analysis(m, abs_opt_gap=1.99)
        self.assertAlmostEqual(m._obbt.optimality_tol_abs.lb, 3.01)

    def test_obbt_warmstart(self):
        """Check that warmstarting works."""
        m = tc.get_2d_diamond_problem()
        m.x.value = 0
        m.y.value = 0
        results = obbt_analysis(m, solver=mip_solver, warmstart=True, tee=True)
        self.assertEqual(results.keys(), m.continuous_bounds.keys())
        for var, bounds in results.items():
            assert_array_almost_equal(bounds, m.continuous_bounds[var])

    def test_obbt_mip(self):
        """Check that bound tightening only occurs for continuous variables
        that can be tightened."""
        m = tc.get_bloated_pentagonal_pyramid_mip()
        results = obbt_analysis(m, solver=mip_solver, tee=True)
        bounds_tightened = False
        bounds_not_tightned = False
        for var, bounds in results.items():
            if bounds[0] > var.lb:
                bounds_tightened = True
            else:
                bounds_not_tightened = True
            if bounds[1] < var.ub:
                bounds_tightened = True
            else:
                bounds_not_tightened = True
        self.assertTrue(bounds_tightened)
        self.assertTrue(bounds_not_tightened)

    def test_obbt_unbounded(self):
        """Check that the correct bounds are found for an unbounded problem."""
        m = tc.get_2d_unbounded_problem()
        results = obbt_analysis(m, solver=mip_solver)
        self.assertEqual(results.keys(), m.continuous_bounds.keys())
        for var, bounds in results.items():
            assert_array_almost_equal(bounds, m.continuous_bounds[var])

    def test_bound_tightening(self):
        """
        Check that the correct bounds are found for a discrete problem where
        more restrictive bounds are implied by the constraints."""
        m = tc.get_implied_bound_ip()
        results = obbt_analysis(m, solver=mip_solver)
        self.assertEqual(results.keys(), m.var_bounds.keys())
        for var, bounds in results.items():
            assert_array_almost_equal(bounds, m.var_bounds[var])

    def test_bound_refinement(self):
        """
        Check that the correct bounds are found for a discrete problem where
        more restrictive bounds are implied by the constraints and constraints
        are added."""
        m = tc.get_implied_bound_ip()
        results = obbt_analysis(m, solver=mip_solver, refine_discrete_bounds=True)
        for var, bounds in results.items():
            if m.var_bounds[var][0] > var.lb:
                self.assertTrue(hasattr(m._obbt, var.name + "_lb"))
            if m.var_bounds[var][1] < var.ub:
                self.assertTrue(hasattr(m._obbt, var.name + "_ub"))

    def test_obbt_infeasible(self):
        """Check that code catches cases where the problem is infeasible."""
        m = tc.get_2d_diamond_problem()
        m.infeasible_constraint = pe.Constraint(expr=m.x >= 10)
        with self.assertRaises(Exception):
            obbt_analysis(m, solver=mip_solver)


if __name__ == "__main__":
    unittest.main()
