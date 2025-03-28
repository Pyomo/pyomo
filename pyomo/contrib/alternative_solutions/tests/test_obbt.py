#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import math

from pyomo.common.dependencies import numpy as numpy, numpy_available

if numpy_available:
    from numpy.testing import assert_array_almost_equal

import pyomo.environ as pyo
from pyomo.common import unittest

import pyomo.opt
from pyomo.contrib.alternative_solutions import (
    obbt_analysis_bounds_and_solutions,
    obbt_analysis,
)
import pyomo.contrib.alternative_solutions.tests.test_cases as tc

solvers = list(pyomo.opt.check_available_solvers("glpk", "gurobi", "appsi_gurobi"))
pytestmark = unittest.pytest.mark.parametrize("mip_solver", solvers)

timelimit = {"gurobi": "TimeLimit", "appsi_gurobi": "TimeLimit", "glpk": "tmlim"}


@unittest.pytest.mark.default
class TestOBBTUnit:

    @unittest.skipIf(not numpy_available, "Numpy not installed")
    def test_bad_solver(self, mip_solver):
        """
        Confirm that an exception is thrown with a bad solver name.
        """
        m = tc.get_2d_diamond_problem()
        try:
            obbt_analysis(m, solver="unknown_solver")
        except pyomo.common.errors.ApplicationError as e:
            pass

    @unittest.skipIf(not numpy_available, "Numpy not installed")
    def test_obbt_analysis(self, mip_solver):
        """
        Check that the correct bounds are found for a continuous problem.
        """
        m = tc.get_2d_diamond_problem()
        all_bounds = obbt_analysis(m, solver=mip_solver)
        assert all_bounds.keys() == m.continuous_bounds.keys()
        for var, bounds in all_bounds.items():
            assert_array_almost_equal(bounds, m.continuous_bounds[var])

    def test_obbt_error1(self, mip_solver):
        """
        ERROR: Cannot restrict variable list when warmstart is specified
        """
        m = tc.get_2d_diamond_problem()
        with unittest.pytest.raises(AssertionError):
            obbt_analysis_bounds_and_solutions(m, variables=[m.x], solver=mip_solver)

    @unittest.skipIf(not numpy_available, "Numpy not installed")
    def test_obbt_some_vars(self, mip_solver):
        """
        Check that the correct bounds are found for a continuous problem.
        """
        m = tc.get_2d_diamond_problem()
        all_bounds, solns = obbt_analysis_bounds_and_solutions(
            m, variables=[m.x], warmstart=False, solver=mip_solver
        )
        assert len(all_bounds) == 1
        assert len(solns) == 2 * len(all_bounds) + 1
        for var, bounds in all_bounds.items():
            assert_array_almost_equal(bounds, m.continuous_bounds[var])

    @unittest.skipIf(not numpy_available, "Numpy not installed")
    def test_obbt_continuous(self, mip_solver):
        """
        Check that the correct bounds are found for a continuous problem.
        """
        m = tc.get_2d_diamond_problem()
        all_bounds, solns = obbt_analysis_bounds_and_solutions(m, solver=mip_solver)
        assert len(solns) == 2 * len(all_bounds) + 1
        assert all_bounds.keys() == m.continuous_bounds.keys()
        for var, bounds in all_bounds.items():
            assert_array_almost_equal(bounds, m.continuous_bounds[var])

    @unittest.skipIf(not numpy_available, "Numpy not installed")
    def test_mip_rel_objective(self, mip_solver):
        """
        Check that relative mip gap constraints are added for a mip with indexed vars and constraints
        """
        m = tc.get_indexed_pentagonal_pyramid_mip()
        all_bounds, solns = obbt_analysis_bounds_and_solutions(
            m, rel_opt_gap=0.5, solver=mip_solver
        )
        assert len(solns) == 2 * len(all_bounds) + 1
        assert m._obbt.optimality_tol_rel.lb == unittest.pytest.approx(2.5)

    @unittest.skipIf(not numpy_available, "Numpy not installed")
    def test_mip_abs_objective(self, mip_solver):
        """
        Check that absolute mip gap constraints are added
        """
        m = tc.get_pentagonal_pyramid_mip()
        all_bounds, solns = obbt_analysis_bounds_and_solutions(
            m, abs_opt_gap=1.99, solver=mip_solver
        )
        assert len(solns) == 2 * len(all_bounds) + 1
        assert m._obbt.optimality_tol_abs.lb == unittest.pytest.approx(3.01)

    @unittest.skipIf(not numpy_available, "Numpy not installed")
    def test_obbt_warmstart(self, mip_solver):
        """
        Check that warmstarting works.
        """
        m = tc.get_2d_diamond_problem()
        m.x.value = 0
        m.y.value = 0
        all_bounds, solns = obbt_analysis_bounds_and_solutions(
            m, solver=mip_solver, warmstart=True, tee=False
        )
        assert len(solns) == 2 * len(all_bounds) + 1
        assert all_bounds.keys() == m.continuous_bounds.keys()
        for var, bounds in all_bounds.items():
            assert_array_almost_equal(bounds, m.continuous_bounds[var])

    @unittest.skipIf(not numpy_available, "Numpy not installed")
    def test_obbt_mip(self, mip_solver):
        """
        Check that bound tightening only occurs for continuous variables
        that can be tightened.
        """
        m = tc.get_bloated_pentagonal_pyramid_mip()
        all_bounds, solns = obbt_analysis_bounds_and_solutions(
            m, solver=mip_solver, tee=False
        )
        assert len(solns) == 2 * len(all_bounds) + 1
        bounds_tightened = False
        bounds_not_tightned = False
        for var, bounds in all_bounds.items():
            if bounds[0] > var.lb:
                bounds_tightened = True
            else:
                bounds_not_tightened = True
            if bounds[1] < var.ub:
                bounds_tightened = True
            else:
                bounds_not_tightened = True
        assert bounds_tightened
        assert bounds_not_tightened

    @unittest.skipIf(not numpy_available, "Numpy not installed")
    def test_obbt_unbounded(self, mip_solver):
        """
        Check that the correct bounds are found for an unbounded problem.
        """
        m = tc.get_2d_unbounded_problem()
        all_bounds, solns = obbt_analysis_bounds_and_solutions(m, solver=mip_solver)
        assert all_bounds.keys() == m.continuous_bounds.keys()
        num = 1
        for var, bounds in all_bounds.items():
            if not math.isinf(bounds[0]):
                num += 1
            if not math.isinf(bounds[1]):
                num += 1
            assert_array_almost_equal(bounds, m.continuous_bounds[var])
        assert len(solns) == num

    @unittest.skipIf(not numpy_available, "Numpy not installed")
    def test_bound_tightening(self, mip_solver):
        """
        Check that the correct bounds are found for a discrete problem where
        more restrictive bounds are implied by the constraints.
        """
        m = tc.get_implied_bound_ip()
        all_bounds, solns = obbt_analysis_bounds_and_solutions(m, solver=mip_solver)
        assert len(solns) == 2 * len(all_bounds) + 1
        assert all_bounds.keys() == m.var_bounds.keys()
        for var, bounds in all_bounds.items():
            assert_array_almost_equal(bounds, m.var_bounds[var])

    @unittest.skipIf(True, "Ignoring fragile test for solver timeout.")
    def test_no_time(self, mip_solver):
        """
        Check that the correct bounds are found for a discrete problem where
        more restrictive bounds are implied by the constraints.
        """
        m = tc.get_implied_bound_ip()
        with unittest.pytest.raises(RuntimeError):
            obbt_analysis_bounds_and_solutions(
                m, solver=mip_solver, solver_options={timelimit[mip_solver]: 0}
            )

    def test_bound_refinement(self, mip_solver):
        """
        Check that the correct bounds are found for a discrete problem where
        more restrictive bounds are implied by the constraints and constraints
        are added.
        """
        m = tc.get_implied_bound_ip()
        all_bounds, solns = obbt_analysis_bounds_and_solutions(
            m, solver=mip_solver, refine_discrete_bounds=True
        )
        assert len(solns) == 2 * len(all_bounds) + 1
        for var, bounds in all_bounds.items():
            if m.var_bounds[var][0] > var.lb:
                match = False
                for idx in m._obbt.bound_constraints:
                    const = m._obbt.bound_constraints[idx]
                    if var is const.body and bounds[0] == const.lb:
                        match = True
                        break
                assert match, "Constraint not found for {} lower bound {}".format(
                    var, bounds[0]
                )
            if m.var_bounds[var][1] < var.ub:
                match = False
                for idx in m._obbt.bound_constraints:
                    const = m._obbt.bound_constraints[idx]
                    if var is const.body and bounds[1] == const.ub:
                        match = True
                        break
                assert match, "Constraint not found for {} upper bound {}".format(
                    var, bounds[1]
                )

    def test_obbt_infeasible(self, mip_solver):
        """
        Check that code catches cases where the problem is infeasible.
        """
        m = tc.get_2d_diamond_problem()
        m.infeasible_constraint = pyo.Constraint(expr=m.x >= 10)
        with unittest.pytest.raises(Exception):
            obbt_analysis_bounds_and_solutions(m, solver=mip_solver)


if __name__ == "__main__":
    unittest.main()
