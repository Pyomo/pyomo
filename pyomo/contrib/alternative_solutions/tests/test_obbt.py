from numpy.testing import assert_array_almost_equal
import pytest

import pyomo.environ as pe
import pyomo.common.unittest as unittest

import pyomo.opt
from pyomo.contrib.alternative_solutions import obbt_analysis
import pyomo.contrib.alternative_solutions.tests.test_cases as tc

solvers = list(pyomo.opt.check_available_solvers("glpk", "gurobi", "appsi_gurobi"))
pytestmark = pytest.mark.parametrize("mip_solver", solvers)

timelimit={"gurobi":"TimeLimit", "appsi_gurobi":"TimeLimit", "glpk":"tmlim"}

@unittest.pytest.mark.default
class TestOBBTUnit:

    def test_obbt_error1(self, mip_solver):
        m = tc.get_2d_diamond_problem()
        with pytest.raises(AssertionError):
            obbt_analysis(m, variables=[m.x], solver=mip_solver)

    def test_obbt_some_vars(self, mip_solver):
        """
        Check that the correct bounds are found for a continuous problem.
        """
        m = tc.get_2d_diamond_problem()
        results = obbt_analysis(m, variables=[m.x], warmstart=False, solver=mip_solver)
        assert len(results) == 1
        for var, bounds in results.items():
            assert_array_almost_equal(bounds, m.continuous_bounds[var])

    def test_obbt_continuous(self, mip_solver):
        """
        Check that the correct bounds are found for a continuous problem.
        """
        m = tc.get_2d_diamond_problem()
        results = obbt_analysis(m, solver=mip_solver)
        assert results.keys() == m.continuous_bounds.keys()
        for var, bounds in results.items():
            assert_array_almost_equal(bounds, m.continuous_bounds[var])

    def test_mip_rel_objective(self, mip_solver):
        """
        Check that relative mip gap constraints are added for a mip with indexed vars and constraints
        """
        m = tc.get_indexed_pentagonal_pyramid_mip()
        results = obbt_analysis(m, rel_opt_gap=0.5)
        assert m._obbt.optimality_tol_rel.lb == pytest.approx(2.5)

    def test_mip_abs_objective(self, mip_solver):
        """
        Check that absolute mip gap constraints are added
        """
        m = tc.get_pentagonal_pyramid_mip()
        results = obbt_analysis(m, abs_opt_gap=1.99)
        assert m._obbt.optimality_tol_abs.lb == pytest.approx(3.01)

    def test_obbt_warmstart(self, mip_solver):
        """
        Check that warmstarting works.
        """
        m = tc.get_2d_diamond_problem()
        m.x.value = 0
        m.y.value = 0
        results = obbt_analysis(m, solver=mip_solver, warmstart=True, tee=False)
        assert results.keys() == m.continuous_bounds.keys()
        for var, bounds in results.items():
            assert_array_almost_equal(bounds, m.continuous_bounds[var])

    def test_obbt_mip(self, mip_solver):
        """
        Check that bound tightening only occurs for continuous variables
        that can be tightened.
        """
        m = tc.get_bloated_pentagonal_pyramid_mip()
        results = obbt_analysis(m, solver=mip_solver, tee=False)
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
        assert bounds_tightened
        assert bounds_not_tightened

    def test_obbt_unbounded(self, mip_solver):
        """
        Check that the correct bounds are found for an unbounded problem.
        """
        m = tc.get_2d_unbounded_problem()
        results = obbt_analysis(m, solver=mip_solver)
        assert results.keys() == m.continuous_bounds.keys()
        for var, bounds in results.items():
            assert_array_almost_equal(bounds, m.continuous_bounds[var])

    def test_bound_tightening(self, mip_solver):
        """
        Check that the correct bounds are found for a discrete problem where
        more restrictive bounds are implied by the constraints.
        """
        m = tc.get_implied_bound_ip()
        results = obbt_analysis(m, solver=mip_solver)
        assert results.keys() == m.var_bounds.keys()
        for var, bounds in results.items():
            assert_array_almost_equal(bounds, m.var_bounds[var])

    def test_no_time(self, mip_solver):
        """
        Check that the correct bounds are found for a discrete problem where
        more restrictive bounds are implied by the constraints.
        """
        m = tc.get_implied_bound_ip()
        with pytest.raises(RuntimeError):
            obbt_analysis(m, solver=mip_solver, solver_options={timelimit[mip_solver]: 0})

    def test_bound_refinement(self, mip_solver):
        """
        Check that the correct bounds are found for a discrete problem where
        more restrictive bounds are implied by the constraints and constraints
        are added.
        """
        m = tc.get_implied_bound_ip()
        results = obbt_analysis(m, solver=mip_solver, refine_discrete_bounds=True)
        for var, bounds in results.items():
            if m.var_bounds[var][0] > var.lb:
                assert hasattr(m._obbt, var.name + "_lb")
            if m.var_bounds[var][1] < var.ub:
                assert hasattr(m._obbt, var.name + "_ub")

    def test_obbt_infeasible(self, mip_solver):
        """
        Check that code catches cases where the problem is infeasible.
        """
        m = tc.get_2d_diamond_problem()
        m.infeasible_constraint = pe.Constraint(expr=m.x >= 10)
        with pytest.raises(Exception):
            obbt_analysis(m, solver=mip_solver)


if __name__ == "__main__":
    unittest.main()
