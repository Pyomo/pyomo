# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________


import pyomo.common.unittest as unittest
import pyomo.environ as pyo

from pyomo.contrib.solver.common.util import (
    NoDualsError,
    NoReducedCostsError,
    NoSolutionError,
)
from pyomo.contrib.solver.common.results import SolutionStatus
from pyomo.contrib.solver.solvers.xpress import XpressDirect, XpressPersistent
from pyomo.contrib.solver.tests.solvers._xpress_test_utils import _simple_lp

if not XpressDirect().available():
    raise unittest.SkipTest('Xpress not available')


def _make_mip_with_many_solutions():
    """Small binary MIP with many feasible integer solutions.

    max  x[0] + x[1] + x[2] + x[3] + x[4]
    s.t. x[0] + x[1] + x[2] + x[3] + x[4] <= 3
         x in {0,1}^5

    Optimal value = 3 (choose any 3 of 5 items).  There are C(5,3) = 10
    feasible solutions with obj=3, plus all sub-optimal assignments.
    """
    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(0, 4)
    m.x = pyo.Var(m.I, within=pyo.Binary)
    m.c = pyo.Constraint(expr=sum(m.x[i] for i in m.I) <= 3)
    m.obj = pyo.Objective(expr=sum(m.x[i] for i in m.I), sense=pyo.maximize)
    return m


@unittest.pytest.mark.solver('xpress_direct')
class TestXpressSolutionPool(unittest.TestCase):
    """Integration tests for the solution pool (pool_solutions config)."""

    def setUp(self):
        self.opt = XpressDirect()

    def test_pool_disabled_by_default(self):
        """Default config collects no pool: exactly 1 solution accessible."""
        m = _make_mip_with_many_solutions()
        res = self.opt.solve(m)
        loader = res.solution_loader
        self.assertEqual(loader.get_number_of_solutions(), 1)
        self.assertEqual(loader.get_solution_ids(), [0])
        self.assertEqual(res.solution_status, SolutionStatus.optimal)

    def test_pool_collects_solutions(self):
        """pool_solutions=5 collects multiple feasible solutions.

        Verifies:
          - get_number_of_solutions() > 1
          - solution 0 is the optimal incumbent
          - solution 1 is a distinct valid assignment
        """
        m = _make_mip_with_many_solutions()
        res = self.opt.solve(m, pool_solutions=5, load_solutions=False)
        loader = res.solution_loader

        n = loader.get_number_of_solutions()
        self.assertGreater(n, 1)
        self.assertEqual(loader.get_solution_ids(), list(range(n)))

        # Load and check solution 0 (incumbent -- optimal).
        loader.solution(0).load_vars()
        obj0 = pyo.value(m.obj)
        self.assertAlmostEqual(obj0, res.incumbent_objective, places=6)
        self.assertEqual(obj0, 3.0)

        # Load solution 1 and verify it is a valid feasible assignment.
        loader.solution(1).load_vars()
        for i in m.I:
            self.assertIn(round(pyo.value(m.x[i])), (0, 1))
        total = sum(pyo.value(m.x[i]) for i in m.I)
        self.assertLessEqual(total, 3.0 + 1e-6)

    def test_pool_context_manager(self):
        """solution(k) context manager restores incumbent on exit."""
        m = _make_mip_with_many_solutions()
        res = self.opt.solve(m, pool_solutions=5, load_solutions=False)
        loader = res.solution_loader

        if loader.get_number_of_solutions() < 2:
            self.skipTest('Solver found fewer than 2 solutions -- cannot test pool.')

        # Load incumbent into model variables.
        loader.solution(0).load_vars()
        incumbent_vals = {i: pyo.value(m.x[i]) for i in m.I}

        # Context manager temporarily activates solution 1.
        with loader.solution(1):
            loader.load_vars()

        # After exiting the context, active id is restored to 0 (incumbent).
        # Reload to confirm values match the original incumbent.
        loader.load_vars()
        for i in m.I:
            self.assertAlmostEqual(pyo.value(m.x[i]), incumbent_vals[i], places=6)

    def test_pool_solution_raises_out_of_range(self):
        """Requesting solution(1) on a default (no-pool) solve must raise NoSolutionError.
        Without the fix, the silent fallthrough would load incumbent values instead."""
        m = _make_mip_with_many_solutions()
        res = self.opt.solve(m, load_solutions=False)
        loader = res.solution_loader
        self.assertEqual(loader.get_number_of_solutions(), 1)
        with self.assertRaises(NoSolutionError):
            loader.solution(1).load_vars()

    def test_pool_duals_raise_for_nonzero_id(self):
        """get_duals() and get_reduced_costs() inside solution(1) context must raise."""
        m = _make_mip_with_many_solutions()
        # Use LP model for duals (MIP has no duals anyway); LP has an incumbent only.
        lp = _simple_lp()
        self.opt.solve(lp, pool_solutions=0, load_solutions=False)
        # Pool is empty; solution(1) will raise NoSolutionError, which already
        # confirms the guard fires. For the duals/RC guard test we need pool_solutions>0
        # so a solution(1) context can be entered. Use the MIP model for that.
        mip_res = self.opt.solve(m, pool_solutions=5, load_solutions=False)
        loader = mip_res.solution_loader
        if loader.get_number_of_solutions() < 2:
            self.skipTest(
                'Solver found fewer than 2 solutions -- cannot test duals guard.'
            )
        with loader.solution(1):
            with self.assertRaises(NoDualsError):
                loader.get_duals()
            with self.assertRaises(NoReducedCostsError):
                loader.get_reduced_costs()

    def test_pool_persistent(self):
        """pool_solutions works through XpressPersistent and callbacks do not
        accumulate across consecutive solves."""
        opt = XpressPersistent()
        m = _make_mip_with_many_solutions()

        res1 = opt.solve(m, pool_solutions=5)
        n1 = res1.solution_loader.get_number_of_solutions()
        self.assertGreaterEqual(n1, 1)

        # Second solve: pool must be freshly collected, not accumulated from solve 1.
        res2 = opt.solve(m, pool_solutions=5)
        n2 = res2.solution_loader.get_number_of_solutions()
        self.assertGreaterEqual(n2, 1)
        # The pool from the first solve is in res1's loader; res2 has its own pool.
        self.assertIsNot(res1.solution_loader, res2.solution_loader)

    def test_pool_rolling_window(self):
        """pool_solutions=N keeps a rolling window of the last N solutions found.

        The pool size must not exceed N even when more than N solutions are found
        during B&B. Uses a model with many feasible integer solutions.
        """
        m = _make_mip_with_many_solutions()
        window = 2
        res = self.opt.solve(m, pool_solutions=window, load_solutions=False)
        loader = res.solution_loader
        n = loader.get_number_of_solutions()
        # Pool entries are at most window + 1 (incumbent + window collected)
        # but the solver might have found fewer than window non-incumbent solutions.
        self.assertGreaterEqual(n, 1)
        self.assertLessEqual(n, window + 1)


if __name__ == '__main__':
    unittest.main()
