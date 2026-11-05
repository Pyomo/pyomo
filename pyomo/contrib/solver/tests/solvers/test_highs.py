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

from pyomo.common.log import LoggingIntercept
from pyomo.contrib.solver.solvers.highs import Highs

opt = Highs()
if not opt.available():
    raise unittest.SkipTest


@unittest.pytest.mark.solver("highs")
class TestBugs(unittest.TestCase):
    def test_mutable_params_with_remove_cons(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(-10, 10))
        m.y = pyo.Var()

        m.p1 = pyo.Param(mutable=True)
        m.p2 = pyo.Param(mutable=True)

        m.obj = pyo.Objective(expr=m.y)
        m.c1 = pyo.Constraint(expr=m.y >= m.x + m.p1)
        m.c2 = pyo.Constraint(expr=m.y >= -m.x + m.p2)

        m.p1.value = 1
        m.p2.value = 1

        opt = Highs()
        res = opt.solve(m)
        self.assertAlmostEqual(res.objective_bound, 1)

        del m.c1
        m.p2.value = 2
        res = opt.solve(m)
        self.assertAlmostEqual(res.objective_bound, -8)

    def test_mutable_params_with_remove_vars(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()

        m.p1 = pyo.Param(mutable=True)
        m.p2 = pyo.Param(mutable=True)

        m.y.setlb(m.p1)
        m.y.setub(m.p2)

        m.obj = pyo.Objective(expr=m.y)
        m.c1 = pyo.Constraint(expr=m.y >= m.x + 1)
        m.c2 = pyo.Constraint(expr=m.y >= -m.x + 1)

        m.p1.value = -10
        m.p2.value = 10

        opt = Highs()
        res = opt.solve(m)
        self.assertAlmostEqual(res.objective_bound, 1)

        del m.c1
        del m.c2
        m.p1.value = -9
        m.p2.value = 9
        res = opt.solve(m)
        self.assertAlmostEqual(res.objective_bound, -9)

    def test_fix_and_unfix(self):
        # Tests issue https://github.com/Pyomo/pyomo/issues/3127

        m = pyo.ConcreteModel()
        m.x = pyo.Var(domain=pyo.Binary)
        m.y = pyo.Var(domain=pyo.Binary)
        m.fx = pyo.Var(domain=pyo.NonNegativeReals)
        m.fy = pyo.Var(domain=pyo.NonNegativeReals)
        m.c1 = pyo.Constraint(expr=m.fx <= m.x)
        m.c2 = pyo.Constraint(expr=m.fy <= m.y)
        m.c3 = pyo.Constraint(expr=m.x + m.y <= 1)

        m.obj = pyo.Objective(expr=m.fx * 0.5 + m.fy * 0.4, sense=pyo.maximize)

        opt = Highs()

        # solution 1 has m.x == 1 and m.y == 0
        r = opt.solve(m)
        self.assertAlmostEqual(m.fx.value, 1, places=5)
        self.assertAlmostEqual(m.fy.value, 0, places=5)
        self.assertAlmostEqual(r.objective_bound, 0.5, places=5)

        # solution 2 has m.x == 0 and m.y == 1
        m.y.fix(1)
        r = opt.solve(m)
        self.assertAlmostEqual(m.fx.value, 0, places=5)
        self.assertAlmostEqual(m.fy.value, 1, places=5)
        self.assertAlmostEqual(r.objective_bound, 0.4, places=5)

        # solution 3 should be equal solution 1
        m.y.unfix()
        m.x.fix(1)
        r = opt.solve(m)
        self.assertAlmostEqual(m.fx.value, 1, places=5)
        self.assertAlmostEqual(m.fy.value, 0, places=5)
        self.assertAlmostEqual(r.objective_bound, 0.5, places=5)


@unittest.pytest.mark.solver("highs")
class TestWarmStart(unittest.TestCase):
    def make_model(self):
        m = pyo.ConcreteModel()

        # decision variables
        m.x1 = pyo.Var(domain=pyo.Integers, name="x1", bounds=(0, 10))
        m.x2 = pyo.Var(domain=pyo.Reals, name="x2", bounds=(0, 10))
        m.x3 = pyo.Var(domain=pyo.Binary, name="x3")

        # objective function
        m.obj = pyo.Objective(expr=3 * m.x1 + 2 * m.x2 + 4 * m.x3, sense=pyo.maximize)

        # constraints
        m.c1 = pyo.Constraint(expr=m.x1 + m.x2 <= 9)
        m.c2 = pyo.Constraint(expr=3 * m.x1 + m.x2 <= 18)
        m.c3 = pyo.Constraint(expr=m.x1 <= 7)
        m.c4 = pyo.Constraint(expr=m.x2 <= 6)

        return m

    @unittest.skipUnless(
        opt.version()[:2] >= (1, 8), "Partial MIP starts require HiGHS>=1.8"
    )
    def test_warm_start(self):
        m = self.make_model()

        # MIP start
        m.x1 = 4
        m.x2 = 4.5
        m.x3 = 1

        # solving process
        res = Highs().solve(m, warmstart_discrete_vars=True)
        # Only the discrete variables are passed to HiGHS, which then solves
        # an LP for x2 (x2 = 5, objective value 26). If x2 = 4.5 were passed
        # as well, the objective value would be 25.
        self.assertIn(
            "MIP start solution is feasible, objective value is 26", res.solver_log
        )

    @unittest.skipUnless(
        opt.version()[:2] >= (1, 8), "Partial MIP starts require HiGHS>=1.8"
    )
    def test_partial_warm_start(self):
        m = self.make_model()

        # partial MIP start: x3 is left unset
        m.x1 = 4
        m.x2 = 4.5

        # We add one more constraint compared to test_warm_start:
        # with x1 = 4, the constraint forces x3 = 1, so the start is
        # infeasible if the unset x3 is passed to HiGHS as 0.
        # NOTE: a tighter c5 (e.g. x1 <= 3 + x3) lets presolve solve the model
        # before the start is used, so the "MIP start" line is never printed
        # and the test fails.
        m.c5 = pyo.Constraint(expr=m.x1 <= 3 + 7 * m.x3)

        # solving process
        res = Highs().solve(m, warmstart_discrete_vars=True)
        # We just check whether the MIP start solution is feasible, not for an
        # actual objective value, since that depends on how HiGHS completes the
        # partial start.
        self.assertIn("MIP start solution is feasible", res.solver_log)

    @unittest.skipUnless(
        opt.version()[:2] >= (1, 8), "Partial MIP starts require HiGHS>=1.8"
    )
    def test_warm_start_from_previous_results(self):
        m = self.make_model()
        opt = Highs()

        # first solve: same MIP start as in test_warm_start
        m.x1 = 4
        m.x3 = 1
        res = opt.solve(m, warmstart_discrete_vars=True)
        self.assertIn(
            "MIP start solution is feasible, objective value is 26", res.solver_log
        )

        # Second solve on the same persistent instance: x3 = 1 is still set
        # from the loaded solution, x1 is changed. HiGHS then solves an LP for
        # x2 (x2 = 6, objective value 25). A start left over from the first
        # solve would give 26 again.
        m.x1 = 3
        res = opt.solve(m, warmstart_discrete_vars=True)
        self.assertIn(
            "MIP start solution is feasible, objective value is 25", res.solver_log
        )

    @unittest.skipUnless(
        opt.version()[:2] >= (1, 8), "Partial MIP starts require HiGHS>=1.8"
    )
    def test_warm_start_resolve_drops_unset_values(self):
        m = self.make_model()
        opt = Highs()

        # same constraint as in test_partial_warm_start: x1 = 4 forces x3 = 1
        m.c5 = pyo.Constraint(expr=m.x1 <= 3 + 7 * m.x3)

        # first solve: x3 = 0 is part of the MIP start
        m.x1 = 3
        m.x3 = 0
        res = opt.solve(m, warmstart_discrete_vars=True)
        self.assertIn("MIP start solution is feasible", res.solver_log)

        # Second solve on the same persistent instance with x3 unset: the
        # start is infeasible if x3 keeps its value 0 from the first start,
        # since x1 = 4 forces x3 = 1.
        m.x1 = 4
        m.x3 = None
        res = opt.solve(m, warmstart_discrete_vars=True)
        self.assertIn("MIP start solution is feasible", res.solver_log)

    def test_warm_start_warns_and_skips_for_highs_below_1_8(self):
        class HighsVersion17(Highs):
            def version(self):
                return (1, 7, 0)

        m = self.make_model()
        m.x1 = 4
        m.x3 = 1

        with LoggingIntercept() as LOG:
            res = HighsVersion17().solve(m, warmstart_discrete_vars=True)
        self.assertIn("Partial MIP starts require HiGHS >= 1.8", LOG.getvalue())
        self.assertNotIn("MIP start", res.solver_log)
