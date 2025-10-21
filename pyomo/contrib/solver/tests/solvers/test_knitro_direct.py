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

import io
from contextlib import redirect_stdout

import pyomo.common.unittest as unittest
import pyomo.environ as pyo

from pyomo.contrib.solver.common.results import SolutionStatus, TerminationCondition
from pyomo.contrib.solver.solvers.knitro.config import KnitroConfig
from pyomo.contrib.solver.solvers.knitro.direct import KnitroDirectSolver

avail = KnitroDirectSolver().available()


@unittest.skipIf(not avail, "KNITRO solver is not available")
class TestKnitroDirectSolverConfig(unittest.TestCase):
    def test_default_instantiation(self):
        config = KnitroConfig()
        self.assertIsNone(config._description)
        self.assertEqual(config._visibility, 0)
        self.assertFalse(config.tee)
        self.assertTrue(config.load_solutions)
        self.assertTrue(config.raise_exception_on_nonoptimal_result)
        self.assertFalse(config.symbolic_solver_labels)
        self.assertIsNone(config.timer)
        self.assertIsNone(config.threads)
        self.assertIsNone(config.time_limit)

    def test_custom_instantiation(self):
        config = KnitroConfig(description="A description")
        config.tee = True
        self.assertTrue(config.tee)
        self.assertEqual(config._description, "A description")
        self.assertIsNone(config.time_limit)


@unittest.skipIf(not avail, "KNITRO solver is not available")
class TestKnitroDirectSolverInterface(unittest.TestCase):
    def test_class_member_list(self):
        opt = KnitroDirectSolver()
        expected_list = [
            "CONFIG",
            "available",
            "config",
            "api_version",
            "is_persistent",
            "name",
            "solve",
            "version",
        ]
        method_list = [
            m for m in dir(opt) if not m.startswith("_") and not m.startswith("get")
        ]
        self.assertListEqual(sorted(method_list), sorted(expected_list))

    def test_default_instantiation(self):
        opt = KnitroDirectSolver()
        self.assertFalse(opt.is_persistent())
        self.assertIsNotNone(opt.version())
        self.assertEqual(opt.name, "knitro_direct")
        self.assertEqual(opt.CONFIG, opt.config)
        self.assertTrue(opt.available())

    def test_instantiation_as_context(self):
        with KnitroDirectSolver() as opt:
            self.assertFalse(opt.is_persistent())
            self.assertIsNotNone(opt.version())
            self.assertEqual(opt.name, "knitro_direct")
            self.assertEqual(opt.CONFIG, opt.config)
            self.assertTrue(opt.available())

    def test_available_cache(self):
        opt = KnitroDirectSolver()
        opt.available()
        self.assertTrue(opt._available_cache)
        self.assertIsNotNone(opt._available_cache)

    def test_solution_status_mapping(self):
        opt = KnitroDirectSolver()
        for opt_status in [0, -100]:
            status = opt._get_solution_status(opt_status)
            self.assertEqual(status, SolutionStatus.optimal)

        for opt_status in [*range(-101, -103, -1), *range(-400, -406, -1)]:
            status = opt._get_solution_status(opt_status)
            self.assertEqual(status, SolutionStatus.feasible)

        for opt_status in [-200, -204, -205, -206]:
            status = opt._get_solution_status(opt_status)
            self.assertEqual(status, SolutionStatus.infeasible)

        for opt_status in [-501, -99999, -1]:
            status = opt._get_solution_status(opt_status)
            self.assertEqual(status, SolutionStatus.noSolution)

    def test_termination_condition_mapping(self):
        opt = KnitroDirectSolver()
        for opt_status in [0, -100]:
            term_cond = opt._get_termination_condition(opt_status)
            self.assertEqual(
                term_cond, TerminationCondition.convergenceCriteriaSatisfied
            )
        term_cond = opt._get_termination_condition(-202)
        self.assertEqual(term_cond, TerminationCondition.locallyInfeasible)
        for opt_status in [-200, -204, -205]:
            term_cond = opt._get_termination_condition(opt_status)
            self.assertEqual(term_cond, TerminationCondition.provenInfeasible)
        for opt_status in [-300, -301]:
            term_cond = opt._get_termination_condition(opt_status)
            self.assertEqual(term_cond, TerminationCondition.infeasibleOrUnbounded)
        for opt_status in [-400, -410]:
            term_cond = opt._get_termination_condition(opt_status)
            self.assertEqual(term_cond, TerminationCondition.iterationLimit)
        for opt_status in [-401, -411]:
            term_cond = opt._get_termination_condition(opt_status)
            self.assertEqual(term_cond, TerminationCondition.maxTimeLimit)
        term_cond = opt._get_termination_condition(-500)
        self.assertEqual(term_cond, TerminationCondition.interrupted)
        for opt_status in [-501, -550, -599]:
            term_cond = opt._get_termination_condition(opt_status)
            self.assertEqual(term_cond, TerminationCondition.error)
        for opt_status in [-600, -99999, -1]:
            term_cond = opt._get_termination_condition(opt_status)
            self.assertEqual(term_cond, TerminationCondition.unknown)


@unittest.skipIf(not avail, "KNITRO solver is not available")
class TestKnitroDirectSolver(unittest.TestCase):
    def setUp(self):
        self.opt = KnitroDirectSolver()

    def test_solve_tee(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=1.5, bounds=(-5, 5))
        m.y = pyo.Var(initialize=1.5, bounds=(-5, 5))
        m.obj = pyo.Objective(
            expr=(1.0 - m.x) + 100.0 * (m.y - m.x), sense=pyo.minimize
        )
        stream = io.StringIO()
        with redirect_stdout(stream):
            self.opt.solve(m, tee=True)
        output = stream.getvalue()
        self.assertTrue(bool(output.strip()))

    def test_solve_no_tee(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=1.5, bounds=(-5, 5))
        m.y = pyo.Var(initialize=1.5, bounds=(-5, 5))
        m.obj = pyo.Objective(
            expr=(1.0 - m.x) + 100.0 * (m.y - m.x), sense=pyo.minimize
        )
        stream = io.StringIO()
        with redirect_stdout(stream):
            self.opt.solve(m, tee=False)
        output = stream.getvalue()
        self.assertFalse(bool(output.strip()))

    def test_solve(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=1.5, bounds=(-5, 5))
        m.y = pyo.Var(initialize=1.5, bounds=(-5, 5))
        m.obj = pyo.Objective(
            expr=(1.0 - m.x) + 100.0 * (m.y - m.x), sense=pyo.minimize
        )
        res = self.opt.solve(m)
        self.assertAlmostEqual(res.incumbent_objective, -1004)
        self.assertAlmostEqual(m.x.value, 5)
        self.assertAlmostEqual(m.y.value, -5)

    def test_qp_solve(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=1.5, bounds=(-5, 5))
        m.y = pyo.Var(initialize=1.5, bounds=(-5, 5))
        m.obj = pyo.Objective(
            expr=(1.0 - m.x) + 100.0 * (m.y - m.x) ** 2, sense=pyo.minimize
        )
        results = self.opt.solve(m)
        self.assertAlmostEqual(results.incumbent_objective, -4.0, 3)
        self.assertAlmostEqual(m.x.value, 5.0, 3)
        self.assertAlmostEqual(m.y.value, 5.0, 3)

    def test_qcp_solve(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=1.5, bounds=(-5, 5))
        m.y = pyo.Var(initialize=1.5, bounds=(-5, 5))
        m.obj = pyo.Objective(expr=(m.y - m.x) ** 2, sense=pyo.minimize)
        m.c1 = pyo.Constraint(expr=m.x**2 + m.y**2 <= 4)
        results = self.opt.solve(m)
        self.assertAlmostEqual(results.incumbent_objective, 0.0)

    def test_solve_exp(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.obj = pyo.Objective(expr=m.x**2 + m.y**2)
        m.c1 = pyo.Constraint(expr=m.y >= pyo.exp(m.x))
        self.opt.solve(m)
        self.assertAlmostEqual(m.x.value, -0.42630274815985264)
        self.assertAlmostEqual(m.y.value, 0.6529186341994245)

    def test_solve_log(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=1)
        m.y = pyo.Var()
        m.obj = pyo.Objective(expr=m.x**2 + m.y**2)
        m.c1 = pyo.Constraint(expr=m.y <= pyo.log(m.x))
        self.opt.solve(m)
        self.assertAlmostEqual(m.x.value, 0.6529186341994245)
        self.assertAlmostEqual(m.y.value, -0.42630274815985264)

    def test_solve_HS071(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(pyo.RangeSet(1, 4), bounds=(1.0, 5.0))
        m.obj = pyo.Objective(
            expr=m.x[1] * m.x[4] * (m.x[1] + m.x[2] + m.x[3]) + m.x[3],
            sense=pyo.minimize,
        )
        m.c1 = pyo.Constraint(expr=m.x[1] * m.x[2] * m.x[3] * m.x[4] >= 25.0)
        m.c2 = pyo.Constraint(
            expr=m.x[1] ** 2 + m.x[2] ** 2 + m.x[3] ** 2 + m.x[4] ** 2 == 40.0
        )
        self.opt.solve(m, solver_options={"opttol": 1e-5})
        self.assertAlmostEqual(pyo.value(m.x[1]), 1.0, 3)
        self.assertAlmostEqual(pyo.value(m.x[2]), 4.743, 3)
        self.assertAlmostEqual(pyo.value(m.x[3]), 3.821, 3)
        self.assertAlmostEqual(pyo.value(m.x[4]), 1.379, 3)
