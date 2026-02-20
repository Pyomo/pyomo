# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import io
from contextlib import redirect_stdout

import pyomo.common.unittest as unittest
import pyomo.environ as pyo

from pyomo.contrib.solver.common.results import SolutionStatus, TerminationCondition
from pyomo.contrib.solver.solvers.knitro.api import knitro
from pyomo.contrib.solver.solvers.knitro.config import KnitroConfig
from pyomo.contrib.solver.solvers.knitro.direct import KnitroDirectSolver
from pyomo.contrib.solver.solvers.knitro.engine import Engine

avail = KnitroDirectSolver().available()


@unittest.skipIf(not avail, "KNITRO solver is not available")
@unittest.pytest.mark.solver("knitro_direct")
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
@unittest.pytest.mark.solver("knitro_direct")
class TestKnitroSolverResultsExtraInfo(unittest.TestCase):
    def test_results_extra_info_mip(self):
        """Test that MIP-specific extra info is populated for MIP problems."""
        opt = KnitroDirectSolver()
        m = pyo.ConcreteModel()
        m.x = pyo.Var(domain=pyo.Integers, bounds=(0, 10))
        m.y = pyo.Var(domain=pyo.Integers, bounds=(0, 10))
        m.obj = pyo.Objective(expr=m.x + m.y, sense=pyo.maximize)
        m.c1 = pyo.Constraint(expr=2 * m.x + m.y <= 15)
        m.c2 = pyo.Constraint(expr=m.x + 2 * m.y <= 15)

        results = opt.solve(m)

        # Check that MIP-specific fields are populated
        self.assertIsNotNone(results.extra_info.mip_number_nodes)
        self.assertIsNotNone(results.extra_info.mip_abs_gap)
        self.assertIsNotNone(results.extra_info.mip_rel_gap)
        self.assertIsNotNone(results.extra_info.mip_number_solves)

        # Check that MIP-specific fields are of correct type
        self.assertIsInstance(results.extra_info.mip_number_nodes, int)
        self.assertIsInstance(results.extra_info.mip_abs_gap, float)
        self.assertIsInstance(results.extra_info.mip_rel_gap, float)
        self.assertIsInstance(results.extra_info.mip_number_solves, int)

        # Check that non-MIP field does not exist
        self.assertFalse(hasattr(results.extra_info, 'number_iters'))

    def test_results_extra_info_no_mip(self):
        """Test that iteration info is populated for non-MIP problems."""
        opt = KnitroDirectSolver()
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=1.5, bounds=(-5, 5))
        m.y = pyo.Var(initialize=1.5, bounds=(-5, 5))
        m.obj = pyo.Objective(
            expr=(1.0 - m.x) ** 2 + 100.0 * (m.y - m.x**2) ** 2, sense=pyo.minimize
        )

        results = opt.solve(m)

        # Check that num_iters is populated for non-MIP
        self.assertIsNotNone(results.extra_info.number_iters)
        self.assertIsInstance(results.extra_info.number_iters, int)
        self.assertGreater(results.extra_info.number_iters, 0)

        # Check that MIP-specific fields do not exist
        self.assertFalse(hasattr(results.extra_info, 'mip_number_nodes'))
        self.assertFalse(hasattr(results.extra_info, 'mip_abs_gap'))
        self.assertFalse(hasattr(results.extra_info, 'mip_rel_gap'))
        self.assertFalse(hasattr(results.extra_info, 'mip_number_solves'))


@unittest.skipIf(not avail, "KNITRO solver is not available")
@unittest.pytest.mark.solver("knitro_direct")
class TestKnitroSolverObjectiveBound(unittest.TestCase):
    def test_objective_bound_mip(self):
        """Test that objective bound is retrieved for MIP problems."""
        opt = KnitroDirectSolver()
        m = pyo.ConcreteModel()
        m.x = pyo.Var(domain=pyo.Integers, bounds=(0, 10))
        m.y = pyo.Var(domain=pyo.Integers, bounds=(0, 10))
        m.obj = pyo.Objective(expr=m.x + m.y, sense=pyo.maximize)
        m.c1 = pyo.Constraint(expr=2 * m.x + m.y <= 15)
        m.c2 = pyo.Constraint(expr=m.x + 2 * m.y <= 15)

        results = opt.solve(m)

        # Check that objective_bound is populated
        self.assertIsNotNone(results.objective_bound)
        self.assertIsInstance(results.objective_bound, float)

        # For maximization, bound should be >= incumbent objective
        self.assertGreaterEqual(results.objective_bound, results.incumbent_objective)

    def test_objective_bound_no_mip(self):
        """Test that objective bound is not set for non-MIP problems."""
        opt = KnitroDirectSolver()
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=1.5, bounds=(-5, 5))
        m.y = pyo.Var(initialize=1.5, bounds=(-5, 5))
        m.obj = pyo.Objective(
            expr=(1.0 - m.x) ** 2 + 100.0 * (m.y - m.x**2) ** 2, sense=pyo.minimize
        )

        results = opt.solve(m)

        # Check that objective_bound is None for non-MIP
        self.assertIsNone(results.objective_bound)


@unittest.skipIf(not avail, "KNITRO solver is not available")
@unittest.pytest.mark.solver("knitro_direct")
class TestKnitroSolverIncumbentObjective(unittest.TestCase):
    def test_none_without_objective(self):
        """Test that incumbent objective is None when no objective is present."""
        opt = KnitroDirectSolver()
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=1.5, bounds=(-5, 5))
        m.y = pyo.Var(initialize=1.5, bounds=(-5, 5))
        m.c1 = pyo.Constraint(expr=m.x + m.y >= 1)

        results = opt.solve(m)
        self.assertIsNone(results.incumbent_objective)

    def test_none_when_infeasible(self):
        """Test that incumbent objective is None when problem is infeasible."""
        opt = KnitroDirectSolver()
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=1.5, bounds=(-5, 5))
        m.y = pyo.Var(initialize=1.5, bounds=(-5, 5))
        m.obj = pyo.Objective(expr=m.x + m.y)
        m.c1 = pyo.Constraint(expr=m.x + m.y <= -20)

        opt.config.raise_exception_on_nonoptimal_result = False
        opt.config.load_solutions = False
        results = opt.solve(m)
        print(results.solution_status)
        print(results.termination_condition)
        self.assertIsNone(results.incumbent_objective)

    def test_none_when_unbounded(self):
        """Test that incumbent objective is None when problem is unbounded."""
        opt = KnitroDirectSolver()
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=1.5, bounds=(-5, None))
        m.y = pyo.Var(initialize=1.5, bounds=(-5, None))
        m.obj = pyo.Objective(expr=m.x + m.y, sense=pyo.maximize)
        m.c1 = pyo.Constraint(expr=m.x + m.y >= -10)

        opt.config.raise_exception_on_nonoptimal_result = False
        opt.config.load_solutions = False
        results = opt.solve(m)
        self.assertIsNone(results.incumbent_objective)

    def test_value_when_optimal(self):
        """Test that incumbent objective is correct for optimal solution."""
        opt = KnitroDirectSolver()
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=1.5, bounds=(-5, 5))
        m.y = pyo.Var(initialize=1.5, bounds=(-5, 5))
        m.obj = pyo.Objective(expr=m.x + m.y, sense=pyo.maximize)
        m.c1 = pyo.Constraint(expr=m.x + m.y <= 3)

        results = opt.solve(m)
        self.assertIsNotNone(results.incumbent_objective)
        self.assertAlmostEqual(results.incumbent_objective, 3.0)


@unittest.skipIf(not avail, "KNITRO solver is not available")
@unittest.pytest.mark.solver("knitro_direct")
class TestKnitroSolverSolutionStatus(unittest.TestCase):
    def test_solution_status_mapping(self):
        """Test that solution status is correctly mapped from KNITRO status."""
        engine = Engine()

        # Test that RuntimeError is raised for None status
        engine._status = None
        with self.assertRaises(RuntimeError):
            engine.get_solution_status()

        # Test optimal statuses
        for code in [
            knitro.KN_RC_OPTIMAL,
            knitro.KN_RC_OPTIMAL_OR_SATISFACTORY,
            knitro.KN_RC_NEAR_OPT,
        ]:
            engine._status = code
            self.assertEqual(engine.get_solution_status(), SolutionStatus.optimal)

        # Test feasible statuses
        for code in [
            knitro.KN_RC_FEAS_XTOL,
            knitro.KN_RC_FEAS_NO_IMPROVE,
            knitro.KN_RC_FEAS_FTOL,
            -103,  # KN_RC_FEAS_BEST
            -104,  # KN_RC_FEAS_MULTISTART
            knitro.KN_RC_ITER_LIMIT_FEAS,
            knitro.KN_RC_TIME_LIMIT_FEAS,
            knitro.KN_RC_FEVAL_LIMIT_FEAS,
            knitro.KN_RC_MIP_EXH_FEAS,
            knitro.KN_RC_MIP_TERM_FEAS,
            knitro.KN_RC_MIP_SOLVE_LIMIT_FEAS,
            knitro.KN_RC_MIP_NODE_LIMIT_FEAS,
        ]:
            engine._status = code
            self.assertEqual(engine.get_solution_status(), SolutionStatus.feasible)

        # Test infeasible statuses
        for code in [
            knitro.KN_RC_INFEASIBLE,
            knitro.KN_RC_INFEAS_CON_BOUNDS,
            knitro.KN_RC_INFEAS_VAR_BOUNDS,
            knitro.KN_RC_INFEAS_MULTISTART,
        ]:
            engine._status = code
            self.assertEqual(engine.get_solution_status(), SolutionStatus.infeasible)

        # Test noSolution statuses (anything else)
        for code in [-501, -600, -99999, -1]:
            engine._status = code
            self.assertEqual(engine.get_solution_status(), SolutionStatus.noSolution)


@unittest.skipIf(not avail, "KNITRO solver is not available")
@unittest.pytest.mark.solver("knitro_direct")
class TestKnitroSolverTerminationCondition(unittest.TestCase):
    def test_termination_condition_mapping(self):
        """Test that termination condition is correctly mapped from KNITRO status."""
        engine = Engine()

        # Test that RuntimeError is raised for None status
        engine._status = None
        with self.assertRaises(RuntimeError):
            engine.get_termination_condition()

        # Test convergenceCriteriaSatisfied
        for code in [
            knitro.KN_RC_OPTIMAL,
            knitro.KN_RC_OPTIMAL_OR_SATISFACTORY,
            knitro.KN_RC_NEAR_OPT,
        ]:
            engine._status = code
            self.assertEqual(
                engine.get_termination_condition(),
                TerminationCondition.convergenceCriteriaSatisfied,
            )

        # Test locallyInfeasible
        for code in [knitro.KN_RC_INFEAS_NO_IMPROVE, knitro.KN_RC_INFEAS_MULTISTART]:
            engine._status = code
            self.assertEqual(
                engine.get_termination_condition(),
                TerminationCondition.locallyInfeasible,
            )

        # Test provenInfeasible
        for code in [
            knitro.KN_RC_INFEASIBLE,
            knitro.KN_RC_INFEAS_CON_BOUNDS,
            knitro.KN_RC_INFEAS_VAR_BOUNDS,
        ]:
            engine._status = code
            self.assertEqual(
                engine.get_termination_condition(),
                TerminationCondition.provenInfeasible,
            )

        # Test infeasibleOrUnbounded
        for code in [knitro.KN_RC_UNBOUNDED, knitro.KN_RC_UNBOUNDED_OR_INFEAS]:
            engine._status = code
            self.assertEqual(
                engine.get_termination_condition(),
                TerminationCondition.infeasibleOrUnbounded,
            )

        # Test iterationLimit
        for code in [
            knitro.KN_RC_ITER_LIMIT_FEAS,
            knitro.KN_RC_FEVAL_LIMIT_FEAS,
            knitro.KN_RC_MIP_EXH_FEAS,
            knitro.KN_RC_MIP_TERM_FEAS,
            knitro.KN_RC_MIP_SOLVE_LIMIT_FEAS,
            knitro.KN_RC_MIP_NODE_LIMIT_FEAS,
            knitro.KN_RC_ITER_LIMIT_INFEAS,
            knitro.KN_RC_FEVAL_LIMIT_INFEAS,
            knitro.KN_RC_MIP_EXH_INFEAS,
            knitro.KN_RC_MIP_SOLVE_LIMIT_INFEAS,
            knitro.KN_RC_MIP_NODE_LIMIT_INFEAS,
        ]:
            engine._status = code
            self.assertEqual(
                engine.get_termination_condition(), TerminationCondition.iterationLimit
            )

        # Test interrupted
        engine._status = knitro.KN_RC_USER_TERMINATION
        self.assertEqual(
            engine.get_termination_condition(), TerminationCondition.interrupted
        )

        # Test error (-500 > status >= -600)
        for code in [-501, -550, -600]:
            engine._status = code
            self.assertEqual(
                engine.get_termination_condition(), TerminationCondition.error
            )

        # Test unknown (anything else outside the defined ranges)
        for code in [-601, -99999, -1]:
            engine._status = code
            self.assertEqual(
                engine.get_termination_condition(), TerminationCondition.unknown
            )


@unittest.skipIf(not avail, "KNITRO solver is not available")
@unittest.pytest.mark.solver("knitro_direct")
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


@unittest.skipIf(not avail, "KNITRO solver is not available")
@unittest.pytest.mark.solver("knitro_direct")
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
