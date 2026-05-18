# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import datetime

import pyomo.common.unittest as unittest
import pyomo.environ as pyo

from pyomo.common.config import ConfigDict
from pyomo.common.timing import HierarchicalTimer
from pyomo.contrib.solver.common.base import Availability
from pyomo.contrib.solver.common.results import SolutionStatus, TerminationCondition
from pyomo.contrib.solver.common.util import (
    NoFeasibleSolutionError,
    NoOptimalSolutionError,
    NoSolutionError,
)
from pyomo.contrib.solver.solvers.scip.base import ScipConfig
from pyomo.contrib.solver.solvers.scip.scip_direct import ScipDirect
from pyomo.contrib.solver.tests.solvers.test_gurobi_persistent import (
    create_pmedian_model,
)

scip_available = ScipDirect().available()


@unittest.pytest.mark.solver("scip_direct")
class TestScipDirectConfig(unittest.TestCase):
    def test_default_instantiation(self):
        config = ScipConfig()
        self.assertIsNone(config._description)
        self.assertEqual(config._visibility, 0)
        self.assertFalse(config.tee)
        self.assertTrue(config.load_solutions)
        self.assertTrue(config.raise_exception_on_nonoptimal_result)
        self.assertFalse(config.symbolic_solver_labels)
        self.assertIsNone(config.timer)
        self.assertIsNone(config.threads)
        self.assertIsNone(config.time_limit)
        self.assertIsNone(config.rel_gap)
        self.assertIsNone(config.abs_gap)
        self.assertFalse(config.warmstart_discrete_vars)
        self.assertIsInstance(config.solver_options, ConfigDict)

    def test_custom_instantiation(self):
        config = ScipConfig(description="A description")
        config.tee = True
        config.warmstart_discrete_vars = True
        self.assertTrue(config.tee)
        self.assertEqual(config._description, "A description")
        self.assertTrue(config.warmstart_discrete_vars)


@unittest.pytest.mark.solver("scip_direct")
class TestScipDirectInterface(unittest.TestCase):
    def test_class_member_list(self):
        opt = ScipDirect()
        expected_list = [
            'CONFIG',
            'available',
            'config',
            'api_version',
            'is_persistent',
            'name',
            'solve',
            'version',
        ]
        method_list = [method for method in dir(opt) if not method.startswith('_')]
        self.assertEqual(sorted(expected_list), sorted(method_list))

    def test_default_instantiation(self):
        opt = ScipDirect()
        self.assertFalse(opt.is_persistent())
        self.assertEqual(opt.name, 'scip_direct')
        self.assertEqual(opt.CONFIG, opt.config)
        self.assertIn(
            opt.available(),
            {Availability.NotFound, Availability.BadVersion, Availability.FullLicense},
        )

    def test_context_manager(self):
        with ScipDirect() as opt:
            self.assertFalse(opt.is_persistent())
            self.assertEqual(opt.name, 'scip_direct')
            self.assertEqual(opt.CONFIG, opt.config)

    def test_version(self):
        opt = ScipDirect()
        if opt.available() == Availability.FullLicense:
            ver = opt.version()
            self.assertIsInstance(ver, tuple)
            self.assertGreaterEqual(len(ver), 3)
            self.assertTrue(all(isinstance(_, int) for _ in ver))

    def test_get_tc_map(self):
        opt = ScipDirect()
        tc_map = opt._get_tc_map()
        self.assertEqual(
            tc_map["optimal"], TerminationCondition.convergenceCriteriaSatisfied
        )
        self.assertEqual(tc_map["timelimit"], TerminationCondition.maxTimeLimit)
        self.assertEqual(tc_map["infeasible"], TerminationCondition.provenInfeasible)
        self.assertEqual(tc_map["unbounded"], TerminationCondition.unbounded)
        self.assertEqual(
            tc_map["inforunbd"], TerminationCondition.infeasibleOrUnbounded
        )

    def test_scip_vtype_from_var(self):
        m = pyo.ConcreteModel()
        m.b = pyo.Var(within=pyo.Binary)
        m.i = pyo.Var(within=pyo.Integers)
        m.c = pyo.Var(within=pyo.Reals)

        opt = ScipDirect()
        self.assertEqual(opt._scip_vtype_from_var(m.b), "B")
        self.assertEqual(opt._scip_vtype_from_var(m.i), "I")
        self.assertEqual(opt._scip_vtype_from_var(m.c), "C")


@unittest.skipIf(not scip_available, "SCIP is not available")
@unittest.pytest.mark.solver("scip_direct")
class TestScipDirect(unittest.TestCase):
    def create_lp_model(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, None), initialize=0)
        m.y = pyo.Var(bounds=(0, None), initialize=0)
        m.obj = pyo.Objective(expr=m.x + 2 * m.y)
        m.c = pyo.Constraint(expr=m.x + m.y >= 1)
        return m

    def create_feasible_model_no_objective(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, None), initialize=0)
        m.c = pyo.Constraint(expr=m.x >= 1)
        return m

    def create_infeasible_model(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 1), initialize=0)
        m.obj = pyo.Objective(expr=m.x)
        m.c = pyo.Constraint(expr=m.x >= 2)
        return m

    def create_sos_model(self, level):
        m = pyo.ConcreteModel()
        m.I = pyo.RangeSet(3)
        m.x = pyo.Var(m.I, bounds=(0, 1))
        m.obj = pyo.Objective(expr=sum(i * m.x[i] for i in m.I))
        m.c = pyo.Constraint(expr=sum(m.x[i] for i in m.I) == 1)
        m.sos = pyo.SOSConstraint(var=m.x, sos=level)
        return m

    def test_solve(self):
        m = self.create_lp_model()
        opt = ScipDirect()
        res = opt.solve(m)

        self.assertEqual(
            res.termination_condition, TerminationCondition.convergenceCriteriaSatisfied
        )
        self.assertEqual(res.solution_status, SolutionStatus.optimal)
        self.assertAlmostEqual(m.x.value, 1)
        self.assertAlmostEqual(m.y.value, 0)
        self.assertAlmostEqual(res.incumbent_objective, 1)
        self.assertIsNotNone(res.objective_bound)
        self.assertEqual(res.solver_name, 'scip_direct')
        self.assertIsInstance(res.solver_version, tuple)
        self.assertIsNotNone(res.solver_log)
        self.assertIsNotNone(res.timing_info.scip_time)
        self.assertEqual(res.timing_info.start_timestamp.tzinfo, datetime.timezone.utc)
        self.assertGreaterEqual(res.timing_info.wall_time, 0)
        self.assertIn('NNodes', res.extra_info)

    def test_solve_load_solutions_false(self):
        m = self.create_lp_model()
        opt = ScipDirect()
        res = opt.solve(m, load_solutions=False)

        self.assertEqual(
            res.termination_condition, TerminationCondition.convergenceCriteriaSatisfied
        )
        self.assertEqual(res.solution_status, SolutionStatus.optimal)

        self.assertAlmostEqual(m.x.value, 0)
        self.assertAlmostEqual(m.y.value, 0)

        self.assertEqual(res.solution_loader.get_number_of_solutions(), 1)
        self.assertEqual(res.solution_loader.get_solution_ids(), [0])

        vals = res.solution_loader.get_vars()
        self.assertAlmostEqual(vals[m.x], 1)
        self.assertAlmostEqual(vals[m.y], 0)

        res.solution_loader.load_vars()
        self.assertAlmostEqual(m.x.value, 1)
        self.assertAlmostEqual(m.y.value, 0)

    def test_no_objective(self):
        m = self.create_feasible_model_no_objective()
        opt = ScipDirect()
        res = opt.solve(m)

        self.assertEqual(
            res.termination_condition, TerminationCondition.convergenceCriteriaSatisfied
        )
        self.assertEqual(res.solution_status, SolutionStatus.optimal)
        self.assertIsNone(res.incumbent_objective)
        self.assertIsNone(res.objective_bound)

    def test_infeasible_no_exception(self):
        m = self.create_infeasible_model()
        opt = ScipDirect()
        res = opt.solve(
            m, load_solutions=False, raise_exception_on_nonoptimal_result=False
        )

        self.assertEqual(
            res.termination_condition, TerminationCondition.provenInfeasible
        )
        self.assertEqual(res.solution_status, SolutionStatus.noSolution)
        self.assertIsNone(res.incumbent_objective)
        self.assertEqual(res.solution_loader.get_number_of_solutions(), 0)
        with self.assertRaises(NoSolutionError):
            res.solution_loader.get_vars()

    def test_infeasible_raises_no_optimal_solution_error(self):
        m = self.create_infeasible_model()
        opt = ScipDirect()
        with self.assertRaises(NoOptimalSolutionError):
            opt.solve(m, load_solutions=False)

    def test_infeasible_raises_no_feasible_solution_error(self):
        m = self.create_infeasible_model()
        opt = ScipDirect()
        with self.assertRaises(NoFeasibleSolutionError):
            opt.solve(
                m, load_solutions=True, raise_exception_on_nonoptimal_result=False
            )

    def test_timer(self):
        m = self.create_lp_model()
        timer = HierarchicalTimer()
        opt = ScipDirect()
        res = opt.solve(m, timer=timer)
        self.assertIs(res.timing_info.timer, timer)

    def test_fixed_var(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(-10, 10), initialize=0)
        m.y = pyo.Var(bounds=(-10, 10), initialize=0)
        m.x.fix(2)
        m.obj = pyo.Objective(expr=m.y)
        m.c = pyo.Constraint(expr=m.y >= m.x + 1)

        opt = ScipDirect()
        res = opt.solve(m)

        self.assertEqual(res.solution_status, SolutionStatus.optimal)
        self.assertAlmostEqual(m.x.value, 2)
        self.assertAlmostEqual(m.y.value, 3)

    def test_sos1(self):
        m = self.create_sos_model(level=1)
        opt = ScipDirect()
        res = opt.solve(m)

        self.assertEqual(
            res.termination_condition, TerminationCondition.convergenceCriteriaSatisfied
        )
        self.assertEqual(res.solution_status, SolutionStatus.optimal)

    def test_sos2(self):
        m = self.create_sos_model(level=2)
        opt = ScipDirect()
        res = opt.solve(m)

        self.assertEqual(
            res.termination_condition, TerminationCondition.convergenceCriteriaSatisfied
        )
        self.assertEqual(res.solution_status, SolutionStatus.optimal)

    def test_bad_sos_level(self):
        m = self.create_sos_model(level=1)
        m.del_component(m.sos)
        m.sos = pyo.SOSConstraint(var=m.x, sos=3)

        opt = ScipDirect()
        with self.assertRaisesRegex(ValueError, "does not support SOS level 3"):
            opt.solve(m)

    def test_warmstart_discrete_vars(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(within=pyo.Binary, initialize=1)
        m.y = pyo.Var(within=pyo.Binary, initialize=0)
        m.obj = pyo.Objective(expr=m.x + 2 * m.y, sense=pyo.maximize)
        m.c = pyo.Constraint(expr=m.x + m.y <= 1)

        opt = ScipDirect()
        res = opt.solve(m, warmstart_discrete_vars=True)

        self.assertEqual(
            res.termination_condition, TerminationCondition.convergenceCriteriaSatisfied
        )
        self.assertEqual(res.solution_status, SolutionStatus.optimal)
        self.assertAlmostEqual(m.x.value, 0)
        self.assertAlmostEqual(m.y.value, 1)

    def test_multiple_solutions(self):
        m = create_pmedian_model()

        # The solutions found by scip may change from version to version.
        # Let's warmstart scip with a suboptimal solution to ensure we
        # have at least 2 solutions.

        init_sol = {1, 2, 3}
        for k, y in m.y.items():
            if k in init_sol:
                y.value = 1
            else:
                y.value = 0

        opt = ScipDirect()
        opt.config.warmstart_discrete_vars = True
        opt.config.solver_options['limits/maxsol'] = 100000
        opt.config.solver_options['heuristics/completesol/maxunknownrate'] = 1.0
        res = opt.solve(m, load_solutions=True)
        num_solutions = res.solution_loader.get_number_of_solutions()
        self.assertGreaterEqual(num_solutions, 2)

        # the best solution
        self.assertAlmostEqual(pyo.value(m.obj.expr), 6.431184939357673)
        sol = {3, 6, 9}
        for k, v in m.y.items():
            if k in sol:
                self.assertAlmostEqual(v.value, 1)
            else:
                self.assertAlmostEqual(v.value, 0)

        # the worst solution that we used to warmstart
        res.solution_loader.solution(num_solutions - 1).load_vars()
        self.assertAlmostEqual(pyo.value(m.obj.expr), 7.607295680844689)
        sol = {1, 2, 3}
        for k, v in m.y.items():
            if k in sol:
                self.assertAlmostEqual(v.value, 1)
            else:
                self.assertAlmostEqual(v.value, 0)
