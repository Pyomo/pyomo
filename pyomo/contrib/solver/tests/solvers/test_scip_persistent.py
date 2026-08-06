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

from pyomo.common.timing import HierarchicalTimer
from pyomo.contrib.solver.common.base import Availability
from pyomo.contrib.solver.common.results import SolutionStatus, TerminationCondition
from pyomo.contrib.solver.common.util import (
    NoFeasibleSolutionError,
    NoOptimalSolutionError,
)
from pyomo.contrib.solver.solvers.scip.scip_persistent import (
    ScipPersistent,
    ScipPersistentConfig,
)

scip_available = ScipPersistent().available()


@unittest.pytest.mark.solver("scip_persistent")
class TestScipPersistentConfig(unittest.TestCase):
    def test_default_instantiation(self):
        config = ScipPersistentConfig()
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
        self.assertTrue(hasattr(config, 'auto_updates'))

    def test_custom_instantiation(self):
        config = ScipPersistentConfig(description="A description")
        config.tee = True
        config.warmstart_discrete_vars = True
        self.assertTrue(config.tee)
        self.assertEqual(config._description, "A description")
        self.assertTrue(config.warmstart_discrete_vars)


@unittest.pytest.mark.solver("scip_persistent")
class TestScipPersistentInterface(unittest.TestCase):
    def test_default_instantiation(self):
        opt = ScipPersistent()
        self.assertTrue(opt.is_persistent())
        self.assertEqual(opt.name, 'scip_persistent')
        self.assertEqual(opt.CONFIG, opt.config)
        self.assertIn(
            opt.available(),
            {Availability.NotFound, Availability.BadVersion, Availability.FullLicense},
        )

    def test_context_manager(self):
        with ScipPersistent() as opt:
            self.assertTrue(opt.is_persistent())
            self.assertEqual(opt.name, 'scip_persistent')
            self.assertEqual(opt.CONFIG, opt.config)

    def test_update_before_set_instance_raises(self):
        opt = ScipPersistent()
        with self.assertRaisesRegex(
            RuntimeError, 'must call set_instance or solve before update'
        ):
            opt.update()

    def test_add_constraints_before_set_instance_raises(self):
        opt = ScipPersistent()
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.c = pyo.Constraint(expr=m.x >= 1)
        with self.assertRaisesRegex(RuntimeError, 'call set_instance first'):
            opt.add_constraints([m.c])


@unittest.skipIf(not scip_available, "SCIP is not available")
@unittest.pytest.mark.solver("scip_persistent")
class TestScipPersistent(unittest.TestCase):
    def create_lp_model(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, None), initialize=0)
        m.y = pyo.Var(bounds=(0, None), initialize=0)
        m.obj = pyo.Objective(expr=m.x + 2 * m.y)
        m.c = pyo.Constraint(expr=m.x + m.y >= 1)
        return m

    def create_range_model(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.xl = pyo.Param(initialize=-1, mutable=True)
        m.xu = pyo.Param(initialize=1, mutable=True)
        m.c = pyo.Constraint(expr=pyo.inequality(m.xl, m.x, m.xu))
        m.obj = pyo.Objective(expr=m.x)
        return m

    def test_set_instance_and_solve(self):
        m = self.create_lp_model()
        opt = ScipPersistent()
        opt.set_instance(m)
        res = opt.solve(m)

        self.assertEqual(
            res.termination_condition, TerminationCondition.convergenceCriteriaSatisfied
        )
        self.assertEqual(res.solution_status, SolutionStatus.optimal)
        self.assertAlmostEqual(m.x.value, 1)
        self.assertAlmostEqual(m.y.value, 0)

    def test_solve_twice_same_instance(self):
        m = self.create_lp_model()
        opt = ScipPersistent()

        res = opt.solve(m)
        self.assertEqual(res.solution_status, SolutionStatus.optimal)
        self.assertAlmostEqual(m.x.value, 1)
        self.assertAlmostEqual(m.y.value, 0)

        m.c.set_value(m.x + m.y >= 2)
        res = opt.solve(m)
        self.assertEqual(res.solution_status, SolutionStatus.optimal)
        self.assertAlmostEqual(m.x.value, 2)
        self.assertAlmostEqual(m.y.value, 0)

    def test_load_solutions_false(self):
        m = self.create_lp_model()
        opt = ScipPersistent()
        res = opt.solve(m, load_solutions=False)

        self.assertEqual(res.solution_status, SolutionStatus.optimal)
        self.assertAlmostEqual(m.x.value, 0)
        self.assertAlmostEqual(m.y.value, 0)

        vals = res.solution_loader.get_vars()
        self.assertAlmostEqual(vals[m.x], 1)
        self.assertAlmostEqual(vals[m.y], 0)

        res.solution_loader.load_vars()
        self.assertAlmostEqual(m.x.value, 1)
        self.assertAlmostEqual(m.y.value, 0)

    def test_solution_loader_invalidated_after_update(self):
        m = self.create_lp_model()
        opt = ScipPersistent()
        res = opt.solve(m, load_solutions=False)

        m.c.set_value(m.x + m.y >= 2)
        opt.update()

        with self.assertRaisesRegex(
            RuntimeError, 'The results in the solver are no longer valid.'
        ):
            res.solution_loader.get_vars()

    def test_range_constraint_mutable_params(self):
        m = self.create_range_model()
        opt = ScipPersistent()
        opt.set_instance(m)

        res = opt.solve(m)
        self.assertEqual(res.solution_status, SolutionStatus.optimal)
        self.assertAlmostEqual(m.x.value, -1)

        m.xl.value = -3
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, -3)

        del m.obj
        m.obj = pyo.Objective(expr=m.x, sense=pyo.maximize)
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, 1)

        m.xu.value = 3
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, 3)

    def test_add_remove_constraints(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(-10, 10))
        m.y = pyo.Var()
        m.obj = pyo.Objective(expr=m.y)
        m.c1 = pyo.Constraint(expr=m.y >= 2 * m.x + 1)

        opt = ScipPersistent()
        opt.set_instance(m)

        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, -10)
        self.assertAlmostEqual(m.y.value, -19)

        m.c2 = pyo.Constraint(expr=m.y >= -m.x + 1)
        opt.add_constraints([m.c2])

        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, 0)
        self.assertAlmostEqual(m.y.value, 1)

        opt.remove_constraints([m.c2])
        m.del_component(m.c2)

        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, -10)
        self.assertAlmostEqual(m.y.value, -19)

    def test_update_variables_manual(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(-1, 1))
        m.obj = pyo.Objective(expr=m.x)

        opt = ScipPersistent()
        opt.config.auto_updates.update_vars = False
        opt.set_instance(m)

        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, -1)

        m.x.setlb(-3)
        opt.update_variables([m.x])

        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, -3)

    def test_update_parameters_manual(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(-10, 10))
        m.p = pyo.Param(initialize=1, mutable=True)
        m.obj = pyo.Objective(expr=m.x)
        m.c = pyo.Constraint(expr=m.x >= m.p)

        opt = ScipPersistent()
        opt.config.auto_updates.update_parameters = False
        opt.set_instance(m)

        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, 1)

        m.p.value = 3
        opt.update_parameters([m.p])

        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, 3)

    def test_timer(self):
        m = self.create_lp_model()
        timer = HierarchicalTimer()
        opt = ScipPersistent()
        res = opt.solve(m, timer=timer)
        self.assertIs(res.timing_info.timer, timer)

    def test_infeasible_no_exception(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 1))
        m.obj = pyo.Objective(expr=m.x)
        m.c = pyo.Constraint(expr=m.x >= 2)

        opt = ScipPersistent()
        res = opt.solve(
            m, load_solutions=False, raise_exception_on_nonoptimal_result=False
        )

        self.assertEqual(
            res.termination_condition, TerminationCondition.provenInfeasible
        )
        self.assertEqual(res.solution_status, SolutionStatus.noSolution)

    def test_infeasible_raises(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 1))
        m.obj = pyo.Objective(expr=m.x)
        m.c = pyo.Constraint(expr=m.x >= 2)

        opt = ScipPersistent()
        with self.assertRaises(NoOptimalSolutionError):
            opt.solve(m, load_solutions=False)

        with self.assertRaises(NoFeasibleSolutionError):
            opt.solve(
                m, load_solutions=True, raise_exception_on_nonoptimal_result=False
            )
