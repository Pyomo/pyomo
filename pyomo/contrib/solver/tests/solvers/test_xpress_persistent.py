# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import math
import os
import tempfile

import pyomo.common.unittest as unittest
import pyomo.environ as pyo

from pyomo.contrib.solver.common.results import TerminationCondition
from pyomo.contrib.solver.common.util import IncompatibleModelError, NoSolutionError
from pyomo.contrib.solver.common.results import SolutionStatus
from pyomo.contrib.solver.solvers.xpress import XpressPersistent
from pyomo.contrib.solver.tests.solvers._xpress_test_utils import (
    _simple_lp,
    _simple_mip,
    _solve_and_check,
    _solve_check_mutate_check,
    _trivial_model,
)

if not XpressPersistent().available():
    raise unittest.SkipTest('Xpress not available')


@unittest.pytest.mark.solver('xpress_persistent')
class TestXpressPersistentObjective(unittest.TestCase):
    def setUp(self):
        self.opt = XpressPersistent()

    def test_remove_objective_between_solves(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(1, 5))
        m.c = pyo.Constraint(expr=m.x >= 2)
        m.obj = pyo.Objective(expr=m.x)

        _solve_and_check(self, self.opt, m, {'objective': 2.0, 'vars': [(m.x, 2.0)]})

        del m.obj
        res = self.opt.solve(m)
        self.assertEqual(res.solution_status, SolutionStatus.optimal)
        self.assertIsNone(res.incumbent_objective)

    def test_active_objective_toggle(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 5))
        m.obj_min = pyo.Objective(expr=m.x, sense=pyo.minimize)
        m.obj_max = pyo.Objective(expr=m.x, sense=pyo.maximize)
        m.obj_max.deactivate()

        _solve_and_check(self, self.opt, m, {'objective': 0.0, 'vars': [(m.x, 0.0)]})

        m.obj_min.deactivate()
        m.obj_max.activate()
        _solve_and_check(self, self.opt, m, {'objective': 5.0, 'vars': [(m.x, 5.0)]})

    def test_two_active_objectives_at_set_instance_raises(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 5))
        m.obj1 = pyo.Objective(expr=m.x, sense=pyo.minimize)
        m.obj2 = pyo.Objective(expr=-m.x, sense=pyo.minimize)
        with self.assertRaises(IncompatibleModelError):
            self.opt.solve(m)

    def test_two_active_objectives_at_update_raises(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 5))
        m.obj1 = pyo.Objective(expr=m.x, sense=pyo.minimize)
        self.opt.solve(m)

        m.obj2 = pyo.Objective(expr=-m.x, sense=pyo.minimize)
        with self.assertRaises(IncompatibleModelError):
            self.opt.solve(m)

    def test_recovery_after_two_objectives_raises(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 5))
        m.obj1 = pyo.Objective(expr=m.x, sense=pyo.minimize)
        self.opt.solve(m)

        m.obj2 = pyo.Objective(expr=-m.x, sense=pyo.minimize)
        with self.assertRaises(IncompatibleModelError):
            self.opt.solve(m)

        m.obj2.deactivate()
        self.opt.set_instance(m)
        _solve_and_check(self, self.opt, m, {'objective': 0.0, 'vars': [(m.x, 0.0)]})


@unittest.pytest.mark.solver('xpress_persistent')
class TestXpressPersistentLifecycle(unittest.TestCase):
    def setUp(self):
        self.opt = XpressPersistent()

    def test_eager_invalidation_on_mutation(self):
        m = _simple_lp()
        res = _solve_and_check(
            self, self.opt, m, {'objective': -8.0, 'vars': [(m.x, 0.0), (m.y, 4.0)]}
        )
        res.solution_loader.get_vars()
        m.c3 = pyo.Constraint(expr=m.x + m.y >= 1)
        self.opt.add_constraints([m.c3])
        with self.assertRaises(NoSolutionError):
            res.solution_loader.get_vars()

    def test_eager_invalidation_on_param_change(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 10))
        m.p = pyo.Param(mutable=True, initialize=5.0)
        m.c = pyo.Constraint(expr=m.x <= m.p)
        m.obj = pyo.Objective(expr=-m.x)
        res = _solve_and_check(
            self, self.opt, m, {'objective': -5.0, 'vars': [(m.x, 5.0)]}
        )
        res.solution_loader.get_vars()
        m.p.value = 7.0
        self.opt.update_parameters([m.p])
        with self.assertRaises(NoSolutionError):
            res.solution_loader.get_vars()

    def test_symbolic_solver_labels_persistent(self):
        m = pyo.ConcreteModel()
        m.distinctive_var = pyo.Var(domain=pyo.NonNegativeReals)
        m.distinctive_con = pyo.Constraint(expr=m.distinctive_var <= 5)
        m.obj = pyo.Objective(expr=m.distinctive_var)

        _solve_and_check(
            self,
            self.opt,
            m,
            {'objective': 0.0, 'vars': [(m.distinctive_var, 0.0)]},
            symbolic_solver_labels=True,
        )
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, 'm')
            self.opt.write(base, flags='l')
            with open(base + '.lp', 'r') as f:
                content = f.read()
        self.assertIn('distinctive_var', content)
        self.assertIn('distinctive_con', content)

    def test_auto_updates_disable_parameter_tracking(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 10))
        m.p = pyo.Param(mutable=True, initialize=5.0)
        m.c = pyo.Constraint(expr=m.x <= m.p)
        m.obj = pyo.Objective(expr=-m.x)

        _solve_and_check(self, self.opt, m, {'objective': -5.0, 'vars': [(m.x, 5.0)]})

        m.p.value = 7.0
        _solve_and_check(
            self,
            self.opt,
            m,
            {'objective': -5.0, 'vars': [(m.x, 5.0)]},
            auto_updates={'update_parameters': False},
        )

        _solve_and_check(self, self.opt, m, {'objective': -7.0, 'vars': [(m.x, 7.0)]})

    def test_write_mps_and_lp(self):
        m = _simple_lp()
        _solve_and_check(
            self, self.opt, m, {'objective': -8.0, 'vars': [(m.x, 0.0), (m.y, 4.0)]}
        )
        with tempfile.TemporaryDirectory() as tmp:
            mps_base = os.path.join(tmp, 'mps_model')
            self.opt.write(mps_base)
            self.assertTrue(os.path.exists(mps_base + '.mps'))
            self.assertGreater(os.path.getsize(mps_base + '.mps'), 0)

            lp_base = os.path.join(tmp, 'lp_model')
            self.opt.write(lp_base, flags='l')
            self.assertTrue(os.path.exists(lp_base + '.lp'))
            self.assertGreater(os.path.getsize(lp_base + '.lp'), 0)

    def test_warmstart_disabled(self):
        m = _simple_mip()
        m.x.set_value(100)
        m.y.set_value(100)
        _solve_and_check(
            self,
            self.opt,
            m,
            {'objective': -8.0, 'vars': [(m.x, 0.0), (m.y, 4.0)]},
            warmstart=False,
        )


@unittest.pytest.mark.solver('xpress_persistent')
class TestXpressPersistentSOS(unittest.TestCase):
    def setUp(self):
        self.opt = XpressPersistent()

    def test_sos1_initial_and_remove(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3], domain=pyo.NonNegativeReals, bounds=(0, 1))
        m.sos1 = pyo.SOSConstraint(var=m.x, sos=1, weights={1: 1.0, 2: 2.0, 3: 3.0})
        m.obj = pyo.Objective(expr=m.x[1] + 2 * m.x[2] + 3 * m.x[3], sense=pyo.maximize)

        _solve_and_check(
            self,
            self.opt,
            m,
            {'objective': 3.0, 'vars': [(m.x[3], 1.0), (m.x[1], 0.0), (m.x[2], 0.0)]},
        )

        del m.sos1
        _solve_and_check(
            self,
            self.opt,
            m,
            {'objective': 6.0, 'vars': [(m.x[1], 1.0), (m.x[2], 1.0), (m.x[3], 1.0)]},
        )

    # Public persistent API (explicit call paths)

    def test_add_variables_public_api(self):
        m = _trivial_model()
        self.opt.set_instance(m)
        ncols_before = self.opt._xp_prob.attributes.cols
        m.y = pyo.Var(bounds=(0, 1))
        self.opt.add_variables([m.y])
        self.assertGreater(self.opt._xp_prob.attributes.cols, ncols_before)
        self.assertIn(id(m.y), self.opt._maps.vars)

    def test_remove_variables_public_api(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 5))
        m.y = pyo.Var(bounds=(0, 5))
        m.obj = pyo.Objective(expr=m.x + m.y)
        self.opt.set_instance(m)
        ncols_before = self.opt._xp_prob.attributes.cols
        self.opt.remove_variables([m.y])
        self.assertLess(self.opt._xp_prob.attributes.cols, ncols_before)
        self.assertNotIn(id(m.y), self.opt._maps.vars)

    def test_update_variables_public_api(self):
        m = _trivial_model()
        m.c = pyo.Constraint(expr=m.x >= 0.5)
        self.opt.set_instance(m)
        m.x.setub(3.0)
        self.opt.update_variables([m.x])
        _solve_and_check(self, self.opt, m, {'objective': 0.5, 'vars': [(m.x, 0.5)]})

    def test_set_objective_public_api(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(1, 5))
        m.obj = pyo.Objective(expr=m.x)
        self.opt.set_instance(m)
        _solve_and_check(self, self.opt, m, {'objective': 1.0, 'vars': [(m.x, 1.0)]})
        m.obj.deactivate()
        m.obj2 = pyo.Objective(expr=-m.x)
        self.opt.set_objective(m.obj2)
        _solve_and_check(self, self.opt, m, {'objective': -5.0, 'vars': [(m.x, 5.0)]})

    def test_add_remove_sos_constraints_public_api(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3], domain=pyo.NonNegativeReals, bounds=(0, 1))
        m.obj = pyo.Objective(expr=m.x[1] + 2 * m.x[2] + 3 * m.x[3], sense=pyo.maximize)
        self.opt.set_instance(m)
        _solve_and_check(
            self,
            self.opt,
            m,
            {'objective': 6.0, 'vars': [(m.x[1], 1.0), (m.x[2], 1.0), (m.x[3], 1.0)]},
        )
        m.sos1 = pyo.SOSConstraint(var=m.x, sos=1, weights={1: 1.0, 2: 2.0, 3: 3.0})
        self.opt.add_sos_constraints(list(m.sos1.values()))
        _solve_and_check(
            self,
            self.opt,
            m,
            {'objective': 3.0, 'vars': [(m.x[3], 1.0), (m.x[1], 0.0), (m.x[2], 0.0)]},
        )
        self.opt.remove_sos_constraints(list(m.sos1.values()))
        m.sos1.deactivate()
        _solve_and_check(
            self,
            self.opt,
            m,
            {'objective': 6.0, 'vars': [(m.x[1], 1.0), (m.x[2], 1.0), (m.x[3], 1.0)]},
        )

    def test_add_remove_block_public_api(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 10))
        m.obj = pyo.Objective(expr=m.x)
        self.opt.set_instance(m)
        _solve_and_check(self, self.opt, m, {'objective': 0.0, 'vars': [(m.x, 0.0)]})
        m.b = pyo.Block()
        m.b.c = pyo.Constraint(expr=m.x >= 5)
        self.opt.add_block(m.b)
        _solve_and_check(self, self.opt, m, {'objective': 5.0, 'vars': [(m.x, 5.0)]})
        self.opt.remove_block(m.b)
        m.b.deactivate()
        _solve_and_check(self, self.opt, m, {'objective': 0.0, 'vars': [(m.x, 0.0)]})

    def test_xpress_control_and_attribute(self):
        m = _trivial_model()
        self.opt.set_instance(m)
        self.opt.set_xpress_control('threads', 1)
        self.assertEqual(self.opt.get_xpress_control('threads'), 1)
        rows = self.opt.get_xpress_attribute('rows')
        self.assertGreaterEqual(rows, 0)

    def test_get_xpress_problem_returns_problem(self):
        m = _trivial_model()
        m.c = pyo.Constraint(expr=m.x >= 0.5)
        self.opt.set_instance(m)
        prob = self.opt.get_xpress_problem()
        self.assertIsNotNone(prob)
        xp_con = self.opt.get_xpress_constraint(m.c)
        _solve_and_check(self, self.opt, m, {'objective': 0.5, 'vars': [(m.x, 0.5)]})
        slack = prob.getSlacks(xp_con)
        self.assertAlmostEqual(slack, 0.0, places=6)

    def test_update_before_set_instance_raises(self):
        with self.assertRaises(RuntimeError):
            XpressPersistent().update()

    def test_get_xpress_var_returns_handle(self):
        m = _trivial_model()
        self.opt.set_instance(m)
        handle = self.opt.get_xpress_var(m.x)
        self.assertIsNotNone(handle)
        self.assertGreaterEqual(handle.index, 0)

    def test_get_xpress_constraint_returns_handle(self):
        m = _trivial_model()
        m.c = pyo.Constraint(expr=m.x >= 0.5)
        self.opt.set_instance(m)
        handle = self.opt.get_xpress_constraint(m.c)
        self.assertIsNotNone(handle)
        self.assertGreaterEqual(handle.index, 0)

    def test_get_xpress_sos_returns_handle(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2], domain=pyo.NonNegativeReals, bounds=(0, 1))
        m.sos = pyo.SOSConstraint(var=m.x, sos=1, weights={1: 1.0, 2: 2.0})
        m.obj = pyo.Objective(expr=m.x[1] + m.x[2])
        self.opt.set_instance(m)
        handle = self.opt.get_xpress_sos(list(m.sos.values())[0])
        self.assertIsNotNone(handle)

    def test_release_clears_state(self):
        m = _trivial_model()
        self.opt.set_instance(m)
        self.assertIsNotNone(self.opt._xp_prob)
        self.opt.release()
        self.assertIsNone(self.opt._xp_prob)
        self.assertIsNone(self.opt._maps)
        self.assertIsNone(self.opt._change_detector)
        self.assertIsNone(self.opt._pyomo_model)
        self.assertIsNone(self.opt._vars)
        self.assertEqual(self.opt._mutable_helpers, {})

    def test_reset_clears_state(self):
        m = _trivial_model()
        self.opt.set_instance(m)
        self.assertIsNotNone(self.opt._xp_prob)
        self.opt.reset()
        self.assertIsNone(self.opt._xp_prob)
        self.assertIsNone(self.opt._maps)
        self.assertIsNone(self.opt._change_detector)
        self.assertIsNone(self.opt._pyomo_model)
        self.assertIsNone(self.opt._vars)
        self.assertEqual(self.opt._mutable_helpers, {})


@unittest.pytest.mark.solver('xpress_persistent')
class TestXpressPersistentQuadratic(unittest.TestCase):
    def setUp(self):
        self.opt = XpressPersistent()

    def test_qp_objective_persistent(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(domain=pyo.NonNegativeReals)
        m.y = pyo.Var(domain=pyo.NonNegativeReals)
        m.c = pyo.Constraint(expr=m.x + m.y >= 1)
        m.obj = pyo.Objective(expr=m.x**2 + m.y**2)
        _solve_and_check(
            self, self.opt, m, {'objective': 0.5, 'vars': [(m.x, 0.5), (m.y, 0.5)]}
        )
        _solve_and_check(
            self, self.opt, m, {'objective': 0.5, 'vars': [(m.x, 0.5), (m.y, 0.5)]}
        )

    def test_qcp_add_remove_persistent(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 2))
        m.y = pyo.Var(bounds=(0, 2))
        m.obj = pyo.Objective(expr=-(m.x + m.y))
        self.opt.set_instance(m)
        _solve_and_check(
            self, self.opt, m, {'objective': -4.0, 'vars': [(m.x, 2.0), (m.y, 2.0)]}
        )

        m.qc = pyo.Constraint(expr=m.x**2 + m.y**2 <= 1)
        self.opt.add_constraints([m.qc])
        _solve_and_check(
            self,
            self.opt,
            m,
            {
                'objective': -math.sqrt(2),
                'vars': [(m.x, math.sqrt(2) / 2), (m.y, math.sqrt(2) / 2)],
                'obj_places': 5,
                'var_places': 5,
            },
        )

        self.opt.remove_constraints([m.qc])
        m.qc.deactivate()
        _solve_and_check(
            self, self.opt, m, {'objective': -4.0, 'vars': [(m.x, 2.0), (m.y, 2.0)]}
        )

    def test_mutable_param_in_quadratic_obj(self):
        m = pyo.ConcreteModel()
        m.p = pyo.Param(mutable=True, initialize=1.0)
        m.x = pyo.Var(bounds=(1, None))
        m.obj = pyo.Objective(expr=m.p * m.x**2)
        _solve_check_mutate_check(
            self,
            self.opt,
            m,
            {'objective': 1.0, 'vars': [(m.x, 1.0)]},
            m.p,
            4.0,
            {'objective': 4.0, 'vars': [(m.x, 1.0)], 'obj_places': 4, 'var_places': 4},
        )

    def test_mutable_param_in_quadratic_constraint(self):
        m = pyo.ConcreteModel()
        m.p = pyo.Param(mutable=True, initialize=1.0)
        m.x = pyo.Var(bounds=(0, None))
        m.y = pyo.Var(bounds=(0, None))
        m.qc = pyo.Constraint(expr=m.x**2 + m.p * m.y**2 <= 1)
        m.obj = pyo.Objective(expr=-(m.x + m.y))
        _solve_and_check(
            self,
            self.opt,
            m,
            {
                'objective': -math.sqrt(2),
                'vars': [(m.x, math.sqrt(2) / 2), (m.y, math.sqrt(2) / 2)],
                'obj_places': 5,
                'var_places': 5,
            },
        )
        m.p.set_value(4.0)
        _solve_and_check(
            self,
            self.opt,
            m,
            {
                'objective': -(2 / math.sqrt(5) + 1 / (2 * math.sqrt(5))),
                'vars': [(m.x, 2 / math.sqrt(5)), (m.y, 1 / (2 * math.sqrt(5)))],
                'obj_places': 5,
                'var_places': 5,
            },
        )

    def test_mutable_param_in_quadratic_constraint_monomial_form(self):
        m = pyo.ConcreteModel()
        m.p = pyo.Param(mutable=True, initialize=1.0)
        m.x = pyo.Var(bounds=(0, None))
        m.y = pyo.Var(bounds=(0, None))
        m.qc = pyo.Constraint(expr=m.x**2 + (m.p * m.y) * m.y <= 1)
        m.obj = pyo.Objective(expr=-(m.x + m.y))
        _solve_and_check(
            self,
            self.opt,
            m,
            {
                'objective': -math.sqrt(2),
                'vars': [(m.x, math.sqrt(2) / 2), (m.y, math.sqrt(2) / 2)],
                'obj_places': 5,
                'var_places': 5,
            },
        )
        m.p.set_value(4.0)
        _solve_and_check(
            self,
            self.opt,
            m,
            {
                'objective': -(2 / math.sqrt(5) + 1 / (2 * math.sqrt(5))),
                'vars': [(m.x, 2 / math.sqrt(5)), (m.y, 1 / (2 * math.sqrt(5)))],
                'obj_places': 5,
                'var_places': 5,
            },
        )

    def test_mutable_quadratic_coef_plus_mutable_linear_coef_objective(self):
        m = pyo.ConcreteModel()
        m.p1 = pyo.Param(mutable=True, initialize=1.0)
        m.p2 = pyo.Param(mutable=True, initialize=1.0)
        m.p3 = pyo.Param(mutable=True, initialize=4.0)
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.obj = pyo.Objective(
            expr=m.p1 * (m.x - 1) ** 2 + m.p2 * (m.y - 6) ** 2 - m.p3 * m.y
        )
        m.c = pyo.Constraint(expr=m.x >= m.y)
        _solve_and_check(
            self, self.opt, m, {'objective': -3.5, 'vars': [(m.x, 4.5), (m.y, 4.5)]}
        )
        m.p2.set_value(2.0)
        _solve_and_check(
            self, self.opt, m, {'objective': -2.0, 'vars': [(m.x, 5.0), (m.y, 5.0)]}
        )

    def test_mutable_quadratic_coef_persistent_analytic(self):
        m = pyo.ConcreteModel()
        m.p = pyo.Param(mutable=True, initialize=1.0)
        m.x = pyo.Var(bounds=(0, None))
        m.y = pyo.Var(bounds=(0, None))
        m.qc = pyo.Constraint(expr=m.x**2 + m.p * m.y**2 <= 1)
        m.obj = pyo.Objective(expr=-(m.x + m.y))
        _solve_and_check(
            self,
            self.opt,
            m,
            {
                'objective': -math.sqrt(2),
                'vars': [(m.x, math.sqrt(2) / 2), (m.y, math.sqrt(2) / 2)],
                'obj_places': 5,
                'var_places': 5,
            },
        )

        m.p.set_value(4.0)
        x_analytic = 2.0 / math.sqrt(5)
        y_analytic = 1.0 / (2.0 * math.sqrt(5))
        _solve_and_check(
            self,
            self.opt,
            m,
            {
                'objective': -(x_analytic + y_analytic),
                'vars': [(m.x, x_analytic), (m.y, y_analytic)],
            },
        )

    def test_nl_cubic_constraint_persistent(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 10))
        m.c = pyo.Constraint(expr=m.x**3 >= 1)
        m.obj = pyo.Objective(expr=m.x)
        _solve_and_check(self, self.opt, m, {'objective': 1.0, 'vars': [(m.x, 1.0)]})

    def test_nl_cubic_objective_persistent(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 1))
        m.obj = pyo.Objective(expr=m.x**3)
        _solve_and_check(self, self.opt, m, {'objective': 0.0, 'vars': [(m.x, 0.0)]})


@unittest.pytest.mark.solver('xpress_persistent')
class TestXpressPersistentMisc(unittest.TestCase):

    def setUp(self):
        self.opt = XpressPersistent()

    def test_mutable_param_in_objective_coefficient(self):
        m = pyo.ConcreteModel()
        m.p = pyo.Param(mutable=True, initialize=1.0)
        m.x = pyo.Var(bounds=(0, 10))
        m.obj = pyo.Objective(expr=m.p * m.x, sense=pyo.maximize)
        _solve_check_mutate_check(
            self,
            self.opt,
            m,
            {'objective': 10.0, 'vars': [(m.x, 10.0)]},
            m.p,
            -1.0,
            {'objective': 0.0, 'vars': [(m.x, 0.0)]},
        )

    def test_mutable_param_as_variable_bound(self):
        m = pyo.ConcreteModel()
        m.p = pyo.Param(mutable=True, initialize=5.0)
        m.x = pyo.Var(bounds=(0, m.p))
        m.obj = pyo.Objective(expr=-m.x)
        _solve_check_mutate_check(
            self,
            self.opt,
            m,
            {'objective': -5.0, 'vars': [(m.x, 5.0)]},
            m.p,
            3.0,
            {'objective': -3.0, 'vars': [(m.x, 3.0)]},
        )

    def test_has_instance(self):
        self.assertFalse(self.opt.has_instance())
        m = _trivial_model()
        self.opt.set_instance(m)
        self.assertTrue(self.opt.has_instance())
        self.opt.release()
        self.assertFalse(self.opt.has_instance())

    def test_add_variables_empty_list(self):
        m = _trivial_model()
        self.opt.set_instance(m)
        ncols_before = self.opt._xp_prob.attributes.cols
        self.opt.add_variables([])
        self.assertEqual(self.opt._xp_prob.attributes.cols, ncols_before)

    def test_add_constraints_empty_list(self):
        m = _trivial_model()
        self.opt.set_instance(m)
        self.opt.add_constraints([])
        self.opt._add_constraints([])

    def test_add_block_sos_only(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3], domain=pyo.NonNegativeReals, bounds=(0, 1))
        m.obj = pyo.Objective(expr=m.x[1] + 2 * m.x[2] + 3 * m.x[3], sense=pyo.maximize)
        self.opt.set_instance(m)
        m.b = pyo.Block()
        m.b.sos = pyo.SOSConstraint(var=m.x, sos=1, weights={1: 1.0, 2: 2.0, 3: 3.0})
        self.opt.add_block(m.b)
        _solve_and_check(
            self,
            self.opt,
            m,
            {'objective': 3.0, 'vars': [(m.x[1], 0.0), (m.x[2], 0.0), (m.x[3], 1.0)]},
        )

    def test_remove_block_sos_only(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3], domain=pyo.NonNegativeReals, bounds=(0, 1))
        m.obj = pyo.Objective(expr=m.x[1] + 2 * m.x[2] + 3 * m.x[3], sense=pyo.maximize)
        m.b = pyo.Block()
        m.b.sos = pyo.SOSConstraint(var=m.x, sos=1, weights={1: 1.0, 2: 2.0, 3: 3.0})
        self.opt.set_instance(m)
        _solve_and_check(
            self,
            self.opt,
            m,
            {'objective': 3.0, 'vars': [(m.x[1], 0.0), (m.x[2], 0.0), (m.x[3], 1.0)]},
        )
        self.opt.remove_block(m.b)
        m.b.deactivate()
        _solve_and_check(
            self,
            self.opt,
            m,
            {'objective': 6.0, 'vars': [(m.x[1], 1.0), (m.x[2], 1.0), (m.x[3], 1.0)]},
        )

    def test_objective_sense_change_only(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 5))
        m.obj = pyo.Objective(expr=m.x, sense=pyo.minimize)
        _solve_and_check(self, self.opt, m, {'objective': 0.0, 'vars': [(m.x, 0.0)]})
        m.obj.sense = pyo.maximize
        _solve_and_check(self, self.opt, m, {'objective': 5.0, 'vars': [(m.x, 5.0)]})

    def test_constant_objective_persistent(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 10))
        m.c = pyo.Constraint(expr=m.x >= 1)
        m.obj = pyo.Objective(expr=7.0)
        _solve_and_check(self, self.opt, m, {'objective': 7.0, 'vars': [(m.x, 1.0)]})

    def test_range_constraint_persistent(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(domain=pyo.NonNegativeReals)
        m.y = pyo.Var(domain=pyo.NonNegativeReals)
        m.c = pyo.Constraint(expr=pyo.inequality(1, m.x + m.y, 3))
        m.obj = pyo.Objective(expr=-2 * m.x - m.y)
        _solve_and_check(
            self, self.opt, m, {'objective': -6.0, 'vars': [(m.x, 3.0), (m.y, 0.0)]}
        )

    def test_mutable_param_in_range_constraint(self):
        m = pyo.ConcreteModel()
        m.p = pyo.Param(mutable=True, initialize=1.0)
        m.x = pyo.Var(domain=pyo.NonNegativeReals)
        m.c = pyo.Constraint(expr=pyo.inequality(m.p, m.x, 5))
        m.obj = pyo.Objective(expr=m.x)
        _solve_check_mutate_check(
            self,
            self.opt,
            m,
            {'objective': 1.0, 'vars': [(m.x, 1.0)]},
            m.p,
            3.0,
            {'objective': 3.0, 'vars': [(m.x, 3.0)]},
        )

    def test_mutable_param_in_range_constraint_ub(self):
        m = pyo.ConcreteModel()
        m.p = pyo.Param(mutable=True, initialize=5.0)
        m.x = pyo.Var(domain=pyo.NonNegativeReals)
        m.c = pyo.Constraint(expr=pyo.inequality(1, m.x, m.p))
        m.obj = pyo.Objective(expr=-m.x)
        _solve_check_mutate_check(
            self,
            self.opt,
            m,
            {'objective': -5.0, 'vars': [(m.x, 5.0)]},
            m.p,
            3.0,
            {'objective': -3.0, 'vars': [(m.x, 3.0)]},
        )

    def test_range_constraint_lower_bound_direction(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(domain=pyo.NonNegativeReals)
        m.y = pyo.Var(domain=pyo.NonNegativeReals)
        m.c = pyo.Constraint(expr=pyo.inequality(1, m.x + m.y, 3))
        m.obj = pyo.Objective(expr=m.x + 2 * m.y)
        _solve_and_check(
            self, self.opt, m, {'objective': 1.0, 'vars': [(m.x, 1.0), (m.y, 0.0)]}
        )

    def test_mutable_param_changes_constraint_coefficient(self):
        m = pyo.ConcreteModel()
        m.p = pyo.Param(mutable=True, initialize=1.0)
        m.x = pyo.Var(bounds=(0, 10))
        m.c = pyo.Constraint(expr=m.p * m.x <= 5)
        m.obj = pyo.Objective(expr=-m.x)
        _solve_check_mutate_check(
            self,
            self.opt,
            m,
            {'objective': -5.0, 'vars': [(m.x, 5.0)]},
            m.p,
            2.0,
            {'objective': -2.5, 'vars': [(m.x, 2.5)]},
        )

    def test_fix_unfix_variable_via_bounds(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 10))
        m.y = pyo.Var(bounds=(0, 10))
        m.c = pyo.Constraint(expr=m.x + m.y <= 8)
        m.obj = pyo.Objective(expr=-2 * m.x - m.y)
        _solve_and_check(
            self, self.opt, m, {'objective': -16.0, 'vars': [(m.x, 8.0), (m.y, 0.0)]}
        )
        m.y.fix(2.0)
        _solve_and_check(
            self, self.opt, m, {'objective': -14.0, 'vars': [(m.x, 6.0), (m.y, 2.0)]}
        )
        m.y.unfix()
        _solve_and_check(
            self, self.opt, m, {'objective': -16.0, 'vars': [(m.x, 8.0), (m.y, 0.0)]}
        )

    def test_remove_constraint_drops_mutable_helper(self):
        m = pyo.ConcreteModel()
        m.p = pyo.Param(mutable=True, initialize=1.0)
        m.x = pyo.Var(bounds=(0, 10))
        m.c = pyo.Constraint(expr=m.p * m.x <= 5)
        m.obj = pyo.Objective(expr=-m.x)
        self.opt.set_instance(m)
        _solve_and_check(self, self.opt, m, {'objective': -5.0, 'vars': [(m.x, 5.0)]})
        self.assertIn(m.c, self.opt._mutable_helpers)
        self.opt.remove_constraints([m.c])
        m.c.deactivate()
        self.assertNotIn(m.c, self.opt._mutable_helpers)
        m.p.set_value(2.0)
        _solve_and_check(self, self.opt, m, {'objective': -10.0, 'vars': [(m.x, 10.0)]})

    def test_warmstart_column_indices_match_after_variable_removal(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(within=pyo.Binary)
        m.y = pyo.Var(within=pyo.Binary)
        m.z = pyo.Var(within=pyo.Binary)
        m.obj = pyo.Objective(expr=m.x + m.y + m.z)
        m.c = pyo.Constraint(expr=m.x + m.y + m.z >= 1)
        self.opt.set_instance(m)
        self.opt.remove_variables([m.y])
        del m.y
        for j, var in enumerate(self.opt._vars):
            xp_idx = self.opt._maps.vars[id(var)].index
            self.assertEqual(
                j,
                xp_idx,
                f"After variable removal: Python list position {j} != "
                f"Xpress column index {xp_idx} for {var.name}",
            )
        m.x.set_value(1)
        m.z.set_value(0)
        self.opt.remove_constraints([m.c])
        m.c.deactivate()
        _solve_and_check(
            self, self.opt, m, {'objective': 0.0, 'vars': [(m.x, 0.0), (m.z, 0.0)]}
        )


@unittest.pytest.mark.solver('xpress_persistent')
class TestXpressPersistentNLP(unittest.TestCase):
    """NLP integration tests for the persistent interface."""

    def setUp(self):
        self.opt = XpressPersistent()

    def _check_optimal(self, res):
        self.assertEqual(
            res.termination_condition, TerminationCondition.convergenceCriteriaSatisfied
        )

    def test_nl_add_constraint_registers_nl_rebuild(self):
        m = pyo.ConcreteModel()
        m.p = pyo.Param(mutable=True, initialize=2.0)
        m.x = pyo.Var(bounds=(0, 10))
        m.c = pyo.Constraint(expr=m.p * pyo.sin(m.x) <= 5)
        m.obj = pyo.Objective(expr=m.x)
        self.opt.set_instance(m)
        self.opt.solve(m)
        self.assertIn(m.c, self.opt._mutable_helpers)
        self.assertIsNotNone(self.opt._mutable_helpers[m.c]._nl_expr)

    def test_nl_add_constraint_always_registered(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, math.pi))
        m.c = pyo.Constraint(expr=pyo.sin(m.x) <= 0.5)
        m.obj = pyo.Objective(expr=m.x)
        self.opt.set_instance(m)
        self.opt.solve(m)
        self.assertIn(m.c, self.opt._mutable_helpers)
        helper = self.opt._mutable_helpers[m.c]
        self.assertIsNotNone(helper._nl_expr)
        self.assertEqual(len(helper._lin_coefs), 0)
        self.assertEqual(len(helper._quad_coefs), 0)

    def test_nl_remove_constraint_cleans_up(self):
        m = pyo.ConcreteModel()
        m.p = pyo.Param(mutable=True, initialize=2.0)
        m.x = pyo.Var(bounds=(0, 10))
        m.c = pyo.Constraint(expr=m.p * pyo.sin(m.x) <= 5)
        m.obj = pyo.Objective(expr=m.x)
        self.opt.set_instance(m)
        self.opt.solve(m)
        self.assertIn(m.c, self.opt._mutable_helpers)
        self.opt.remove_constraints([m.c])
        m.c.deactivate()
        self.assertNotIn(m.c, self.opt._mutable_helpers)

    def test_nl_mutable_linear_coef_in_nl_constraint(self):
        m = pyo.ConcreteModel()
        m.p = pyo.Param(mutable=True, initialize=1.0)
        m.x = pyo.Var(bounds=(0, math.pi))
        m.y = pyo.Var(bounds=(0, 10))
        m.c = pyo.Constraint(expr=pyo.sin(m.x) + m.p * m.y <= 5)
        m.obj = pyo.Objective(expr=-m.y)
        self.opt.set_instance(m)
        _solve_and_check(
            self, self.opt, m, {'objective': -5.0, 'vars': [(m.x, 0.0), (m.y, 5.0)]}
        )
        y1 = pyo.value(m.y)
        m.p.set_value(2.0)
        _solve_and_check(
            self, self.opt, m, {'objective': -2.5, 'vars': [(m.x, 0.0), (m.y, 2.5)]}
        )
        y2 = pyo.value(m.y)
        self.assertLess(y2, y1)

    def test_nl_mutable_nl_coef_full_rebuild(self):
        m = pyo.ConcreteModel()
        m.p = pyo.Param(mutable=True, initialize=1.0)
        m.x = pyo.Var(bounds=(0, math.pi / 2))
        m.c = pyo.Constraint(expr=m.p * pyo.sin(m.x) <= 0.5)
        m.obj = pyo.Objective(expr=-m.x)
        self.opt.set_instance(m)
        _solve_and_check(
            self,
            self.opt,
            m,
            {'objective': -math.asin(0.5), 'vars': [(m.x, math.asin(0.5))]},
        )
        x1 = pyo.value(m.x)
        m.p.set_value(2.0)
        _solve_and_check(
            self,
            self.opt,
            m,
            {'objective': -math.asin(0.25), 'vars': [(m.x, math.asin(0.25))]},
        )
        x2 = pyo.value(m.x)
        self.assertLess(x2, x1 - 0.1)

    def test_nl_mutable_bound(self):
        m = pyo.ConcreteModel()
        m.p = pyo.Param(mutable=True, initialize=0.5)
        m.x = pyo.Var(bounds=(0, math.pi / 2))
        m.c = pyo.Constraint(expr=pyo.sin(m.x) >= m.p)
        m.obj = pyo.Objective(expr=m.x)
        self.opt.set_instance(m)
        _solve_and_check(
            self,
            self.opt,
            m,
            {'objective': math.asin(0.5), 'vars': [(m.x, math.asin(0.5))]},
        )
        x1 = pyo.value(m.x)

        m.p.set_value(0.9)
        _solve_and_check(
            self,
            self.opt,
            m,
            {'objective': math.asin(0.9), 'vars': [(m.x, math.asin(0.9))]},
        )
        x2 = pyo.value(m.x)
        self.assertGreater(x2, x1 + 0.4)

    def test_nl_solve_modify_resolve(self):
        m = pyo.ConcreteModel()
        m.p = pyo.Param(mutable=True, initialize=1.0)
        m.x = pyo.Var(bounds=(0, 3))
        m.c = pyo.Constraint(expr=pyo.exp(m.x) >= m.p)
        m.obj = pyo.Objective(expr=m.x)
        self.opt.set_instance(m)
        _solve_check_mutate_check(
            self,
            self.opt,
            m,
            {'objective': 0.0, 'vars': [(m.x, 0.0)]},
            m.p,
            math.e,
            {'objective': 1.0, 'vars': [(m.x, 1.0)]},
        )

    def test_fix_variable_no_nl_constraint_rebuild(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, math.pi))
        m.y = pyo.Var(bounds=(0, 10))
        m.c = pyo.Constraint(expr=pyo.sin(m.x) + m.y <= 5)
        m.obj = pyo.Objective(expr=m.y)
        self.opt.set_instance(m)
        _solve_and_check(
            self, self.opt, m, {'objective': 0.0, 'vars': [(m.x, 0.0), (m.y, 0.0)]}
        )
        xp_con_before = self.opt._mutable_helpers[m.c]._xp_con
        nrows_before = self.opt._xp_prob.attributes.rows
        m.x.fix(math.pi / 6)
        _solve_and_check(
            self,
            self.opt,
            m,
            {'objective': 0.0, 'vars': [(m.x, math.pi / 6), (m.y, 0.0)]},
        )
        nrows_after = self.opt._xp_prob.attributes.rows
        self.assertEqual(nrows_before, nrows_after)
        self.assertIs(self.opt._mutable_helpers[m.c]._xp_con, xp_con_before)

    def _run_nl_linear_shared_param_test(self, nl_first: bool):
        """Test same param in NL (rebuild) and linear (chgMCoef) constraints.

        nl_first=True: NL at row 0, linear at row 1 (delConstraint shifts linear).
        nl_first=False: linear at row 0, NL at row 1 (no shift on linear).
        """
        m = pyo.ConcreteModel()
        m.p = pyo.Param(mutable=True, initialize=1.0)
        m.x = pyo.Var(bounds=(0, math.pi / 2))
        m.y = pyo.Var(bounds=(0, 10))
        if nl_first:
            m.c_nl = pyo.Constraint(expr=m.p * pyo.sin(m.x) <= 0.5)
            m.c_lin = pyo.Constraint(expr=m.p * m.y <= 4)
        else:
            m.c_lin = pyo.Constraint(expr=m.p * m.y <= 4)
            m.c_nl = pyo.Constraint(expr=m.p * pyo.sin(m.x) <= 0.5)
        m.obj = pyo.Objective(expr=m.x + m.y, sense=pyo.maximize)
        self.opt.set_instance(m)
        _solve_and_check(
            self,
            self.opt,
            m,
            {
                'objective': 4.0 + math.asin(0.5),
                'vars': [(m.y, 4.0), (m.x, math.asin(0.5))],
            },
        )

        m.p.set_value(2.0)
        _solve_and_check(
            self,
            self.opt,
            m,
            {
                'objective': 2.0 + math.asin(0.25),
                'vars': [(m.y, 2.0), (m.x, math.asin(0.25))],
            },
        )

    def test_param_shared_nl_before_linear(self):
        self._run_nl_linear_shared_param_test(nl_first=True)

    def test_param_shared_linear_before_nl(self):
        self._run_nl_linear_shared_param_test(nl_first=False)

    def test_param_shared_multiple_nl_and_linear(self):
        m = pyo.ConcreteModel()
        m.p = pyo.Param(mutable=True, initialize=1.0)
        m.x = pyo.Var(bounds=(0, math.pi))
        m.y = pyo.Var(bounds=(0, 10))
        m.z = pyo.Var(bounds=(0, 10))
        m.c_nl1 = pyo.Constraint(expr=m.p * pyo.sin(m.x) <= 5)
        m.c_lin1 = pyo.Constraint(expr=m.p * m.y <= 4)
        m.c_nl2 = pyo.Constraint(expr=m.p * pyo.cos(m.x) >= -1)
        m.c_lin2 = pyo.Constraint(expr=m.p * m.z <= 3)
        m.obj = pyo.Objective(expr=m.y + m.z, sense=pyo.maximize)
        self.opt.set_instance(m)
        _solve_and_check(
            self,
            self.opt,
            m,
            {'objective': 7.0, 'vars': [(m.x, math.pi / 2), (m.y, 4.0), (m.z, 3.0)]},
        )
        self.assertAlmostEqual(pyo.value(m.y) + pyo.value(m.z), 7.0, places=6)

        m.p.set_value(2.0)
        _solve_and_check(
            self,
            self.opt,
            m,
            {'objective': 3.5, 'vars': [(m.x, math.pi / 3), (m.y, 2.0), (m.z, 1.5)]},
        )
        self.assertAlmostEqual(pyo.value(m.y) + pyo.value(m.z), 3.5, places=6)

    def test_param_quadratic_rebuild_before_linear_collect(self):
        m = pyo.ConcreteModel()
        m.p = pyo.Param(mutable=True, initialize=1.0)
        m.x = pyo.Var(bounds=(0, 5))
        m.y = pyo.Var(bounds=(0, 10))
        m.c_quad = pyo.Constraint(expr=m.p * m.x**2 <= 5)

        m.c_lin = pyo.Constraint(expr=m.p * m.y <= 4)
        m.obj = pyo.Objective(expr=m.y, sense=pyo.maximize)
        self.opt.set_instance(m)
        _solve_and_check(
            self, self.opt, m, {'objective': 4.0, 'vars': [(m.x, 0.0), (m.y, 4.0)]}
        )

        m.p.set_value(2.0)
        _solve_and_check(
            self, self.opt, m, {'objective': 2.0, 'vars': [(m.x, 0.0), (m.y, 2.0)]}
        )

    def test_nl_formula_mutable_and_affine_mutable_in_same_constraint(self):
        m = pyo.ConcreteModel()
        m.p = pyo.Param(mutable=True, initialize=1.0)
        m.q = pyo.Param(mutable=True, initialize=1.0)
        m.x = pyo.Var(bounds=(0, math.pi))
        m.y = pyo.Var(bounds=(0, 10))
        m.c = pyo.Constraint(expr=pyo.sin(m.p * m.x) + m.q * m.y <= 5)
        m.obj = pyo.Objective(expr=-m.y)
        self.opt.set_instance(m)
        _solve_and_check(
            self, self.opt, m, {'objective': -5.0, 'vars': [(m.x, 0.0), (m.y, 5.0)]}
        )
        self.assertIn(m.c, self.opt._mutable_helpers)
        self.assertIsNotNone(self.opt._mutable_helpers[m.c]._nl_expr)
        pyo.value(m.y)

        m.q.set_value(2.0)
        _solve_and_check(
            self, self.opt, m, {'objective': -2.5, 'vars': [(m.x, 0.0), (m.y, 2.5)]}
        )
        pyo.value(m.y)

        m.p.set_value(0.0)
        _solve_and_check(
            self, self.opt, m, {'objective': -2.5, 'vars': [(m.x, 0.0), (m.y, 2.5)]}
        )
        pyo.value(m.y)

    def test_mutable_objective_does_not_interfere_with_linear_update(self):
        m = pyo.ConcreteModel()
        m.p = pyo.Param(mutable=True, initialize=1.0)
        m.x = pyo.Var(bounds=(0, 3))
        m.y = pyo.Var(bounds=(0, 10))
        m.c_nl = pyo.Constraint(expr=m.p * pyo.sin(m.x) <= 2)
        m.c_lin = pyo.Constraint(expr=m.p * m.y <= 4)
        m.obj = pyo.Objective(expr=m.p * m.y, sense=pyo.maximize)
        self.opt.set_instance(m)
        _solve_and_check(
            self, self.opt, m, {'objective': 4.0, 'vars': [(m.x, 1.5), (m.y, 4.0)]}
        )

        m.p.set_value(2.0)
        _solve_and_check(
            self, self.opt, m, {'objective': 4.0, 'vars': [(m.x, 1.5), (m.y, 2.0)]}
        )

    def test_nl_mutable_objective_persistent(self):
        m = pyo.ConcreteModel()
        m.p = pyo.Param(mutable=True, initialize=1.0)
        m.x = pyo.Var(bounds=(0, math.pi / 2))
        m.obj = pyo.Objective(expr=m.p * pyo.sin(m.x), sense=pyo.maximize)
        _solve_and_check(
            self, self.opt, m, {'objective': 1.0, 'vars': [(m.x, math.pi / 2)]}
        )
        pyo.value(m.x)

        m.p.set_value(-1.0)
        _solve_and_check(self, self.opt, m, {'objective': 0.0, 'vars': [(m.x, 0.0)]})
        pyo.value(m.x)

    def test_nl_objective_stable_xp_not_mutated_by_constant_update(self):
        import math

        m = pyo.ConcreteModel()
        m.p = pyo.Param(mutable=True, initialize=1.0)
        m.x = pyo.Var(bounds=(0, 1))
        m.z = pyo.Var(bounds=(0, math.pi / 2))
        m.obj = pyo.Objective(expr=2 * m.x + m.p + pyo.sin(m.z))
        _solve_and_check(
            self, self.opt, m, {'objective': 1.0, 'vars': [(m.x, 0.0), (m.z, 0.0)]}
        )

        m.p.set_value(3.0)
        _solve_and_check(
            self, self.opt, m, {'objective': 3.0, 'vars': [(m.x, 0.0), (m.z, 0.0)]}
        )

        m.p.set_value(5.0)
        _solve_and_check(
            self, self.opt, m, {'objective': 5.0, 'vars': [(m.x, 0.0), (m.z, 0.0)]}
        )

    def test_nl_constraint_stable_quadratic_term(self):
        m = pyo.ConcreteModel()
        m.p = pyo.Param(mutable=True, initialize=1.0)
        m.x = pyo.Var(bounds=(0, 3))
        m.y = pyo.Var(bounds=(0, 1))
        m.c = pyo.Constraint(expr=2 * m.x**2 + m.p * pyo.exp(m.y) <= 5)
        m.obj = pyo.Objective(expr=-m.x)
        _solve_and_check(
            self,
            self.opt,
            m,
            {'objective': -math.sqrt(2), 'vars': [(m.x, math.sqrt(2)), (m.y, 0.0)]},
        )
        x1 = pyo.value(m.x)

        m.p.set_value(2.0)
        _solve_and_check(
            self,
            self.opt,
            m,
            {'objective': -math.sqrt(1.5), 'vars': [(m.x, math.sqrt(1.5)), (m.y, 0.0)]},
        )
        x2 = pyo.value(m.x)
        self.assertLess(x2, x1 - 0.1)

    def test_nl_objective_stable_lin_quad_terms(self):
        m = pyo.ConcreteModel()
        m.p = pyo.Param(mutable=True, initialize=1.0)
        m.x = pyo.Var(bounds=(0, 1))
        m.y = pyo.Var(bounds=(0, 1))
        m.z = pyo.Var(bounds=(0, math.pi / 2))
        m.obj = pyo.Objective(expr=2 * m.x + m.y**2 + m.p * pyo.sin(m.z))
        _solve_and_check(
            self,
            self.opt,
            m,
            {'objective': 0.0, 'vars': [(m.x, 0.0), (m.y, 0.0), (m.z, 0.0)]},
        )

        m.p.set_value(-1.0)
        _solve_and_check(
            self,
            self.opt,
            m,
            {'objective': -1.0, 'vars': [(m.x, 0.0), (m.y, 0.0), (m.z, math.pi / 2)]},
        )

    def test_nl_cubic_constraint_mutable_param_persistent(self):
        m = pyo.ConcreteModel()
        m.p = pyo.Param(mutable=True, initialize=1.0)
        m.x = pyo.Var(bounds=(0, 10))
        m.c = pyo.Constraint(expr=m.p * m.x**3 >= 1)
        m.obj = pyo.Objective(expr=m.x)
        _solve_check_mutate_check(
            self,
            self.opt,
            m,
            {'objective': 1.0, 'vars': [(m.x, 1.0)]},
            m.p,
            8.0,
            {'objective': 0.5, 'vars': [(m.x, 0.5)]},
        )

    def test_add_remove_readd_changes_row_ordering(self):
        m = pyo.ConcreteModel()
        m.p = pyo.Param(mutable=True, initialize=1.0)
        m.x = pyo.Var(bounds=(0, math.pi))
        m.y = pyo.Var(bounds=(0, 10))
        m.c_lin = pyo.Constraint(expr=m.p * m.y <= 4)
        m.c_nl = pyo.Constraint(expr=m.p * pyo.sin(m.x) <= 5)
        m.obj = pyo.Objective(expr=m.y, sense=pyo.maximize)
        self.opt.set_instance(m)
        _solve_and_check(
            self,
            self.opt,
            m,
            {'objective': 4.0, 'vars': [(m.x, math.pi / 2), (m.y, 4.0)]},
        )

        m.c_lin.deactivate()
        self.opt.remove_constraints([m.c_lin])
        m.c_lin.activate()
        self.opt.add_constraints([m.c_lin])
        self.assertIn(m.c_lin, self.opt._mutable_helpers)

        m.p.set_value(2.0)
        _solve_and_check(
            self,
            self.opt,
            m,
            {'objective': 2.0, 'vars': [(m.x, math.pi / 2), (m.y, 2.0)]},
        )


@unittest.pytest.mark.solver('xpress_persistent')
class TestXpressPersistentIIS(unittest.TestCase):

    def setUp(self):
        self.opt = XpressPersistent()

    def _infeasible_model(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(within=pyo.Binary)
        m.y = pyo.Var(within=pyo.NonNegativeReals)
        m.c1 = pyo.Constraint(expr=m.y <= 100.0 * m.x)
        m.c2 = pyo.Constraint(expr=m.y <= -100.0 * m.x)
        m.c3 = pyo.Constraint(expr=m.x >= 0.5)
        m.obj = pyo.Objective(expr=-m.y)
        return m

    def test_write_iis_produces_file(self):
        import os, tempfile

        m = self._infeasible_model()
        self.opt.solve(
            m,
            raise_exception_on_nonoptimal_result=False,
            load_solutions=False,
            symbolic_solver_labels=True,
        )
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, 'iis')
            result = self.opt.write_iis(base)
            self.assertEqual(result, base)
            lp_file = base + '.lp'
            self.assertTrue(os.path.exists(lp_file))
            with open(lp_file) as f:
                content = f.read()
            self.assertIn('c2', content)
            self.assertIn('c3', content)

    def test_get_iis_returns_pyomo_objects(self):
        m = self._infeasible_model()
        self.opt.solve(
            m, raise_exception_on_nonoptimal_result=False, load_solutions=False
        )
        iis = self.opt.get_iis()
        self.assertIn('constraints', iis)
        self.assertIn('variables', iis)
        con_names = {c.name for c in iis['constraints']}
        self.assertIn('c2', con_names)
        self.assertIn('c3', con_names)
        var_names = {v.name for v in iis['variables']}
        self.assertIn('y', var_names)

    def test_get_iis_objects_are_model_constraints(self):
        m = self._infeasible_model()
        self.opt.solve(
            m, raise_exception_on_nonoptimal_result=False, load_solutions=False
        )
        iis = self.opt.get_iis()
        model_cons = list(m.component_data_objects(pyo.Constraint, active=True))
        model_vars = list(m.component_data_objects(pyo.Var))
        for con in iis['constraints']:
            self.assertTrue(
                any(con is c for c in model_cons),
                f"{con.name} is not a model constraint object",
            )
        for var in iis['variables']:
            self.assertTrue(
                any(var is v for v in model_vars),
                f"{var.name} is not a model variable object",
            )


if __name__ == '__main__':
    unittest.main()
