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
        # Exercises the Reason.removed path in _update_objectives, which calls
        # _set_objective(None). Not covered by any parameterized test because
        # those use set_instance (new model), not incremental update.
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
        # Pyomo idiom: define multiple Objective components as alternatives,
        # toggle which one is active via activate()/deactivate(). The change
        # detector treats deactivate as Reason.removed and activate as
        # Reason.added; our handler clears the old and sets the new.
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
        # Solve once with one active obj, then activate a second
        # without deactivating the first. The detector fires _update_objectives
        # with only the new obj added; the multi-active hardening must catch
        # the model state and raise.
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 5))
        m.obj1 = pyo.Objective(expr=m.x, sense=pyo.minimize)
        self.opt.solve(m)

        m.obj2 = pyo.Objective(expr=-m.x, sense=pyo.minimize)
        with self.assertRaises(IncompatibleModelError):
            self.opt.solve(m)

    def test_recovery_after_two_objectives_raises(self):
        # After a two-active-obj raise, the solver must be recoverable via
        # set_instance. The IncompatibleModelError leaves the Xpress problem in
        # a partially-updated state (obj2 was written before the raise). Calling
        # set_instance rebuilds from scratch and resolves cleanly.
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 5))
        m.obj1 = pyo.Objective(expr=m.x, sense=pyo.minimize)
        self.opt.solve(m)

        m.obj2 = pyo.Objective(expr=-m.x, sense=pyo.minimize)
        with self.assertRaises(IncompatibleModelError):
            self.opt.solve(m)

        # Recovery: deactivate the conflicting objective and reset via set_instance.
        m.obj2.deactivate()
        self.opt.set_instance(m)
        _solve_and_check(self, self.opt, m, {'objective': 0.0, 'vars': [(m.x, 0.0)]})


@unittest.pytest.mark.solver('xpress_persistent')
class TestXpressPersistentLifecycle(unittest.TestCase):
    def setUp(self):
        self.opt = XpressPersistent()

    def test_eager_invalidation_on_mutation(self):
        # After solve, the loader works. After any mutation, the loader must
        # be eagerly invalidated -- subsequent reads raise NoSolutionError.
        m = _simple_lp()
        res = _solve_and_check(
            self, self.opt, m, {'objective': -8.0, 'vars': [(m.x, 0.0), (m.y, 4.0)]}
        )
        # baseline: loader works right after solve
        res.solution_loader.get_vars()
        # mutate: add a constraint
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
        # With symbolic_solver_labels=True the user's Pyomo names should
        # appear in the LP-format output; without it they would not.
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
        # auto_updates.update_parameters=False tells the detector to skip
        # parameter change tracking. Changing a mutable param between solves
        # should then have no effect on the loaded problem.
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 10))
        m.p = pyo.Param(mutable=True, initialize=5.0)
        m.c = pyo.Constraint(expr=m.x <= m.p)
        m.obj = pyo.Objective(expr=-m.x)

        _solve_and_check(self, self.opt, m, {'objective': -5.0, 'vars': [(m.x, 5.0)]})

        m.p.value = 7.0
        # Parameter change was ignored: optimum stays at 5.0, not 7.0
        _solve_and_check(
            self,
            self.opt,
            m,
            {'objective': -5.0, 'vars': [(m.x, 5.0)]},
            auto_updates={'update_parameters': False},
        )

        # Positive control: with default (full) auto_updates the param change IS
        # picked up -- distinguishes "tracking correctly disabled" from
        # "_update_parameters always broken".
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
        # warmstart=False: the solve must complete correctly even with stale
        # variable values set on the model (the hint is not passed to the solver).
        # This test verifies the config branch does not crash or corrupt the solve;
        # it cannot verify that addMipSol was skipped without a mock.
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
        # SOS1: at most one variable nonzero. Maximize x1+2*x2+3*x3 s.t. xi in [0,1].
        # With SOS1: optimal is x3=1, others=0 (obj=3).
        # After removing SOS1: all vars reach upper bound (obj=6).
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

    # -- Public persistent API (explicit call paths) --

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
        # Deactivate old obj in Pyomo model before adding new one -- the
        # _update_objectives active-objective scan would raise on two active objs.
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
        # Deactivate in Pyomo model so auto-update does not re-add it on next solve.
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
        # Deactivate block in Pyomo model so auto-update does not re-add it on next solve.
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
        # Full Xpress API accessible: use prob + entity handles to read slack
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
        # min x^2 + y^2  s.t. x + y >= 1  ->  optimal x=y=0.5, obj=0.5
        m = pyo.ConcreteModel()
        m.x = pyo.Var(domain=pyo.NonNegativeReals)
        m.y = pyo.Var(domain=pyo.NonNegativeReals)
        m.c = pyo.Constraint(expr=m.x + m.y >= 1)
        m.obj = pyo.Objective(expr=m.x**2 + m.y**2)
        _solve_and_check(
            self, self.opt, m, {'objective': 0.5, 'vars': [(m.x, 0.5), (m.y, 0.5)]}
        )
        # re-solve (tests persistent re-use)
        _solve_and_check(
            self, self.opt, m, {'objective': 0.5, 'vars': [(m.x, 0.5), (m.y, 0.5)]}
        )

    def test_qcp_add_remove_persistent(self):
        # Without QC: min -(x+y) s.t. x,y in [0,2]  ->  optimal at x=y=2.
        # Add QC x^2+y^2<=1 -> optimal at x=y=1/sqrt(2).
        # Remove QC -> optimal returns to x=y=2.
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

        # Remove QC: optimal returns to x=y=2 (no longer constrained by unit circle).
        self.opt.remove_constraints([m.qc])
        m.qc.deactivate()
        _solve_and_check(
            self, self.opt, m, {'objective': -4.0, 'vars': [(m.x, 2.0), (m.y, 2.0)]}
        )

    def test_mutable_param_in_quadratic_obj(self):
        # min p*x^2  s.t. x >= 1. Change p, verify chgMQObj is called.
        m = pyo.ConcreteModel()
        m.p = pyo.Param(mutable=True, initialize=1.0)
        m.x = pyo.Var(bounds=(1, None))
        m.obj = pyo.Objective(expr=m.p * m.x**2)
        # chgMQObj updates the quadratic obj coefficient in-place; accuracy ~1e-5
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
        # min -(x+y)  s.t. x^2 + p*y^2 <= 1. Change p, verify rebuild.
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
                # QCP solver for quadratic constraints achieves ~6e-7 on the circle
                'objective': -math.sqrt(2),
                'vars': [(m.x, math.sqrt(2) / 2), (m.y, math.sqrt(2) / 2)],
                'obj_places': 5,
                'var_places': 5,
            },
        )
        # p=4: ellipse x^2 + 4*y^2 <= 1. Lagrange: x=2/sqrt(5), y=1/(2*sqrt(5)).
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
        # Same math as test_mutable_param_in_quadratic_constraint but the
        # expression is written as (m.p * m.y) * m.y (MonomialTerm * Var path)
        # to exercise Case 2 of _before_product_bilinear.
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
        # p=4: ellipse x^2 + 4*y^2 <= 1. Lagrange: x=2/sqrt(5), y=1/(2*sqrt(5)).
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
        # Regression: p1*(x-1)^2 + p2*(y-6)^2 - p3*y has mutable quadratic coefs
        # (p1, p2) AND a mutable linear coef (p3). The linear term populated
        # _mutable_lin_vars which prevented the fallback from setting
        # _has_mutable_nl_formula, so the objective was wrongly tracked as
        # _MutableObjective (chgObj only) instead of requiring a full rebuild.
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
        # Change p2: new unconstrained min at (1, 8), constrained at x=y=5
        m.p2.set_value(2.0)
        _solve_and_check(
            self, self.opt, m, {'objective': -2.0, 'vars': [(m.x, 5.0), (m.y, 5.0)]}
        )

    def test_mutable_quadratic_coef_persistent_analytic(self):
        # min -(x+y) s.t. x^2 + p*y^2 <= 1.
        # p=4: ellipse x^2+4y^2<=1; by Lagrange multipliers:
        #   max x+y on ellipse -> x=2/sqrt(5) ~ 0.8944, y=1/(2*sqrt(5)) ~ 0.2236.
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
                # p=1: circle, QCP ~6e-7
                'objective': -math.sqrt(2),
                'vars': [(m.x, math.sqrt(2) / 2), (m.y, math.sqrt(2) / 2)],
                'obj_places': 5,
                'var_places': 5,
            },
        )

        m.p.set_value(4.0)
        # Analytic: x=2/sqrt(5), y=1/(2*sqrt(5))
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
        # x**3 >= 1, min x: optimal solution is x=1 via Xpress NLP solver.
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 10))
        m.c = pyo.Constraint(expr=m.x**3 >= 1)
        m.obj = pyo.Objective(expr=m.x)
        _solve_and_check(self, self.opt, m, {'objective': 1.0, 'vars': [(m.x, 1.0)]})

    def test_nl_cubic_objective_persistent(self):
        # min x**3 for x in [0, 1]: optimal is x=0 (boundary minimum).
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 1))
        m.obj = pyo.Objective(expr=m.x**3)
        _solve_and_check(self, self.opt, m, {'objective': 0.0, 'vars': [(m.x, 0.0)]})


@unittest.pytest.mark.solver('xpress_persistent')
class TestXpressPersistentMisc(unittest.TestCase):
    """Edge cases and rarely-hit branches for the persistent connector."""

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
        # has_instance() returns False before set_instance and True after (L565).
        self.assertFalse(self.opt.has_instance())
        m = _trivial_model()
        self.opt.set_instance(m)
        self.assertTrue(self.opt.has_instance())
        self.opt.release()
        self.assertFalse(self.opt.has_instance())

    def test_add_variables_empty_list(self):
        # _add_vars_impl early-return when list is empty (line 531 in xpress_base.py).
        m = _trivial_model()
        self.opt.set_instance(m)
        ncols_before = self.opt._xp_prob.attributes.cols
        self.opt.add_variables([])
        self.assertEqual(self.opt._xp_prob.attributes.cols, ncols_before)

    def test_add_constraints_empty_list(self):
        # _add_constraints early-return when list is empty.
        m = _trivial_model()
        self.opt.set_instance(m)
        self.opt.add_constraints([])  # public API path (via change detector)
        self.opt._add_constraints(
            []
        )  # direct path -- exercises the len==0 early return

    def test_add_block_sos_only(self):
        # add_block path with only SOS constraints (no linear constraints).
        # Exercises xpress_persistent.py:_add_block SOS branch.
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
        # remove_block path with only SOS constraints.
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
        # Toggling sense without touching expr exercises the Reason.sense-only
        # branch in _update_objectives.
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 5))
        m.obj = pyo.Objective(expr=m.x, sense=pyo.minimize)
        _solve_and_check(self, self.opt, m, {'objective': 0.0, 'vars': [(m.x, 0.0)]})
        m.obj.sense = pyo.maximize
        _solve_and_check(self, self.opt, m, {'objective': 5.0, 'vars': [(m.x, 5.0)]})

    def test_constant_objective_persistent(self):
        # Persistent path with constant-only objective.
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 10))
        m.c = pyo.Constraint(expr=m.x >= 1)
        m.obj = pyo.Objective(expr=7.0)
        # x value is solver-determined (feasibility only); Xpress returns x=1.0 (lb of constraint)
        _solve_and_check(self, self.opt, m, {'objective': 7.0, 'vars': [(m.x, 1.0)]})

    def test_range_constraint_persistent(self):
        # Range constraint on persistent solver.
        m = pyo.ConcreteModel()
        m.x = pyo.Var(domain=pyo.NonNegativeReals)
        m.y = pyo.Var(domain=pyo.NonNegativeReals)
        m.c = pyo.Constraint(expr=pyo.inequality(1, m.x + m.y, 3))
        # obj=-2x-y: unique optimal at upper bound is x=3, y=0 (x has larger coefficient)
        m.obj = pyo.Objective(expr=-2 * m.x - m.y)
        _solve_and_check(
            self, self.opt, m, {'objective': -6.0, 'vars': [(m.x, 3.0), (m.y, 0.0)]}
        )

    def test_mutable_param_in_range_constraint(self):
        # Range constraint with mutable lb. Exercises the lb_h branch in
        # _build_constraint_helper for rowtype == 'R'.
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
        # Symmetric to test_mutable_param_in_range_constraint (which tests mutable lb).
        # This tests mutable ub: 1 <= x <= p. Exercises the ub_h branch in
        # _build_constraint_helper for rowtype == 'R'.
        m = pyo.ConcreteModel()
        m.p = pyo.Param(mutable=True, initialize=5.0)
        m.x = pyo.Var(domain=pyo.NonNegativeReals)
        m.c = pyo.Constraint(expr=pyo.inequality(1, m.x, m.p))
        m.obj = pyo.Objective(expr=-m.x)  # maximize x
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
        # The existing test_range_constraint_lp maximizes (upper bound is binding).
        # This test minimizes so the LOWER bound is binding.
        m = pyo.ConcreteModel()
        m.x = pyo.Var(domain=pyo.NonNegativeReals)
        m.y = pyo.Var(domain=pyo.NonNegativeReals)
        m.c = pyo.Constraint(expr=pyo.inequality(1, m.x + m.y, 3))
        # obj=x+2y: unique optimal at lower bound is x=1, y=0 (y has larger coeff, min -> y=0)
        m.obj = pyo.Objective(expr=m.x + 2 * m.y)
        _solve_and_check(
            self, self.opt, m, {'objective': 1.0, 'vars': [(m.x, 1.0), (m.y, 0.0)]}
        )

    def test_mutable_param_changes_constraint_coefficient(self):
        # Coefficient updated via _MutableConstraint.collect path (chgMCoef).
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
        # Fixing a variable sets its Xpress column bounds to [val, val] via
        # chgBounds; Xpress handles the fixed variable natively without a
        # constraint rebuild.
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 10))
        m.y = pyo.Var(bounds=(0, 10))
        m.c = pyo.Constraint(expr=m.x + m.y <= 8)
        # obj=-2x-y: unique optimal at x+y=8 is x=8, y=0 (x has larger coefficient)
        m.obj = pyo.Objective(expr=-2 * m.x - m.y)
        _solve_and_check(
            self, self.opt, m, {'objective': -16.0, 'vars': [(m.x, 8.0), (m.y, 0.0)]}
        )
        m.y.fix(2.0)
        # With y=2 fixed: x<=6, objective=-2*6-2=-14. Unique.
        _solve_and_check(
            self, self.opt, m, {'objective': -14.0, 'vars': [(m.x, 6.0), (m.y, 2.0)]}
        )
        m.y.unfix()
        # After unfix: back to unique x=8, y=0.
        _solve_and_check(
            self, self.opt, m, {'objective': -16.0, 'vars': [(m.x, 8.0), (m.y, 0.0)]}
        )

    def test_remove_constraint_drops_mutable_helper(self):
        # After remove_constraints, the mutable helper for that row must be
        # dropped from _mutable_helpers (otherwise stale handle is dereferenced
        # on the next param update).
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
        # Should not raise even though the only mutable helper was removed.
        _solve_and_check(self, self.opt, m, {'objective': -10.0, 'vars': [(m.x, 10.0)]})

    def test_warmstart_column_indices_match_after_variable_removal(self):
        # _warmstart stores Python list-position j as the Xpress column index.
        # After _remove_variables, Xpress renumbers columns. The invariant
        # self._vars[j] <-> Xpress column j must hold; otherwise addMipSol
        # receives warm-start values in the wrong columns.
        m = pyo.ConcreteModel()
        m.x = pyo.Var(within=pyo.Binary)
        m.y = pyo.Var(within=pyo.Binary)
        m.z = pyo.Var(within=pyo.Binary)
        m.obj = pyo.Objective(expr=m.x + m.y + m.z)
        m.c = pyo.Constraint(expr=m.x + m.y + m.z >= 1)
        self.opt.set_instance(m)
        # Remove the middle variable -- this compacts self._vars and renumbers
        # Xpress columns. Afterwards self._vars = [x, z].
        self.opt.remove_variables([m.y])
        # Remove m.y from the Pyomo model so model.nvariables() reflects only the
        # two remaining variables (x and z). The _solve_and_check assertion
        # requires vars to cover all active Pyomo variables.
        del m.y
        # Verify that Python list position j equals the Xpress column index for
        # every remaining variable. If they diverge, _warmstart passes wrong columns.
        for j, var in enumerate(self.opt._vars):
            xp_idx = self.opt._maps.vars[id(var)].index
            self.assertEqual(
                j,
                xp_idx,
                f"After variable removal: Python list position {j} != "
                f"Xpress column index {xp_idx} for {var.name}",
            )
        # Set feasible warm-start hint values on the remaining variables and re-solve.
        # If column indices are wrong, addMipSol passes values to incorrect columns
        # which could produce a wrong solution or a solver error.
        m.x.set_value(1)
        m.z.set_value(0)
        # Remove the constraint that referenced m.y so the Pyomo model is consistent
        # with the reduced Xpress problem (y column deleted, constraint cleaned up).
        self.opt.remove_constraints([m.c])
        m.c.deactivate()
        _solve_and_check(
            self, self.opt, m, {'objective': 0.0, 'vars': [(m.x, 0.0), (m.z, 0.0)]}
        )


@unittest.pytest.mark.solver('xpress_persistent')
class TestXpressPersistentNLP(unittest.TestCase):
    """NLP integration tests for the persistent connector."""

    def setUp(self):
        self.opt = XpressPersistent()

    def _check_optimal(self, res):
        self.assertEqual(
            res.termination_condition, TerminationCondition.convergenceCriteriaSatisfied
        )

    def test_nl_add_constraint_registers_nl_rebuild(self):
        # NL constraint -> registered in _mutable_helpers with nl_expr set.
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
        # NL constraint with no mutable params: still registered in _mutable_helpers
        # (nl_expr set) because the NL row rebuild path is always available;
        # no mutable coef dicts populated since there are no mutable params.
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
        # sin(x) + p*y <= 5: the whole constraint is NL (full row rebuild on any
        # param change); p is stored in _lin_coefs and re-evaluated at rebuild time.
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
        # With p=1: sin(x) + y <= 5, maximize y -> y = 5 - sin(x) near 5
        m.p.set_value(2.0)
        _solve_and_check(
            self, self.opt, m, {'objective': -2.5, 'vars': [(m.x, 0.0), (m.y, 2.5)]}
        )
        y2 = pyo.value(m.y)
        # With p=2: 2y must leave room for sin(x), so y <= (5 - sin(x))/2 < y1
        self.assertLess(y2, y1)

    def test_nl_mutable_nl_coef_full_rebuild(self):
        # p*sin(x) <= 0.5: mutable param inside NL -> full row rebuild.
        # Bound x to [0, pi/2] so sin is strictly increasing and the constraint
        # x <= arcsin(0.5/p) is always the binding upper limit.
        # Objective maximize x (min -x) so the constraint is always active.
        # p=1: x <= arcsin(0.5) = pi/6 ~ 0.5236.  p=2: x <= arcsin(0.25) ~ 0.2527.
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
        # Larger p tightens the constraint: x2 should be noticeably smaller than x1
        self.assertLess(x2, x1 - 0.1)

    def test_nl_mutable_bound(self):
        # sin(x) >= p: mutable lower bound (NL row rebuild path when p changes).
        # x in [0, pi/2] so sin is strictly increasing; constraint is always binding.
        # p=0.5: x >= arcsin(0.5) = pi/6 ~ 0.5236.
        # p=0.9: x >= arcsin(0.9) ~ 1.1198.
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
        # Tighter lower bound means x must be larger
        self.assertGreater(x2, x1 + 0.4)

    def test_nl_solve_modify_resolve(self):
        # Solve NLP, change mutable bound param, re-solve.
        # exp(x) >= p is a pure-NL constraint; p change triggers a full NL row rebuild.
        # exp(x) >= p, min x: optimal is x = ln(p).
        m = pyo.ConcreteModel()
        m.p = pyo.Param(mutable=True, initialize=1.0)
        m.x = pyo.Var(bounds=(0, 3))
        m.c = pyo.Constraint(expr=pyo.exp(m.x) >= m.p)
        m.obj = pyo.Objective(expr=m.x)
        self.opt.set_instance(m)
        # exp(x) >= 1 -> x >= 0, minimize -> x=0
        _solve_check_mutate_check(
            self,
            self.opt,
            m,
            {'objective': 0.0, 'vars': [(m.x, 0.0)]},
            m.p,
            math.e,  # exp(x) >= e -> x >= 1, minimize -> x=1
            {'objective': 1.0, 'vars': [(m.x, 1.0)]},
        )

    def test_fix_variable_no_nl_constraint_rebuild(self):
        # Fixing a variable in an NL constraint must NOT rebuild the row --
        # Xpress handles fixed vars via bounds natively. Row count equality is
        # a weak proxy (del+add restores count). Use xp_con handle identity as
        # the definitive proof: if the handle survives, no del+add occurred.
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, math.pi))
        m.y = pyo.Var(bounds=(0, 10))
        m.c = pyo.Constraint(expr=pyo.sin(m.x) + m.y <= 5)
        m.obj = pyo.Objective(expr=m.y)
        self.opt.set_instance(m)
        # First solve: get internal state before fixing x.
        # sin(x)+y<=5, min y -> y=0.  x free, so x can be anything in [0,pi].
        _solve_and_check(
            self, self.opt, m, {'objective': 0.0, 'vars': [(m.x, 0.0), (m.y, 0.0)]}
        )
        xp_con_before = self.opt._mutable_helpers[m.c]._xp_con
        nrows_before = self.opt._xp_prob.attributes.rows
        m.x.fix(math.pi / 6)  # fix x at pi/6 -> sin(x)=0.5, constraint: 0.5+y<=5
        # Second solve: sin(pi/6)=0.5, constraint becomes 0.5+y<=5, min y -> y=0.
        _solve_and_check(
            self,
            self.opt,
            m,
            {'objective': 0.0, 'vars': [(m.x, math.pi / 6), (m.y, 0.0)]},
        )
        nrows_after = self.opt._xp_prob.attributes.rows
        self.assertEqual(nrows_before, nrows_after)
        # Handle identity: same xp.constraint object means no del+add occurred
        self.assertIs(self.opt._mutable_helpers[m.c]._xp_con, xp_con_before)

    def _run_nl_linear_shared_param_test(self, nl_first: bool):
        """Helper: same param in NL (rebuild) and linear (chgMCoef) constraints.
        nl_first=True  -> NL at row 0, linear at row 1 (delConstraint(NL) shifts linear).
        nl_first=False -> linear at row 0, NL at row 1 (no shift on linear).
        Both orderings must produce the correct solution after param change.

        The NL constraint IS binding at the optimum (p*sin(x) <= 0.5 limits x).
        This is necessary so that a silently dropped NL rebuild would be detectable.
        Objective max x+y so both the NL and linear constraints contribute to the result.
        p=1: sin(x)<=0.5->x<=pi/6~0.524, y<=4 -> x+y~4.524.
        p=2: 2*sin(x)<=0.5->x<=arcsin(0.25)~0.253, y<=2 -> x+y~2.253."""
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
        # NL at row 0, linear at row 1: delConstraint(NL) shifts linear's row.
        # The single-pass bug would corrupt the linear constraint update here.
        self._run_nl_linear_shared_param_test(nl_first=True)

    def test_param_shared_linear_before_nl(self):
        # Linear at row 0, NL at row 1: no row shift on linear after NL deletion.
        self._run_nl_linear_shared_param_test(nl_first=False)

    def test_param_shared_multiple_nl_and_linear(self):
        # Multiple NL constraints interleaved with linear ones.
        # Each NL rebuild (delConstraint) can shift remaining row indices.
        m = pyo.ConcreteModel()
        m.p = pyo.Param(mutable=True, initialize=1.0)
        m.x = pyo.Var(bounds=(0, math.pi))
        m.y = pyo.Var(bounds=(0, 10))
        m.z = pyo.Var(bounds=(0, 10))
        # Declare in interleaved order: NL, linear, NL, linear
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
        # Mutable quadratic coef (has_mutable_quadratic -> _nl_rebuild_cons) at
        # a lower row than a linear mutable constraint. Same row-shift risk as NL.
        m = pyo.ConcreteModel()
        m.p = pyo.Param(mutable=True, initialize=1.0)
        m.x = pyo.Var(bounds=(0, 5))
        m.y = pyo.Var(bounds=(0, 10))
        m.c_quad = pyo.Constraint(
            expr=m.p * m.x**2 <= 5
        )  # mutable_quadratic -> NL rebuild
        m.c_lin = pyo.Constraint(expr=m.p * m.y <= 4)  # mutable_lin -> chgMCoef
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
        # sin(p*x) + q*y <= 5: p is inside the NL formula (-> full rebuild on
        # p change), q is a mutable linear coef stored in _mutable_helpers[con]._lin_coefs
        # (re-evaluated at rebuild time via _make_xp_con). Constraint is in _mutable_helpers.
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
        # NL rebuild registered; mutable lin coef (q) stored in same helper.
        self.assertIn(m.c, self.opt._mutable_helpers)
        self.assertIsNotNone(self.opt._mutable_helpers[m.c]._nl_expr)
        pyo.value(m.y)

        # Change affine param q only: constraint updated, solution changes
        m.q.set_value(2.0)
        # q=2, p=1: sin(1*x) + 2*y <= 5. At x~0, y <= 5/2 = 2.5.
        _solve_and_check(
            self, self.opt, m, {'objective': -2.5, 'vars': [(m.x, 0.0), (m.y, 2.5)]}
        )
        pyo.value(m.y)

        # Change NL formula param p: full rebuild, solution changes
        m.p.set_value(0.0)  # sin(0)=0, so constraint becomes q*y <= 5, y <= 2.5
        _solve_and_check(
            self, self.opt, m, {'objective': -2.5, 'vars': [(m.x, 0.0), (m.y, 2.5)]}
        )
        pyo.value(m.y)

    def test_mutable_objective_does_not_interfere_with_linear_update(self):
        # Mutable param in objective AND in a linear constraint.
        # setObjective does not shift constraint row indices, so the linear
        # update must be applied correctly regardless of objective rebuild.
        m = pyo.ConcreteModel()
        m.p = pyo.Param(mutable=True, initialize=1.0)
        m.x = pyo.Var(bounds=(0, 3))
        m.y = pyo.Var(bounds=(0, 10))
        m.c_nl = pyo.Constraint(expr=m.p * pyo.sin(m.x) <= 2)  # NL rebuild
        m.c_lin = pyo.Constraint(expr=m.p * m.y <= 4)  # chgMCoef
        m.obj = pyo.Objective(expr=m.p * m.y, sense=pyo.maximize)  # mutable obj
        self.opt.set_instance(m)
        _solve_and_check(
            self, self.opt, m, {'objective': 4.0, 'vars': [(m.x, 1.5), (m.y, 4.0)]}
        )

        m.p.set_value(2.0)
        # p=2: y<=2, obj = 2*y = 4.
        _solve_and_check(
            self, self.opt, m, {'objective': 4.0, 'vars': [(m.x, 1.5), (m.y, 2.0)]}
        )

    def test_nl_mutable_objective_persistent(self):
        # Coverage gap: xpress_persistent.py lines 287-295 (_MutableObjective NL rebuild).
        # Objective = p*sin(x), mutable p inside NL formula -> full NL objective rebuild.
        m = pyo.ConcreteModel()
        m.p = pyo.Param(mutable=True, initialize=1.0)
        m.x = pyo.Var(bounds=(0, math.pi / 2))
        m.obj = pyo.Objective(expr=m.p * pyo.sin(m.x), sense=pyo.maximize)
        # p=1, max sin(x) on [0, pi/2] -> x = pi/2; sense=maximize -> obj = sin(pi/2) = 1.0
        _solve_and_check(
            self, self.opt, m, {'objective': 1.0, 'vars': [(m.x, math.pi / 2)]}
        )
        pyo.value(m.x)

        m.p.set_value(-1.0)  # now minimizing sin(x) -> x = 0
        _solve_and_check(self, self.opt, m, {'objective': 0.0, 'vars': [(m.x, 0.0)]})
        pyo.value(m.x)

    def test_nl_objective_stable_xp_not_mutated_by_constant_update(self):
        # Regression test for a mutation bug in _MutableObjective.update().
        #
        # Trigger conditions:
        #   - NL objective
        #   - stable (non-mutable) linear terms  -> stable_xp is xp.expression
        #   - no mutable linear/quad terms       -> mut_terms stays empty
        #   - mutable constant in repn           -> _constant is set
        #
        # Model: obj = 2*x + p + sin(z)
        #   2*x  : stable linear term (non-mutable coef) -> goes into stable_xp
        #   p    : mutable param, no variable -> becomes repn.constant -> _constant
        #   sin(z): NL part -> nl_expr, mut_terms is empty
        import math
        m = pyo.ConcreteModel()
        m.p = pyo.Param(mutable=True, initialize=1.0)
        m.x = pyo.Var(bounds=(0, 1))
        m.z = pyo.Var(bounds=(0, math.pi / 2))
        # min 2*x + p + sin(z): with x,z in bounds, optimum is x=0, z=0.
        # obj_value = 2*0 + p + sin(0) = p
        m.obj = pyo.Objective(expr=2 * m.x + m.p + pyo.sin(m.z))
        _solve_and_check(
            self, self.opt, m,
            {'objective': 1.0, 'vars': [(m.x, 0.0), (m.z, 0.0)]},
        )

        # After the first update p=3: obj = 2*0 + 3 + sin(0) = 3.
        # With the bug, stable_xp has been mutated to (2*x + 1), so the next
        # update would produce (2*x + 1 + 3 + sin(z)) = 4 instead of 3.
        m.p.set_value(3.0)
        _solve_and_check(
            self, self.opt, m,
            {'objective': 3.0, 'vars': [(m.x, 0.0), (m.z, 0.0)]},
        )

        # A third solve with p=5 would give 7 with the bug (accumulated 1+3+5-2 offset)
        # but should give 5.
        m.p.set_value(5.0)
        _solve_and_check(
            self, self.opt, m,
            {'objective': 5.0, 'vars': [(m.x, 0.0), (m.z, 0.0)]},
        )

    def test_nl_constraint_stable_quadratic_term(self):
        # Coverage gap: xpress_persistent.py line 619 (stable quadratic in NL constraint).
        # 2*x^2 is a non-mutable quadratic term; it goes into stable_xp at registration.
        # p*exp(y) is the NL mutable term that triggers row rebuilds.
        # Objective max x (min -x) so the constraint is always the binding limit.
        # p=1: 2*x^2 + exp(0) <= 5 -> x <= sqrt(2) ~ 1.414.
        # p=2: 2*x^2 + 2*exp(0) <= 5 -> x <= sqrt(1.5) ~ 1.225.
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

        # Change mutable param: stable quad term must survive the NL rebuild.
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
        # Coverage gap: xpress_persistent.py lines 715/726 (stable lin/quad in NL objective).
        # 2*x is stable linear, y^2 is stable quadratic, p*sin(z) is NL with mutable p.
        m = pyo.ConcreteModel()
        m.p = pyo.Param(mutable=True, initialize=1.0)
        m.x = pyo.Var(bounds=(0, 1))
        m.y = pyo.Var(bounds=(0, 1))
        m.z = pyo.Var(bounds=(0, math.pi / 2))
        # objective = 2*x + y^2 + p*sin(z): stable lin (2x), stable quad (y^2), NL (p*sin)
        m.obj = pyo.Objective(expr=2 * m.x + m.y**2 + m.p * pyo.sin(m.z))
        # Minimizing: x=0 (coef 2>0), y=0 (convex quad), z=0 (p=1>0, sin>=0) -> obj=0
        _solve_and_check(
            self,
            self.opt,
            m,
            {'objective': 0.0, 'vars': [(m.x, 0.0), (m.y, 0.0), (m.z, 0.0)]},
        )

        # Change mutable param (triggers NL objective rebuild)
        m.p.set_value(-1.0)  # now -sin(z) is negative -> want max z
        # Minimizing 2x + y^2 - sin(z): x=0, y=0, z=pi/2 -> obj = -1
        _solve_and_check(
            self,
            self.opt,
            m,
            {'objective': -1.0, 'vars': [(m.x, 0.0), (m.y, 0.0), (m.z, math.pi / 2)]},
        )

    def test_nl_cubic_constraint_mutable_param_persistent(self):
        # p*x^3 >= 1, min x, x in [0,10].
        # p=1: x^3 >= 1 -> x >= 1 -> optimal x=1.0
        # p=8: 8*x^3 >= 1 -> x^3 >= 1/8 -> x >= 0.5 -> optimal x=0.5
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
        # Start with linear at row 0, NL at row 1 (safe ordering -- no shift).
        # Remove linear: NL shifts to row 0.
        # Re-add linear: NL=0, linear=1 (now the dangerous ordering).
        # A param change must still correctly update the linear constraint.
        m = pyo.ConcreteModel()
        m.p = pyo.Param(mutable=True, initialize=1.0)
        m.x = pyo.Var(bounds=(0, math.pi))
        m.y = pyo.Var(bounds=(0, 10))
        m.c_lin = pyo.Constraint(expr=m.p * m.y <= 4)  # row 0 initially
        m.c_nl = pyo.Constraint(expr=m.p * pyo.sin(m.x) <= 5)  # row 1 initially
        m.obj = pyo.Objective(expr=m.y, sense=pyo.maximize)
        self.opt.set_instance(m)
        _solve_and_check(
            self,
            self.opt,
            m,
            {'objective': 4.0, 'vars': [(m.x, math.pi / 2), (m.y, 4.0)]},
        )

        # Remove c_lin (row 0): c_nl shifts to row 0.
        # Correct Pyomo API ordering: deactivate first, then remove from solver.
        m.c_lin.deactivate()
        self.opt.remove_constraints([m.c_lin])
        # Re-add c_lin: now c_nl=row 0, c_lin=row 1 (dangerous ordering).
        m.c_lin.activate()
        self.opt.add_constraints([m.c_lin])
        # c_lin must be re-registered in _mutable_helpers after re-add.
        self.assertIn(m.c_lin, self.opt._mutable_helpers)

        m.p.set_value(2.0)
        # p=2: c_lin becomes p*y<=4 -> y<=2
        _solve_and_check(
            self,
            self.opt,
            m,
            {'objective': 2.0, 'vars': [(m.x, math.pi / 2), (m.y, 2.0)]},
        )


@unittest.pytest.mark.solver('xpress_persistent')
class TestXpressPersistentIIS(unittest.TestCase):
    """Tests for IIS computation via write_iis() and get_iis()."""

    def setUp(self):
        self.opt = XpressPersistent()

    def _infeasible_model(self):
        # x in {0,1}, y >= 0
        # c1: y <= 100*x   c2: y <= -100*x   c3: x >= 0.5
        # c2 and c3 together with x binary force infeasibility.
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
            # The two infeasible constraints must appear in the IIS file.
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
        # c2 and c3 are the infeasible pair; c1 may or may not be included.
        self.assertIn('c2', con_names)
        self.assertIn('c3', con_names)
        # y's lower bound (0) contributes to the IIS.
        var_names = {v.name for v in iis['variables']}
        self.assertIn('y', var_names)

    def test_get_iis_objects_are_model_constraints(self):
        # Verify the returned ConstraintData and VarData objects are the
        # actual Pyomo components, not copies or proxies (identity check).
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
