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
import logging

import pyomo.common.unittest as unittest
import pyomo.environ as pyo
import pyomo.solvers.plugins.solvers.xpress_direct as xpd

from pyomo.common.log import LoggingIntercept
from pyomo.core.expr.taylor_series import taylor_series_expansion
from pyomo.opt.results.solver import TerminationCondition, SolverStatus
from pyomo.solvers.plugins.solvers.xpress_persistent import XpressPersistent

xpress_available = pyo.SolverFactory('xpress_persistent').available(False)


class TestXpressPersistent(unittest.TestCase):
    @unittest.skipIf(not xpress_available, "xpress is not available")
    def test_basics(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(-10, 10))
        m.y = pyo.Var()
        m.obj = pyo.Objective(expr=m.x**2 + m.y**2)
        m.c1 = pyo.Constraint(expr=m.y >= 2 * m.x + 1)

        opt = pyo.SolverFactory('xpress_persistent')
        opt.set_instance(m)

        self.assertEqual(opt.get_xpress_attribute('cols'), 2)
        self.assertEqual(opt.get_xpress_attribute('rows'), 1)

        res = opt.solve()
        self.assertAlmostEqual(m.x.value, -0.4, delta=1e-6)
        self.assertAlmostEqual(m.y.value, 0.2, delta=1e-6)

        opt.load_duals()
        self.assertEqual(len(m.dual), 1)
        self.assertAlmostEqual(m.dual[m.c1], -0.4, delta=1e-6)
        del m.dual

        opt.load_rc()
        self.assertEqual(len(m.rc), 2)
        self.assertAlmostEqual(m.rc[m.x], 0, delta=1e-8)
        self.assertAlmostEqual(m.rc[m.y], 0, delta=1e-8)
        del m.rc

        opt.load_slacks()
        self.assertEqual(len(m.slack), 1)
        self.assertAlmostEqual(m.slack[m.c1], 0, delta=1e-6)
        del m.slack

        m.c2 = pyo.Constraint(expr=m.y >= -m.x + 1)
        opt.add_constraint(m.c2)
        self.assertEqual(opt.get_xpress_attribute('cols'), 2)
        self.assertEqual(opt.get_xpress_attribute('rows'), 2)

        res = opt.solve(save_results=False, load_solutions=False)
        self.assertAlmostEqual(m.x.value, -0.4, delta=1e-6)
        self.assertAlmostEqual(m.y.value, 0.2, delta=1e-6)
        opt.load_vars()
        self.assertAlmostEqual(m.x.value, 0, delta=2.5e-6)
        self.assertAlmostEqual(m.y.value, 1, delta=2.5e-6)

        opt.remove_constraint(m.c2)
        m.del_component(m.c2)
        self.assertEqual(opt.get_xpress_attribute('cols'), 2)
        self.assertEqual(opt.get_xpress_attribute('rows'), 1)

        self.assertEqual(opt.get_xpress_control('feastol'), 1e-6)
        res = opt.solve(options={'feastol': '1e-7'})
        self.assertEqual(opt.get_xpress_control('feastol'), 1e-7)
        self.assertAlmostEqual(m.x.value, -0.4, delta=1e-6)
        self.assertAlmostEqual(m.y.value, 0.2, delta=1e-6)

        m.x.setlb(-5)
        m.x.setub(5)
        opt.update_var(m.x)
        # a nice wrapper for xpress isn't implemented,
        # so we'll do this directly
        x_idx = opt._solver_model.getIndex(opt._pyomo_var_to_solver_var_map[m.x])
        lb = []
        opt._solver_model.getlb(lb, x_idx, x_idx)
        ub = []
        opt._solver_model.getub(ub, x_idx, x_idx)
        self.assertEqual(lb[0], -5)
        self.assertEqual(ub[0], 5)

        m.x.fix(0)
        opt.update_var(m.x)
        lb = []
        opt._solver_model.getlb(lb, x_idx, x_idx)
        ub = []
        opt._solver_model.getub(ub, x_idx, x_idx)
        self.assertEqual(lb[0], 0)
        self.assertEqual(ub[0], 0)

        m.x.unfix()
        opt.update_var(m.x)
        lb = []
        opt._solver_model.getlb(lb, x_idx, x_idx)
        ub = []
        opt._solver_model.getub(ub, x_idx, x_idx)
        self.assertEqual(lb[0], -5)
        self.assertEqual(ub[0], 5)

        m.c2 = pyo.Constraint(expr=m.y >= m.x**2)
        opt.add_constraint(m.c2)
        self.assertEqual(opt.get_xpress_attribute('cols'), 2)
        self.assertEqual(opt.get_xpress_attribute('rows'), 2)

        opt.remove_constraint(m.c2)
        m.del_component(m.c2)
        self.assertEqual(opt.get_xpress_attribute('cols'), 2)
        self.assertEqual(opt.get_xpress_attribute('rows'), 1)

        m.z = pyo.Var()
        opt.add_var(m.z)
        self.assertEqual(opt.get_xpress_attribute('cols'), 3)
        opt.remove_var(m.z)
        del m.z
        self.assertEqual(opt.get_xpress_attribute('cols'), 2)

    @unittest.skipIf(not xpress_available, "xpress is not available")
    def test_vartype_change(self):
        # test for issue #3565
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 1))
        m.o = pyo.Objective(expr=m.x)

        opt = pyo.SolverFactory('xpress_persistent')
        opt.set_instance(m)

        m.x.fix(1)
        opt.update_var(m.x)

        x_idx = opt._solver_model.getIndex(opt._pyomo_var_to_solver_var_map[m.x])
        lb = []
        opt._solver_model.getlb(lb, x_idx, x_idx)
        self.assertEqual(lb[0], 1)

        m.x.domain = pyo.Binary
        opt.update_var(m.x)

        lb = []
        opt._solver_model.getlb(lb, x_idx, x_idx)
        self.assertEqual(lb[0], 1)

    @unittest.skipIf(not xpress_available, "xpress is not available")
    def test_add_remove_qconstraint(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.z = pyo.Var()
        m.obj = pyo.Objective(expr=m.z)
        m.c1 = pyo.Constraint(expr=m.z >= m.x**2 + m.y**2)

        opt = pyo.SolverFactory('xpress_persistent')
        opt.set_instance(m)
        self.assertEqual(opt.get_xpress_attribute('rows'), 1)

        opt.remove_constraint(m.c1)
        self.assertEqual(opt.get_xpress_attribute('rows'), 0)

        opt.add_constraint(m.c1)
        self.assertEqual(opt.get_xpress_attribute('rows'), 1)

    @unittest.skipIf(not xpress_available, "xpress is not available")
    def test_add_remove_lconstraint(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.z = pyo.Var()
        m.obj = pyo.Objective(expr=m.z)
        m.c2 = pyo.Constraint(expr=m.x + m.y == 1)

        opt = pyo.SolverFactory('xpress_persistent')
        opt.set_instance(m)
        self.assertEqual(opt.get_xpress_attribute('rows'), 1)

        opt.remove_constraint(m.c2)
        self.assertEqual(opt.get_xpress_attribute('rows'), 0)

        opt.add_constraint(m.c2)
        self.assertEqual(opt.get_xpress_attribute('rows'), 1)

    @unittest.skipIf(not xpress_available, "xpress is not available")
    def test_add_remove_sosconstraint(self):
        m = pyo.ConcreteModel()
        m.a = pyo.Set(initialize=[1, 2, 3], ordered=True)
        m.x = pyo.Var(m.a, within=pyo.Binary)
        m.y = pyo.Var(within=pyo.Binary)
        m.obj = pyo.Objective(expr=m.y)
        m.c1 = pyo.SOSConstraint(var=m.x, sos=1)

        opt = pyo.SolverFactory('xpress_persistent')
        opt.set_instance(m)
        self.assertEqual(opt.get_xpress_attribute('sets'), 1)

        opt.remove_sos_constraint(m.c1)
        self.assertEqual(opt.get_xpress_attribute('sets'), 0)

        opt.add_sos_constraint(m.c1)
        self.assertEqual(opt.get_xpress_attribute('sets'), 1)

    @unittest.skipIf(not xpress_available, "xpress is not available")
    def test_add_remove_sosconstraint2(self):
        m = pyo.ConcreteModel()
        m.a = pyo.Set(initialize=[1, 2, 3], ordered=True)
        m.x = pyo.Var(m.a, within=pyo.Binary)
        m.y = pyo.Var(within=pyo.Binary)
        m.obj = pyo.Objective(expr=m.y)
        m.c1 = pyo.SOSConstraint(var=m.x, sos=1)

        opt = pyo.SolverFactory('xpress_persistent')
        opt.set_instance(m)
        self.assertEqual(opt.get_xpress_attribute('sets'), 1)
        m.c2 = pyo.SOSConstraint(var=m.x, sos=2)
        opt.add_sos_constraint(m.c2)
        self.assertEqual(opt.get_xpress_attribute('sets'), 2)
        opt.remove_sos_constraint(m.c2)
        self.assertEqual(opt.get_xpress_attribute('sets'), 1)

    @unittest.skipIf(not xpress_available, "xpress is not available")
    def test_add_remove_var(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()

        opt = pyo.SolverFactory('xpress_persistent')
        opt.set_instance(m)
        self.assertEqual(opt.get_xpress_attribute('cols'), 2)

        opt.remove_var(m.x)
        self.assertEqual(opt.get_xpress_attribute('cols'), 1)

        opt.add_var(m.x)
        self.assertEqual(opt.get_xpress_attribute('cols'), 2)

        opt.remove_var(m.x)
        opt.add_var(m.x)
        opt.remove_var(m.x)
        self.assertEqual(opt.get_xpress_attribute('cols'), 1)

    @unittest.skipIf(not xpress_available, "xpress is not available")
    def test_add_column(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(within=pyo.NonNegativeReals)
        m.c = pyo.Constraint(expr=(0, m.x, 1))
        m.obj = pyo.Objective(expr=-m.x)

        opt = pyo.SolverFactory('xpress_persistent')
        opt.set_instance(m)
        opt.solve()
        self.assertAlmostEqual(m.x.value, 1)

        m.y = pyo.Var(within=pyo.NonNegativeReals)

        opt.add_column(m, m.y, -3, [m.c], [2])
        opt.solve()

        self.assertAlmostEqual(m.x.value, 0)
        self.assertAlmostEqual(m.y.value, 0.5)

    @unittest.skipIf(not xpress_available, "xpress is not available")
    def test_add_column_exceptions(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.c = pyo.Constraint(expr=(0, m.x, 1))
        m.ci = pyo.Constraint([1, 2], rule=lambda m, i: (0, m.x, i + 1))
        m.cd = pyo.Constraint(expr=(0, -m.x, 1))
        m.cd.deactivate()
        m.obj = pyo.Objective(expr=-m.x)

        opt = pyo.SolverFactory('xpress_persistent')

        # set_instance not called
        self.assertRaises(RuntimeError, opt.add_column, m, m.x, 0, [m.c], [1])

        opt.set_instance(m)

        m2 = pyo.ConcreteModel()
        m2.y = pyo.Var()
        m2.c = pyo.Constraint(expr=(0, m.x, 1))

        # different model than attached to opt
        self.assertRaises(RuntimeError, opt.add_column, m2, m2.y, 0, [], [])
        # pyomo var attached to different model
        self.assertRaises(RuntimeError, opt.add_column, m, m2.y, 0, [], [])

        z = pyo.Var()
        # pyomo var floating
        self.assertRaises(RuntimeError, opt.add_column, m, z, -2, [m.c, z], [1])

        m.y = pyo.Var()
        # len(coefficients) == len(constraints)
        self.assertRaises(RuntimeError, opt.add_column, m, m.y, -2, [m.c], [1, 2])
        self.assertRaises(RuntimeError, opt.add_column, m, m.y, -2, [m.c, z], [1])

        # add indexed constraint
        self.assertRaises(AttributeError, opt.add_column, m, m.y, -2, [m.ci], [1])
        # add something not a ConstraintData
        self.assertRaises(AttributeError, opt.add_column, m, m.y, -2, [m.x], [1])

        # constraint not on solver model
        self.assertRaises(KeyError, opt.add_column, m, m.y, -2, [m2.c], [1])

        # inactive constraint
        self.assertRaises(KeyError, opt.add_column, m, m.y, -2, [m.cd], [1])

        opt.add_var(m.y)
        # var already in solver model
        self.assertRaises(RuntimeError, opt.add_column, m, m.y, -2, [m.c], [1])

    @unittest.skipIf(not xpress_available, "xpress is not available")
    def test_nonconvexqp_locally_optimal(self):
        """Test non-convex QP for which xpress_direct should find a locally
        optimal solution."""
        m = pyo.ConcreteModel()
        m.x1 = pyo.Var()
        m.x2 = pyo.Var()
        m.x3 = pyo.Var()

        m.obj = pyo.Objective(rule=lambda m: 2 * m.x1 + m.x2 + m.x3, sense=pyo.minimize)
        m.equ1 = pyo.Constraint(rule=lambda m: m.x1 + m.x2 + m.x3 == 1)
        m.cone = pyo.Constraint(rule=lambda m: m.x2 * m.x2 + m.x3 * m.x3 <= m.x1 * m.x1)
        m.equ2 = pyo.Constraint(rule=lambda m: m.x1 >= 0)

        opt = pyo.SolverFactory('xpress_direct')
        opt.options['XSLP_SOLVER'] = 0
        # xpress 9.5.0 now defaults to trying (and failing) to solve this problem
        # using the global solver. This option forces the use of the local solver.
        opt.options['NLPSOLVER'] = 1

        results = opt.solve(m)
        self.assertEqual(results.solver.status, SolverStatus.ok)
        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.locallyOptimal
        )

        # Cannot test exact values since the may be different depending on
        # random effects. So just test all are non-zero.
        self.assertGreater(m.x1.value, 0.0)
        self.assertGreater(m.x2.value, 0.0)
        self.assertGreater(m.x3.value, 0.0)

    @unittest.skipIf(not xpress_available, "xpress is not available")
    def test_nonconvexqp_infeasible(self):
        """Test non-convex QP which xpress_direct should prove infeasible."""
        m = pyo.ConcreteModel()
        m.x1 = pyo.Var()
        m.x2 = pyo.Var()
        m.x3 = pyo.Var()

        m.obj = pyo.Objective(rule=lambda m: 2 * m.x1 + m.x2 + m.x3, sense=pyo.minimize)
        m.equ1a = pyo.Constraint(rule=lambda m: m.x1 + m.x2 + m.x3 == 1)
        m.equ1b = pyo.Constraint(rule=lambda m: m.x1 + m.x2 + m.x3 == -1)
        m.cone = pyo.Constraint(rule=lambda m: m.x2 * m.x2 + m.x3 * m.x3 <= m.x1 * m.x1)
        m.equ2 = pyo.Constraint(rule=lambda m: m.x1 >= 0)

        opt = pyo.SolverFactory('xpress_direct')
        opt.options['XSLP_SOLVER'] = 0

        results = opt.solve(m)
        self.assertEqual(results.solver.status, SolverStatus.ok)
        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.infeasible
        )

    def test_available(self):
        class mock_xpress(object):
            def __init__(self, importable, initable):
                self._initable = initable
                xpd.xpress_available = importable

            def log_import_warning(self, logger):
                logging.getLogger(logger).warning("import warning")

            def init(self):
                if not self._initable:
                    raise RuntimeError("init failed")

            def free(self):
                pass

        orig = xpd.xpress, xpd.xpress_available
        try:
            _xpress_persistent = XpressPersistent
            xpd.xpress = mock_xpress(True, True)
            with LoggingIntercept() as LOG:
                self.assertTrue(XpressPersistent().available(True))
                self.assertTrue(XpressPersistent().available(False))
            self.assertEqual(LOG.getvalue(), "")

            xpd.xpress = mock_xpress(False, False)
            with LoggingIntercept() as LOG:
                self.assertFalse(XpressPersistent().available(False))
            self.assertEqual(LOG.getvalue(), "")
            with LoggingIntercept() as LOG:
                with self.assertRaisesRegex(
                    xpd.ApplicationError,
                    "No Python bindings available for .*XpressPersistent.* "
                    "solver plugin",
                ):
                    XpressPersistent().available(True)
            self.assertEqual(LOG.getvalue(), "import warning\n")

            xpd.xpress = mock_xpress(True, False)
            with LoggingIntercept() as LOG:
                self.assertFalse(XpressPersistent().available(False))
            self.assertEqual(LOG.getvalue(), "")
            with LoggingIntercept() as LOG:
                with self.assertRaisesRegex(RuntimeError, "init failed"):
                    XpressPersistent().available(True)
            self.assertEqual(LOG.getvalue(), "")
        finally:
            xpd.xpress, xpd.xpress_available = orig
