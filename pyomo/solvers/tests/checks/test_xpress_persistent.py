import pyomo.common.unittest as unittest
import pyomo.environ as pe
from pyomo.core.expr.taylor_series import taylor_series_expansion
from pyomo.solvers.plugins.solvers.xpress_direct import xpress_available
from pyomo.opt.results.solver import TerminationCondition, SolverStatus


class TestXpressPersistent(unittest.TestCase):
    @unittest.skipIf(not xpress_available, "xpress is not available")
    def test_basics(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(-10, 10))
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.x**2 + m.y**2)
        m.c1 = pe.Constraint(expr=m.y >= 2 * m.x + 1)

        opt = pe.SolverFactory('xpress_persistent')
        opt.set_instance(m)

        self.assertEqual(opt.get_xpress_attribute('cols'), 2)
        self.assertEqual(opt.get_xpress_attribute('rows'), 1)

        res = opt.solve()
        self.assertAlmostEqual(m.x.value, -0.4, delta=1e-6)
        self.assertAlmostEqual(m.y.value, 0.2, delta=1e-6)
        opt.load_duals()
        self.assertAlmostEqual(m.dual[m.c1], -0.4, delta=1e-6)
        del m.dual

        m.c2 = pe.Constraint(expr=m.y >= -m.x + 1)
        opt.add_constraint(m.c2)
        self.assertEqual(opt.get_xpress_attribute('cols'), 2)
        self.assertEqual(opt.get_xpress_attribute('rows'), 2)

        res = opt.solve(save_results=False, load_solutions=False)
        self.assertAlmostEqual(m.x.value, -0.4, delta=1e-6)
        self.assertAlmostEqual(m.y.value, 0.2, delta=1e-6)
        opt.load_vars()
        self.assertAlmostEqual(m.x.value, 0, delta=1e-6)
        self.assertAlmostEqual(m.y.value, 1, delta=2e-6)

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

        m.c2 = pe.Constraint(expr=m.y >= m.x**2)
        opt.add_constraint(m.c2)
        self.assertEqual(opt.get_xpress_attribute('cols'), 2)
        self.assertEqual(opt.get_xpress_attribute('rows'), 2)

        opt.remove_constraint(m.c2)
        m.del_component(m.c2)
        self.assertEqual(opt.get_xpress_attribute('cols'), 2)
        self.assertEqual(opt.get_xpress_attribute('rows'), 1)

        m.z = pe.Var()
        opt.add_var(m.z)
        self.assertEqual(opt.get_xpress_attribute('cols'), 3)
        opt.remove_var(m.z)
        del m.z
        self.assertEqual(opt.get_xpress_attribute('cols'), 2)

    @unittest.skipIf(not xpress_available, "xpress is not available")
    def test_add_remove_qconstraint(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.z = pe.Var()
        m.obj = pe.Objective(expr=m.z)
        m.c1 = pe.Constraint(expr=m.z >= m.x**2 + m.y**2)

        opt = pe.SolverFactory('xpress_persistent')
        opt.set_instance(m)
        self.assertEqual(opt.get_xpress_attribute('rows'), 1)

        opt.remove_constraint(m.c1)
        self.assertEqual(opt.get_xpress_attribute('rows'), 0)

        opt.add_constraint(m.c1)
        self.assertEqual(opt.get_xpress_attribute('rows'), 1)

    @unittest.skipIf(not xpress_available, "xpress is not available")
    def test_add_remove_lconstraint(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.z = pe.Var()
        m.obj = pe.Objective(expr=m.z)
        m.c2 = pe.Constraint(expr=m.x + m.y == 1)

        opt = pe.SolverFactory('xpress_persistent')
        opt.set_instance(m)
        self.assertEqual(opt.get_xpress_attribute('rows'), 1)

        opt.remove_constraint(m.c2)
        self.assertEqual(opt.get_xpress_attribute('rows'), 0)

        opt.add_constraint(m.c2)
        self.assertEqual(opt.get_xpress_attribute('rows'), 1)

    @unittest.skipIf(not xpress_available, "xpress is not available")
    def test_add_remove_sosconstraint(self):
        m = pe.ConcreteModel()
        m.a = pe.Set(initialize=[1, 2, 3], ordered=True)
        m.x = pe.Var(m.a, within=pe.Binary)
        m.y = pe.Var(within=pe.Binary)
        m.obj = pe.Objective(expr=m.y)
        m.c1 = pe.SOSConstraint(var=m.x, sos=1)

        opt = pe.SolverFactory('xpress_persistent')
        opt.set_instance(m)
        self.assertEqual(opt.get_xpress_attribute('sets'), 1)

        opt.remove_sos_constraint(m.c1)
        self.assertEqual(opt.get_xpress_attribute('sets'), 0)

        opt.add_sos_constraint(m.c1)
        self.assertEqual(opt.get_xpress_attribute('sets'), 1)

    @unittest.skipIf(not xpress_available, "xpress is not available")
    def test_add_remove_sosconstraint2(self):
        m = pe.ConcreteModel()
        m.a = pe.Set(initialize=[1, 2, 3], ordered=True)
        m.x = pe.Var(m.a, within=pe.Binary)
        m.y = pe.Var(within=pe.Binary)
        m.obj = pe.Objective(expr=m.y)
        m.c1 = pe.SOSConstraint(var=m.x, sos=1)

        opt = pe.SolverFactory('xpress_persistent')
        opt.set_instance(m)
        self.assertEqual(opt.get_xpress_attribute('sets'), 1)
        m.c2 = pe.SOSConstraint(var=m.x, sos=2)
        opt.add_sos_constraint(m.c2)
        self.assertEqual(opt.get_xpress_attribute('sets'), 2)
        opt.remove_sos_constraint(m.c2)
        self.assertEqual(opt.get_xpress_attribute('sets'), 1)

    @unittest.skipIf(not xpress_available, "xpress is not available")
    def test_add_remove_var(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()

        opt = pe.SolverFactory('xpress_persistent')
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
        m = pe.ConcreteModel()
        m.x = pe.Var(within=pe.NonNegativeReals)
        m.c = pe.Constraint(expr=(0, m.x, 1))
        m.obj = pe.Objective(expr=-m.x)

        opt = pe.SolverFactory('xpress_persistent')
        opt.set_instance(m)
        opt.solve()
        self.assertAlmostEqual(m.x.value, 1)

        m.y = pe.Var(within=pe.NonNegativeReals)

        opt.add_column(m, m.y, -3, [m.c], [2])
        opt.solve()

        self.assertAlmostEqual(m.x.value, 0)
        self.assertAlmostEqual(m.y.value, 0.5)

    @unittest.skipIf(not xpress_available, "xpress is not available")
    def test_add_column_exceptions(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.c = pe.Constraint(expr=(0, m.x, 1))
        m.ci = pe.Constraint([1, 2], rule=lambda m, i: (0, m.x, i + 1))
        m.cd = pe.Constraint(expr=(0, -m.x, 1))
        m.cd.deactivate()
        m.obj = pe.Objective(expr=-m.x)

        opt = pe.SolverFactory('xpress_persistent')

        # set_instance not called
        self.assertRaises(RuntimeError, opt.add_column, m, m.x, 0, [m.c], [1])

        opt.set_instance(m)

        m2 = pe.ConcreteModel()
        m2.y = pe.Var()
        m2.c = pe.Constraint(expr=(0, m.x, 1))

        # different model than attached to opt
        self.assertRaises(RuntimeError, opt.add_column, m2, m2.y, 0, [], [])
        # pyomo var attached to different model
        self.assertRaises(RuntimeError, opt.add_column, m, m2.y, 0, [], [])

        z = pe.Var()
        # pyomo var floating
        self.assertRaises(RuntimeError, opt.add_column, m, z, -2, [m.c, z], [1])

        m.y = pe.Var()
        # len(coefficients) == len(constraints)
        self.assertRaises(RuntimeError, opt.add_column, m, m.y, -2, [m.c], [1, 2])
        self.assertRaises(RuntimeError, opt.add_column, m, m.y, -2, [m.c, z], [1])

        # add indexed constraint
        self.assertRaises(AttributeError, opt.add_column, m, m.y, -2, [m.ci], [1])
        # add something not a _ConstraintData
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
        m = pe.ConcreteModel()
        m.x1 = pe.Var()
        m.x2 = pe.Var()
        m.x3 = pe.Var()

        m.obj = pe.Objective(rule=lambda m: 2 * m.x1 + m.x2 + m.x3, sense=pe.minimize)
        m.equ1 = pe.Constraint(rule=lambda m: m.x1 + m.x2 + m.x3 == 1)
        m.cone = pe.Constraint(rule=lambda m: m.x2 * m.x2 + m.x3 * m.x3 <= m.x1 * m.x1)
        m.equ2 = pe.Constraint(rule=lambda m: m.x1 >= 0)

        opt = pe.SolverFactory('xpress_direct')
        opt.options['XSLP_SOLVER'] = 0

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
        m = pe.ConcreteModel()
        m.x1 = pe.Var()
        m.x2 = pe.Var()
        m.x3 = pe.Var()

        m.obj = pe.Objective(rule=lambda m: 2 * m.x1 + m.x2 + m.x3, sense=pe.minimize)
        m.equ1a = pe.Constraint(rule=lambda m: m.x1 + m.x2 + m.x3 == 1)
        m.equ1b = pe.Constraint(rule=lambda m: m.x1 + m.x2 + m.x3 == -1)
        m.cone = pe.Constraint(rule=lambda m: m.x2 * m.x2 + m.x3 * m.x3 <= m.x1 * m.x1)
        m.equ2 = pe.Constraint(rule=lambda m: m.x1 >= 0)

        opt = pe.SolverFactory('xpress_direct')
        opt.options['XSLP_SOLVER'] = 0

        results = opt.solve(m)
        self.assertEqual(results.solver.status, SolverStatus.ok)
        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.infeasible
        )
