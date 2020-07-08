import pyutilib.th as unittest
import pyomo.environ as pe
from pyomo.core.expr.taylor_series import taylor_series_expansion
try:
    import xpress
    p = xpress.problem()
    del p
    xpress_available = True
except:
    xpress_available = False

class TestXpressPersistent(unittest.TestCase):
    @unittest.skipIf(not xpress_available, "xpress is not available")
    def test_basics(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(-10, 10))
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.x**2 + m.y**2)
        m.c1 = pe.Constraint(expr=m.y >= 2*m.x + 1)

        opt = pe.SolverFactory('xpress_persistent')
        opt.set_instance(m)

        self.assertEqual(opt.get_xpress_attribute('cols'), 2)
        self.assertEqual(opt.get_xpress_attribute('rows'), 1)

        res = opt.solve()
        self.assertAlmostEqual(m.x.value, -0.4)
        self.assertAlmostEqual(m.y.value, 0.2)
        opt.load_duals()
        self.assertAlmostEqual(m.dual[m.c1], -0.4)
        del m.dual

        m.c2 = pe.Constraint(expr=m.y >= -m.x + 1)
        opt.add_constraint(m.c2)
        self.assertEqual(opt.get_xpress_attribute('cols'), 2)
        self.assertEqual(opt.get_xpress_attribute('rows'), 2)

        res = opt.solve(save_results=False, load_solutions=False)
        self.assertAlmostEqual(m.x.value, -0.4)
        self.assertAlmostEqual(m.y.value, 0.2)
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
        self.assertAlmostEqual(m.x.value, -0.4)
        self.assertAlmostEqual(m.y.value, 0.2)

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
        m.a = pe.Set(initialize=[1,2,3], ordered=True)
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
        m.a = pe.Set(initialize=[1,2,3], ordered=True)
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
