import pyutilib.th as unittest
import pyomo.environ as pe
from pyomo.core.expr.taylor_series import taylor_series_expansion
try:
    import gurobipy
    m = gurobipy.Model()
    gurobipy_available = True
except:
    gurobipy_available = False


class TestGurobiPersistent(unittest.TestCase):
    @unittest.skipIf(not gurobipy_available, "gurobipy is not available")
    def test_basics(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(-10, 10))
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.x**2 + m.y**2)
        m.c1 = pe.Constraint(expr=m.y >= 2*m.x + 1)

        opt = pe.SolverFactory('gurobi_persistent')
        opt.set_instance(m)

        self.assertEqual(opt.get_model_attr('NumVars'), 2)
        self.assertEqual(opt.get_model_attr('NumConstrs'), 1)
        self.assertEqual(opt.get_model_attr('NumQConstrs'), 0)
        self.assertEqual(opt.get_var_attr(m.x, 'LB'), -10)
        self.assertEqual(opt.get_var_attr(m.x, 'UB'), 10)

        res = opt.solve()
        self.assertAlmostEqual(m.x.value, -0.4)
        self.assertAlmostEqual(m.y.value, 0.2)
        opt.load_duals()
        self.assertAlmostEqual(m.dual[m.c1], -0.4)
        del m.dual

        m.c2 = pe.Constraint(expr=m.y >= -m.x + 1)
        opt.add_constraint(m.c2)
        self.assertEqual(opt.get_model_attr('NumVars'), 2)
        self.assertEqual(opt.get_model_attr('NumConstrs'), 2)
        self.assertEqual(opt.get_model_attr('NumQConstrs'), 0)

        res = opt.solve(save_results=False, load_solutions=False)
        self.assertAlmostEqual(m.x.value, -0.4)
        self.assertAlmostEqual(m.y.value, 0.2)
        opt.load_vars()
        self.assertAlmostEqual(m.x.value, 0)
        self.assertAlmostEqual(m.y.value, 1)

        opt.remove_constraint(m.c2)
        m.del_component(m.c2)
        self.assertEqual(opt.get_model_attr('NumVars'), 2)
        self.assertEqual(opt.get_model_attr('NumConstrs'), 1)
        self.assertEqual(opt.get_model_attr('NumQConstrs'), 0)

        self.assertEqual(opt.get_gurobi_param_info('FeasibilityTol')[2], 1e-6)
        res = opt.solve(options={'FeasibilityTol': '1e-7'})
        self.assertEqual(opt.get_gurobi_param_info('FeasibilityTol')[2], 1e-7)
        self.assertAlmostEqual(m.x.value, -0.4)
        self.assertAlmostEqual(m.y.value, 0.2)

        m.x.setlb(-5)
        m.x.setub(5)
        opt.update_var(m.x)
        self.assertEqual(opt.get_var_attr(m.x, 'LB'), -5)
        self.assertEqual(opt.get_var_attr(m.x, 'UB'), 5)

        m.x.fix(0)
        opt.update_var(m.x)
        self.assertEqual(opt.get_var_attr(m.x, 'LB'), 0)
        self.assertEqual(opt.get_var_attr(m.x, 'UB'), 0)

        m.x.unfix()
        opt.update_var(m.x)
        self.assertEqual(opt.get_var_attr(m.x, 'LB'), -5)
        self.assertEqual(opt.get_var_attr(m.x, 'UB'), 5)

        m.c2 = pe.Constraint(expr=m.y >= m.x**2)
        opt.add_constraint(m.c2)
        self.assertEqual(opt.get_model_attr('NumVars'), 2)
        self.assertEqual(opt.get_model_attr('NumConstrs'), 1)
        self.assertEqual(opt.get_model_attr('NumQConstrs'), 1)

        opt.remove_constraint(m.c2)
        m.del_component(m.c2)
        self.assertEqual(opt.get_model_attr('NumVars'), 2)
        self.assertEqual(opt.get_model_attr('NumConstrs'), 1)
        self.assertEqual(opt.get_model_attr('NumQConstrs'), 0)

        m.z = pe.Var()
        opt.add_var(m.z)
        self.assertEqual(opt.get_model_attr('NumVars'), 3)
        opt.remove_var(m.z)
        del m.z
        self.assertEqual(opt.get_model_attr('NumVars'), 2)

    @unittest.skipIf(not gurobipy_available, "gurobipy is not available")
    def test_update1(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.z = pe.Var()
        m.obj = pe.Objective(expr=m.z)
        m.c1 = pe.Constraint(expr=m.z >= m.x**2 + m.y**2)

        opt = pe.SolverFactory('gurobi_persistent')
        opt.set_instance(m)
        self.assertEqual(opt._solver_model.getAttr('NumQConstrs'), 0)

        opt.remove_constraint(m.c1)
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumQConstrs'), 0)

        opt.add_constraint(m.c1)
        self.assertEqual(opt._solver_model.getAttr('NumQConstrs'), 0)
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumQConstrs'), 1)

    @unittest.skipIf(not gurobipy_available, "gurobipy is not available")
    def test_update2(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.z = pe.Var()
        m.obj = pe.Objective(expr=m.z)
        m.c2 = pe.Constraint(expr=m.x + m.y == 1)

        opt = pe.SolverFactory('gurobi_persistent')
        opt.set_instance(m)
        self.assertEqual(opt._solver_model.getAttr('NumConstrs'), 0)

        opt.remove_constraint(m.c2)
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumConstrs'), 0)

        opt.add_constraint(m.c2)
        self.assertEqual(opt._solver_model.getAttr('NumConstrs'), 0)
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumConstrs'), 1)

    @unittest.skipIf(not gurobipy_available, "gurobipy is not available")
    def test_update3(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.z = pe.Var()
        m.obj = pe.Objective(expr=m.z)
        m.c1 = pe.Constraint(expr=m.z >= m.x**2 + m.y**2)

        opt = pe.SolverFactory('gurobi_persistent')
        opt.set_instance(m)
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumQConstrs'), 1)
        m.c2 = pe.Constraint(expr=m.y >= m.x**2)
        opt.add_constraint(m.c2)
        self.assertEqual(opt._solver_model.getAttr('NumQConstrs'), 1)
        opt.remove_constraint(m.c2)
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumQConstrs'), 1)

    @unittest.skipIf(not gurobipy_available, "gurobipy is not available")
    def test_update4(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.z = pe.Var()
        m.obj = pe.Objective(expr=m.z)
        m.c1 = pe.Constraint(expr=m.z >= m.x + m.y)

        opt = pe.SolverFactory('gurobi_persistent')
        opt.set_instance(m)
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumConstrs'), 1)
        m.c2 = pe.Constraint(expr=m.y >= m.x)
        opt.add_constraint(m.c2)
        self.assertEqual(opt._solver_model.getAttr('NumConstrs'), 1)
        opt.remove_constraint(m.c2)
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumConstrs'), 1)

    @unittest.skipIf(not gurobipy_available, "gurobipy is not available")
    def test_update5(self):
        m = pe.ConcreteModel()
        m.a = pe.Set(initialize=[1,2,3], ordered=True)
        m.x = pe.Var(m.a, within=pe.Binary)
        m.y = pe.Var(within=pe.Binary)
        m.obj = pe.Objective(expr=m.y)
        m.c1 = pe.SOSConstraint(var=m.x, sos=1)

        opt = pe.SolverFactory('gurobi_persistent')
        opt.set_instance(m)
        self.assertEqual(opt._solver_model.getAttr('NumSOS'), 0)

        opt.remove_sos_constraint(m.c1)
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumSOS'), 0)

        opt.add_sos_constraint(m.c1)
        self.assertEqual(opt._solver_model.getAttr('NumSOS'), 0)
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumSOS'), 1)

    @unittest.skipIf(not gurobipy_available, "gurobipy is not available")
    def test_update6(self):
        m = pe.ConcreteModel()
        m.a = pe.Set(initialize=[1,2,3], ordered=True)
        m.x = pe.Var(m.a, within=pe.Binary)
        m.y = pe.Var(within=pe.Binary)
        m.obj = pe.Objective(expr=m.y)
        m.c1 = pe.SOSConstraint(var=m.x, sos=1)

        opt = pe.SolverFactory('gurobi_persistent')
        opt.set_instance(m)
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumSOS'), 1)
        m.c2 = pe.SOSConstraint(var=m.x, sos=2)
        opt.add_sos_constraint(m.c2)
        self.assertEqual(opt._solver_model.getAttr('NumSOS'), 1)
        opt.remove_sos_constraint(m.c2)
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumSOS'), 1)

    @unittest.skipIf(not gurobipy_available, "gurobipy is not available")
    def test_update7(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()

        opt = pe.SolverFactory('gurobi_persistent')
        opt.set_instance(m)
        self.assertEqual(opt._solver_model.getAttr('NumVars'), 0)

        opt.remove_var(m.x)
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumVars'), 1)

        opt.add_var(m.x)
        self.assertEqual(opt._solver_model.getAttr('NumVars'), 1)
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumVars'), 2)

        opt.remove_var(m.x)
        opt.update()
        opt.add_var(m.x)
        opt.remove_var(m.x)
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumVars'), 1)

    @unittest.skipIf(not gurobipy_available, "gurobipy is not available")
    def test_linear_constraint_attr(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.c = pe.Constraint(expr=m.x + m.y == 1)

        opt = pe.SolverFactory('gurobi_persistent')
        opt.set_instance(m)
        opt.set_linear_constraint_attr(m.c, 'Lazy', 1)
        self.assertEqual(opt.get_linear_constraint_attr(m.c, 'Lazy'), 1)

    @unittest.skipIf(not gurobipy_available, "gurobipy is not available")
    def test_quadratic_constraint_attr(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.c = pe.Constraint(expr=m.y >= m.x**2)

        opt = pe.SolverFactory('gurobi_persistent')
        opt.set_instance(m)
        self.assertEqual(opt.get_quadratic_constraint_attr(m.c, 'QCRHS'), 0)

    @unittest.skipIf(not gurobipy_available, "gurobipy is not available")
    def test_var_attr(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(within=pe.Binary)

        opt = pe.SolverFactory('gurobi_persistent')
        opt.set_instance(m)
        opt.set_var_attr(m.x, 'Start', 1)
        self.assertEqual(opt.get_var_attr(m.x, 'Start'), 1)

    @unittest.skipIf(not gurobipy_available, "gurobipy is not available")
    def test_callback(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(0, 4))
        m.y = pe.Var(within=pe.Integers, bounds=(0, None))
        m.obj = pe.Objective(expr=2*m.x + m.y)
        m.cons = pe.ConstraintList()

        def _add_cut(xval):
            m.x.value = xval
            return m.cons.add(m.y >= taylor_series_expansion((m.x - 2)**2))

        _add_cut(0)
        _add_cut(4)

        opt = pe.SolverFactory('gurobi_persistent')
        opt.set_instance(m)
        opt.set_gurobi_param('PreCrush', 1)
        opt.set_gurobi_param('LazyConstraints', 1)

        def _my_callback(cb_m, cb_opt, cb_where):
            if cb_where == gurobipy.GRB.Callback.MIPSOL:
                cb_opt.cbGetSolution(vars=[m.x, m.y])
                if m.y.value < (m.x.value - 2)**2 - 1e-6:
                    cb_opt.cbLazy(_add_cut(m.x.value))

        opt.set_callback(_my_callback)
        opt.solve()
        self.assertAlmostEqual(m.x.value, 1)
        self.assertAlmostEqual(m.y.value, 1)
