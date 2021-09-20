import pyomo.common.unittest as unittest
import pyomo.environ as pe
from pyomo.contrib.appsi.solvers.gurobi import Gurobi
from pyomo.contrib.appsi.base import TerminationCondition
from pyomo.core.expr.numeric_expr import LinearExpression
from pyomo.core.expr.taylor_series import taylor_series_expansion


opt = Gurobi()
if not opt.available():
    raise unittest.SkipTest
import gurobipy


class TestGurobiPersistentSimpleLPUpdates(unittest.TestCase):
    def setUp(self):
        self.m = pe.ConcreteModel()
        m = self.m
        m.x = pe.Var()
        m.y = pe.Var()
        m.p1 = pe.Param(mutable=True)
        m.p2 = pe.Param(mutable=True)
        m.p3 = pe.Param(mutable=True)
        m.p4 = pe.Param(mutable=True)
        m.obj = pe.Objective(expr=m.x + m.y)
        m.c1 = pe.Constraint(expr=m.y - m.p1 * m.x >= m.p2)
        m.c2 = pe.Constraint(expr=m.y - m.p3 * m.x >= m.p4)

    def get_solution(self):
        try:
            import numpy as np
        except:
            raise unittest.SkipTest('numpy is not available')
        p1 = self.m.p1.value
        p2 = self.m.p2.value
        p3 = self.m.p3.value
        p4 = self.m.p4.value
        A = np.array([[1, -p1],
                      [1, -p3]])
        rhs = np.array([p2,
                        p4])
        sol = np.linalg.solve(A, rhs)
        x = float(sol[1])
        y = float(sol[0])
        return x, y

    def set_params(self, p1, p2, p3, p4):
        self.m.p1.value = p1
        self.m.p2.value = p2
        self.m.p3.value = p3
        self.m.p4.value = p4

    def test_lp(self):
        self.set_params(-1, -2, 0.1, -2)
        x, y = self.get_solution()
        opt = Gurobi()
        res = opt.solve(self.m)
        self.assertAlmostEqual(x + y, res.best_feasible_objective)
        self.assertAlmostEqual(x + y, res.best_objective_bound)
        self.assertEqual(res.termination_condition, TerminationCondition.optimal)
        self.assertTrue(res.best_feasible_objective is not None)
        self.assertAlmostEqual(x, self.m.x.value)
        self.assertAlmostEqual(y, self.m.y.value)

        self.set_params(-1.25, -1, 0.5, -2)
        opt.config.load_solution = False
        res = opt.solve(self.m)
        self.assertAlmostEqual(x, self.m.x.value)
        self.assertAlmostEqual(y, self.m.y.value)
        x, y = self.get_solution()
        self.assertNotAlmostEqual(x, self.m.x.value)
        self.assertNotAlmostEqual(y, self.m.y.value)
        opt.load_vars()
        self.assertAlmostEqual(x, self.m.x.value)
        self.assertAlmostEqual(y, self.m.y.value)


class TestGurobiPersistent(unittest.TestCase):
    def test_range_constraints(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.xl = pe.Param(initialize=-1, mutable=True)
        m.xu = pe.Param(initialize=1, mutable=True)
        m.c = pe.Constraint(expr=pe.inequality(m.xl, m.x, m.xu))
        m.obj = pe.Objective(expr=m.x)

        opt = Gurobi()
        opt.set_instance(m)
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, -1)

        m.xl.value = -3
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, -3)

        del m.obj
        m.obj = pe.Objective(expr=m.x, sense=pe.maximize)

        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, 1)

        m.xu.value = 3
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, 3)

    def test_quadratic_constraint_with_params(self):
        m = pe.ConcreteModel()
        m.a = pe.Param(initialize=1, mutable=True)
        m.b = pe.Param(initialize=1, mutable=True)
        m.c = pe.Param(initialize=1, mutable=True)
        m.x = pe.Var()
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.y)
        m.con = pe.Constraint(expr=m.y >= m.a*m.x**2 + m.b*m.x + m.c)

        opt = Gurobi()
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, -m.b.value / (2 * m.a.value))
        self.assertAlmostEqual(m.y.value, m.a.value * m.x.value ** 2 + m.b.value * m.x.value + m.c.value)

        m.a.value = 2
        m.b.value = 4
        m.c.value = -1
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, -m.b.value / (2 * m.a.value))
        self.assertAlmostEqual(m.y.value, m.a.value * m.x.value ** 2 + m.b.value * m.x.value + m.c.value)

    def test_quadratic_objective(self):
        m = pe.ConcreteModel()
        m.a = pe.Param(initialize=1, mutable=True)
        m.b = pe.Param(initialize=1, mutable=True)
        m.c = pe.Param(initialize=1, mutable=True)
        m.x = pe.Var()
        m.obj = pe.Objective(expr=m.a*m.x**2 + m.b*m.x + m.c)

        opt = Gurobi()
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, -m.b.value / (2 * m.a.value))
        self.assertAlmostEqual(res.best_feasible_objective,
                               m.a.value * m.x.value ** 2 + m.b.value * m.x.value + m.c.value)

        m.a.value = 2
        m.b.value = 4
        m.c.value = -1
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, -m.b.value / (2 * m.a.value))
        self.assertAlmostEqual(res.best_feasible_objective,
                               m.a.value * m.x.value ** 2 + m.b.value * m.x.value + m.c.value)

    def test_var_bounds(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(-1, 1))
        m.obj = pe.Objective(expr=m.x)

        opt = Gurobi()
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, -1)

        m.x.setlb(-3)
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, -3)

        del m.obj
        m.obj = pe.Objective(expr=m.x, sense=pe.maximize)

        opt = Gurobi()
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, 1)

        m.x.setub(3)
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, 3)

    def test_fixed_var(self):
        m = pe.ConcreteModel()
        m.a = pe.Param(initialize=1, mutable=True)
        m.b = pe.Param(initialize=1, mutable=True)
        m.c = pe.Param(initialize=1, mutable=True)
        m.x = pe.Var()
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.y)
        m.con = pe.Constraint(expr=m.y >= m.a*m.x**2 + m.b*m.x + m.c)

        m.x.fix(1)
        opt = Gurobi()
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, 1)
        self.assertAlmostEqual(m.y.value, 3)

        m.x.value = 2
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, 2)
        self.assertAlmostEqual(m.y.value, 7)

        m.x.unfix()
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, -m.b.value / (2 * m.a.value))
        self.assertAlmostEqual(m.y.value, m.a.value * m.x.value ** 2 + m.b.value * m.x.value + m.c.value)

    def test_linear_constraint_attr(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.c = pe.Constraint(expr=m.x + m.y == 1)

        opt = Gurobi()
        opt.set_instance(m)
        opt.set_linear_constraint_attr(m.c, 'Lazy', 1)
        self.assertEqual(opt.get_linear_constraint_attr(m.c, 'Lazy'), 1)

    def test_quadratic_constraint_attr(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.c = pe.Constraint(expr=m.y >= m.x**2)

        opt = Gurobi()
        opt.set_instance(m)
        self.assertEqual(opt.get_quadratic_constraint_attr(m.c, 'QCRHS'), 0)

    def test_var_attr(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(within=pe.Binary)

        opt = Gurobi()
        opt.set_instance(m)
        opt.set_var_attr(m.x, 'Start', 1)
        self.assertEqual(opt.get_var_attr(m.x, 'Start'), 1)

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

        opt = Gurobi()
        opt.set_instance(m)
        opt.set_gurobi_param('PreCrush', 1)
        opt.set_gurobi_param('LazyConstraints', 1)

        def _my_callback(cb_m, cb_opt, cb_where):
            if cb_where == gurobipy.GRB.Callback.MIPSOL:
                cb_opt.cbGetSolution(vars=[m.x, m.y])
                if m.y.value < (m.x.value - 2)**2 - 1e-6:
                    cb_opt.cbLazy(_add_cut(m.x.value))

        opt.set_callback(_my_callback)
        opt.solve(m)
        self.assertAlmostEqual(m.x.value, 1)
        self.assertAlmostEqual(m.y.value, 1)

    def test_nonconvex(self):
        if gurobipy.GRB.VERSION_MAJOR < 9:
            raise unittest.SkipTest
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.x**2 + m.y**2)
        m.c = pe.Constraint(expr=m.y == (m.x-1)**2 - 2)
        opt = Gurobi()
        opt.gurobi_options['nonconvex'] = 2
        opt.solve(m)
        self.assertAlmostEqual(m.x.value, -0.3660254037844423, 2)
        self.assertAlmostEqual(m.y.value, -0.13397459621555508, 2)

    def test_nonconvex2(self):
        if gurobipy.GRB.VERSION_MAJOR < 9:
            raise unittest.SkipTest
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.x**2 + m.y**2)
        m.c1 = pe.Constraint(expr=0 <= -m.y + (m.x-1)**2 - 2)
        m.c2 = pe.Constraint(expr=0 >= -m.y + (m.x-1)**2 - 2)
        opt = Gurobi()
        opt.gurobi_options['nonconvex'] = 2
        opt.solve(m)
        self.assertAlmostEqual(m.x.value, -0.3660254037844423, 2)
        self.assertAlmostEqual(m.y.value, -0.13397459621555508, 2)


class TestManualModel(unittest.TestCase):
    def setUp(self):
        opt = Gurobi()
        opt.update_config.check_for_new_or_removed_params = False
        opt.update_config.check_for_new_or_removed_vars = False
        opt.update_config.check_for_new_or_removed_constraints = False
        opt.update_config.update_params = False
        opt.update_config.update_vars = False
        opt.update_config.update_constraints = False
        opt.update_config.update_named_expressions = False
        self.opt = opt

    def test_basics(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(-10, 10))
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.x**2 + m.y**2)
        m.c1 = pe.Constraint(expr=m.y >= 2*m.x + 1)

        opt = self.opt
        opt.set_instance(m)

        self.assertEqual(opt.get_model_attr('NumVars'), 2)
        self.assertEqual(opt.get_model_attr('NumConstrs'), 1)
        self.assertEqual(opt.get_model_attr('NumQConstrs'), 0)
        self.assertEqual(opt.get_var_attr(m.x, 'LB'), -10)
        self.assertEqual(opt.get_var_attr(m.x, 'UB'), 10)

        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, -0.4)
        self.assertAlmostEqual(m.y.value, 0.2)
        duals = opt.get_duals()
        self.assertAlmostEqual(duals[m.c1], -0.4)

        m.c2 = pe.Constraint(expr=m.y >= -m.x + 1)
        opt.add_constraints([m.c2])
        self.assertEqual(opt.get_model_attr('NumVars'), 2)
        self.assertEqual(opt.get_model_attr('NumConstrs'), 2)
        self.assertEqual(opt.get_model_attr('NumQConstrs'), 0)

        opt.config.load_solution = False
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, -0.4)
        self.assertAlmostEqual(m.y.value, 0.2)
        opt.load_vars()
        self.assertAlmostEqual(m.x.value, 0)
        self.assertAlmostEqual(m.y.value, 1)

        opt.remove_constraints([m.c2])
        m.del_component(m.c2)
        self.assertEqual(opt.get_model_attr('NumVars'), 2)
        self.assertEqual(opt.get_model_attr('NumConstrs'), 1)
        self.assertEqual(opt.get_model_attr('NumQConstrs'), 0)

        self.assertEqual(opt.get_gurobi_param_info('FeasibilityTol')[2], 1e-6)
        opt.gurobi_options['FeasibilityTol'] = 1e-7
        opt.config.load_solution = True
        res = opt.solve(m)
        self.assertEqual(opt.get_gurobi_param_info('FeasibilityTol')[2], 1e-7)
        self.assertAlmostEqual(m.x.value, -0.4)
        self.assertAlmostEqual(m.y.value, 0.2)

        m.x.setlb(-5)
        m.x.setub(5)
        opt.update_variables([m.x])
        self.assertEqual(opt.get_var_attr(m.x, 'LB'), -5)
        self.assertEqual(opt.get_var_attr(m.x, 'UB'), 5)

        m.x.fix(0)
        opt.update_variables([m.x])
        self.assertEqual(opt.get_var_attr(m.x, 'LB'), 0)
        self.assertEqual(opt.get_var_attr(m.x, 'UB'), 0)

        m.x.unfix()
        opt.update_variables([m.x])
        self.assertEqual(opt.get_var_attr(m.x, 'LB'), -5)
        self.assertEqual(opt.get_var_attr(m.x, 'UB'), 5)

        m.c2 = pe.Constraint(expr=m.y >= m.x**2)
        opt.add_constraints([m.c2])
        self.assertEqual(opt.get_model_attr('NumVars'), 2)
        self.assertEqual(opt.get_model_attr('NumConstrs'), 1)
        self.assertEqual(opt.get_model_attr('NumQConstrs'), 1)

        opt.remove_constraints([m.c2])
        m.del_component(m.c2)
        self.assertEqual(opt.get_model_attr('NumVars'), 2)
        self.assertEqual(opt.get_model_attr('NumConstrs'), 1)
        self.assertEqual(opt.get_model_attr('NumQConstrs'), 0)

        m.z = pe.Var()
        opt.add_variables([m.z])
        self.assertEqual(opt.get_model_attr('NumVars'), 3)
        opt.remove_variables([m.z])
        del m.z
        self.assertEqual(opt.get_model_attr('NumVars'), 2)

    def test_update1(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.z = pe.Var()
        m.obj = pe.Objective(expr=m.z)
        m.c1 = pe.Constraint(expr=m.z >= m.x**2 + m.y**2)

        opt = self.opt
        opt.set_instance(m)
        self.assertEqual(opt._solver_model.getAttr('NumQConstrs'), 0)

        opt.remove_constraints([m.c1])
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumQConstrs'), 0)

        opt.add_constraints([m.c1])
        self.assertEqual(opt._solver_model.getAttr('NumQConstrs'), 0)
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumQConstrs'), 1)

    def test_update2(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.z = pe.Var()
        m.obj = pe.Objective(expr=m.z)
        m.c2 = pe.Constraint(expr=m.x + m.y == 1)

        opt = self.opt
        opt.config.symbolic_solver_labels = True
        opt.set_instance(m)
        self.assertEqual(opt._solver_model.getAttr('NumConstrs'), 0)

        opt.remove_constraints([m.c2])
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumConstrs'), 0)

        opt.add_constraints([m.c2])
        self.assertEqual(opt._solver_model.getAttr('NumConstrs'), 0)
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumConstrs'), 1)

    def test_update3(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.z = pe.Var()
        m.obj = pe.Objective(expr=m.z)
        m.c1 = pe.Constraint(expr=m.z >= m.x**2 + m.y**2)

        opt = self.opt
        opt.set_instance(m)
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumQConstrs'), 1)
        m.c2 = pe.Constraint(expr=m.y >= m.x**2)
        opt.add_constraints([m.c2])
        self.assertEqual(opt._solver_model.getAttr('NumQConstrs'), 1)
        opt.remove_constraints([m.c2])
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumQConstrs'), 1)

    def test_update4(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.z = pe.Var()
        m.obj = pe.Objective(expr=m.z)
        m.c1 = pe.Constraint(expr=m.z >= m.x + m.y)

        opt = self.opt
        opt.set_instance(m)
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumConstrs'), 1)
        m.c2 = pe.Constraint(expr=m.y >= m.x)
        opt.add_constraints([m.c2])
        self.assertEqual(opt._solver_model.getAttr('NumConstrs'), 1)
        opt.remove_constraints([m.c2])
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumConstrs'), 1)

    def test_update5(self):
        m = pe.ConcreteModel()
        m.a = pe.Set(initialize=[1,2,3], ordered=True)
        m.x = pe.Var(m.a, within=pe.Binary)
        m.y = pe.Var(within=pe.Binary)
        m.obj = pe.Objective(expr=m.y)
        m.c1 = pe.SOSConstraint(var=m.x, sos=1)

        opt = self.opt
        opt.set_instance(m)
        self.assertEqual(opt._solver_model.getAttr('NumSOS'), 0)

        opt.remove_sos_constraints([m.c1])
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumSOS'), 0)

        opt.add_sos_constraints([m.c1])
        self.assertEqual(opt._solver_model.getAttr('NumSOS'), 0)
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumSOS'), 1)

    def test_update6(self):
        m = pe.ConcreteModel()
        m.a = pe.Set(initialize=[1,2,3], ordered=True)
        m.x = pe.Var(m.a, within=pe.Binary)
        m.y = pe.Var(within=pe.Binary)
        m.obj = pe.Objective(expr=m.y)
        m.c1 = pe.SOSConstraint(var=m.x, sos=1)

        opt = self.opt
        opt.set_instance(m)
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumSOS'), 1)
        m.c2 = pe.SOSConstraint(var=m.x, sos=2)
        opt.add_sos_constraints([m.c2])
        self.assertEqual(opt._solver_model.getAttr('NumSOS'), 1)
        opt.remove_sos_constraints([m.c2])
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumSOS'), 1)

    def test_update7(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()

        opt = self.opt
        opt.set_instance(m)
        self.assertEqual(opt._solver_model.getAttr('NumVars'), 0)

        opt.remove_variables([m.x])
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumVars'), 1)

        opt.add_variables([m.x])
        self.assertEqual(opt._solver_model.getAttr('NumVars'), 1)
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumVars'), 2)

        opt.remove_variables([m.x])
        opt.update()
        opt.add_variables([m.x])
        opt.remove_variables([m.x])
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumVars'), 1)
