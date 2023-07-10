import pyomo.environ as pe
import pyomo.common.unittest as unittest
from pyomo.contrib.appsi.base import TerminationCondition, Results, PersistentSolver
from pyomo.contrib.appsi.solvers.wntr import Wntr, wntr_available
import math


_default_wntr_options = dict(TOL=1e-8)


@unittest.skipUnless(wntr_available, 'wntr is not available')
class TestWntrPersistent(unittest.TestCase):
    def test_param_updates(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.p = pe.Param(initialize=1, mutable=True)
        m.c = pe.Constraint(expr=m.x == m.p)
        opt = Wntr()
        opt.wntr_options.update(_default_wntr_options)
        res = opt.solve(m)
        self.assertEqual(res.termination_condition, TerminationCondition.optimal)
        self.assertAlmostEqual(m.x.value, 1)

        m.p.value = 2
        res = opt.solve(m)
        self.assertEqual(res.termination_condition, TerminationCondition.optimal)
        self.assertAlmostEqual(m.x.value, 2)

    def test_remove_add_constraint(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.c1 = pe.Constraint(expr=m.y == (m.x - 1) ** 2)
        m.c2 = pe.Constraint(expr=m.y == pe.exp(m.x))
        opt = Wntr()
        opt.config.symbolic_solver_labels = True
        opt.wntr_options.update(_default_wntr_options)
        res = opt.solve(m)
        self.assertEqual(res.termination_condition, TerminationCondition.optimal)
        self.assertAlmostEqual(m.x.value, 0)
        self.assertAlmostEqual(m.y.value, 1)

        del m.c2
        m.c2 = pe.Constraint(expr=m.y == pe.log(m.x))
        m.x.value = 0.5
        m.y.value = 0.5
        res = opt.solve(m)
        self.assertEqual(res.termination_condition, TerminationCondition.optimal)
        self.assertAlmostEqual(m.x.value, 1)
        self.assertAlmostEqual(m.y.value, 0)

    def test_fixed_var(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.c1 = pe.Constraint(expr=m.y == (m.x - 1) ** 2)
        m.x.fix(0.5)
        opt = Wntr()
        opt.wntr_options.update(_default_wntr_options)
        res = opt.solve(m)
        self.assertEqual(res.termination_condition, TerminationCondition.optimal)
        self.assertAlmostEqual(m.x.value, 0.5)
        self.assertAlmostEqual(m.y.value, 0.25)

        m.x.unfix()
        m.c2 = pe.Constraint(expr=m.y == pe.exp(m.x))
        res = opt.solve(m)
        self.assertEqual(res.termination_condition, TerminationCondition.optimal)
        self.assertAlmostEqual(m.x.value, 0)
        self.assertAlmostEqual(m.y.value, 1)

        m.x.fix(0.5)
        del m.c2
        res = opt.solve(m)
        self.assertEqual(res.termination_condition, TerminationCondition.optimal)
        self.assertAlmostEqual(m.x.value, 0.5)
        self.assertAlmostEqual(m.y.value, 0.25)

    def test_remove_variables_params(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.z = pe.Var()
        m.z.fix(0)
        m.px = pe.Param(mutable=True, initialize=1)
        m.py = pe.Param(mutable=True, initialize=1)
        m.c1 = pe.Constraint(expr=m.x == m.px)
        m.c2 = pe.Constraint(expr=m.y == m.py)
        opt = Wntr()
        opt.wntr_options.update(_default_wntr_options)
        res = opt.solve(m)
        self.assertEqual(res.termination_condition, TerminationCondition.optimal)
        self.assertAlmostEqual(m.x.value, 1)
        self.assertAlmostEqual(m.y.value, 1)
        self.assertAlmostEqual(m.z.value, 0)

        del m.c2
        del m.y
        del m.py
        m.z.value = 2
        m.px.value = 2
        res = opt.solve(m)
        self.assertEqual(res.termination_condition, TerminationCondition.optimal)
        self.assertAlmostEqual(m.x.value, 2)
        self.assertAlmostEqual(m.z.value, 2)

        del m.z
        m.px.value = 3
        res = opt.solve(m)
        self.assertEqual(res.termination_condition, TerminationCondition.optimal)
        self.assertAlmostEqual(m.x.value, 3)

    def test_get_primals(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.c1 = pe.Constraint(expr=m.y == (m.x - 1) ** 2)
        m.c2 = pe.Constraint(expr=m.y == pe.exp(m.x))
        opt = Wntr()
        opt.config.load_solution = False
        opt.wntr_options.update(_default_wntr_options)
        res = opt.solve(m)
        self.assertEqual(res.termination_condition, TerminationCondition.optimal)
        self.assertAlmostEqual(m.x.value, None)
        self.assertAlmostEqual(m.y.value, None)
        primals = opt.get_primals()
        self.assertAlmostEqual(primals[m.x], 0)
        self.assertAlmostEqual(primals[m.y], 1)

    def test_operators(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(initialize=1)
        m.c1 = pe.Constraint(expr=2 / m.x == 1)
        opt = Wntr()
        opt.wntr_options.update(_default_wntr_options)
        res = opt.solve(m)
        self.assertEqual(res.termination_condition, TerminationCondition.optimal)
        self.assertAlmostEqual(m.x.value, 2)

        del m.c1
        m.x.value = 0
        m.c1 = pe.Constraint(expr=pe.sin(m.x) == math.sin(math.pi / 4))
        res = opt.solve(m)
        self.assertEqual(res.termination_condition, TerminationCondition.optimal)
        self.assertAlmostEqual(m.x.value, math.pi / 4)

        del m.c1
        m.c1 = pe.Constraint(expr=pe.cos(m.x) == 0)
        res = opt.solve(m)
        self.assertEqual(res.termination_condition, TerminationCondition.optimal)
        self.assertAlmostEqual(m.x.value, math.pi / 2)

        del m.c1
        m.c1 = pe.Constraint(expr=pe.tan(m.x) == 1)
        m.x.value = 0
        res = opt.solve(m)
        self.assertEqual(res.termination_condition, TerminationCondition.optimal)
        self.assertAlmostEqual(m.x.value, math.pi / 4)

        del m.c1
        m.c1 = pe.Constraint(expr=pe.asin(m.x) == math.asin(0.5))
        res = opt.solve(m)
        self.assertEqual(res.termination_condition, TerminationCondition.optimal)
        self.assertAlmostEqual(m.x.value, 0.5)

        del m.c1
        m.c1 = pe.Constraint(expr=pe.acos(m.x) == math.acos(0.6))
        res = opt.solve(m)
        self.assertEqual(res.termination_condition, TerminationCondition.optimal)
        self.assertAlmostEqual(m.x.value, 0.6)

        del m.c1
        m.c1 = pe.Constraint(expr=pe.atan(m.x) == math.atan(0.5))
        res = opt.solve(m)
        self.assertEqual(res.termination_condition, TerminationCondition.optimal)
        self.assertAlmostEqual(m.x.value, 0.5)

        del m.c1
        m.c1 = pe.Constraint(expr=pe.sqrt(m.x) == math.sqrt(0.6))
        res = opt.solve(m)
        self.assertEqual(res.termination_condition, TerminationCondition.optimal)
        self.assertAlmostEqual(m.x.value, 0.6)
