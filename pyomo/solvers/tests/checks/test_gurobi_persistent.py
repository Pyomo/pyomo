import pyutilib.th as unittest
import pyomo.environ as pe
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
        m.x = pe.Var()
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.x**2 + m.y**2)
        m.c1 = pe.Constraint(expr=m.y >= 2*m.x + 1)

        opt = pe.SolverFactory('gurobi_persistent')
        opt.set_instance(m)

        self.assertEqual(opt.get_model_attr('NumVars'), 2)
        self.assertEqual(opt.get_model_attr('NumConstrs'), 1)

        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, -0.4)
        self.assertAlmostEqual(m.y.value, 0.2)
        opt.load_duals()
        self.assertAlmostEqual(m.dual[m.c1], -0.4)
        del m.dual

        m.c2 = pe.Constraint(expr=m.y >= -m.x + 1)
        opt.add_constraint(m.c2)
        self.assertEqual(opt.get_model_attr('NumVars'), 2)
        self.assertEqual(opt.get_model_attr('NumConstrs'), 2)

        res = opt.solve(save_results=False, load_solutions=False)
        self.assertAlmostEqual(m.x.value, -0.4)
        self.assertAlmostEqual(m.y.value, 0.2)
        opt.load_vars()
        self.assertAlmostEqual(m.x.value, 0)
        self.assertAlmostEqual(m.y.value, 1)

        opt.remove_constraint(m.c2)
        self.assertEqual(opt.get_model_attr('NumVars'), 2)
        self.assertEqual(opt.get_model_attr('NumConstrs'), 1)
