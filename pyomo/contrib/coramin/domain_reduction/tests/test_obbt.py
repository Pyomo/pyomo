import coramin
import unittest
import pyomo.environ as pyo
from pyomo.contrib import appsi


class TestBoundsTightener(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def test_quad(self):
        model = pyo.ConcreteModel()
        model.x = pyo.Var(bounds=(-5.0, 5.0))
        model.y = pyo.Var(bounds=(-100.0, 100.0))

        model.obj_expr = pyo.Expression(expr=model.y)
        model.obj = pyo.Objective(expr=model.obj_expr)

        x_points = [-5.0, 5.0]
        model.under_estimators = pyo.ConstraintList()
        for xp in x_points:
            m = 2*xp
            b = -(xp**2)
            model.under_estimators.add(model.y >= m*model.x + b)

        solver = appsi.solvers.Ipopt()
        (lower, upper) = coramin.domain_reduction.perform_obbt(model=model, solver=solver, varlist=[model.x, model.y],
                                                               update_bounds=True)
        self.assertAlmostEqual(pyo.value(model.x.lb), -5.0, delta=1e-6)
        self.assertAlmostEqual(pyo.value(model.x.ub), 5.0, delta=1e-6)
        self.assertAlmostEqual(pyo.value(model.y.lb), -25.0, delta=1e-6)
        self.assertAlmostEqual(pyo.value(model.y.ub), 100.0, delta=1e-6)
        self.assertAlmostEqual(lower[0], -5.0, delta=1e-6)
        self.assertAlmostEqual(upper[0], 5.0, delta=1e-6)
        self.assertAlmostEqual(lower[1], -25.0, delta=1e-6)
        self.assertAlmostEqual(upper[1], 100.0, delta=1e-6)

    def test_passing_component_not_list(self):
        model = pyo.ConcreteModel()
        model.x = pyo.Var(bounds=(-5.0, 5.0))
        model.y = pyo.Var(bounds=(-100.0, 100.0))

        model.obj_expr = pyo.Expression(expr=model.y)
        model.obj = pyo.Objective(expr=model.obj_expr)

        x_points = [-5.0, 5.0]
        model.under_estimators = pyo.ConstraintList()
        for xp in x_points:
            m = 2 * xp
            b = -(xp ** 2)
            model.under_estimators.add(model.y >= m * model.x + b)

        solver = appsi.solvers.Ipopt()
        (lower, upper) = coramin.domain_reduction.perform_obbt(model=model, solver=solver, varlist=model.y, update_bounds=True)
        self.assertAlmostEqual(pyo.value(model.x.lb), -5.0, delta=1e-6)
        self.assertAlmostEqual(pyo.value(model.x.ub), 5.0, delta=1e-6)
        self.assertAlmostEqual(pyo.value(model.y.lb), -25.0, delta=1e-6)
        self.assertAlmostEqual(pyo.value(model.y.ub), 100.0, delta=1e-6)
        self.assertAlmostEqual(lower[0], -25.0, delta=1e-6)
        self.assertAlmostEqual(upper[0], 100.0, delta=1e-6)

    def test_passing_indexed_component_not_list(self):
        model = pyo.ConcreteModel()
        model.x = pyo.Var(bounds=(-5.0, 5.0))
        model.S = pyo.Set(initialize=['A', 'B'], ordered=True)
        model.y = pyo.Var(model.S, bounds=(-100.0, 100.0))

        model.obj_expr = pyo.Expression(expr=model.y['A'])
        model.obj = pyo.Objective(expr=model.obj_expr)

        x_points = [-5.0, 5.0]
        model.under_estimators = pyo.ConstraintList()
        for xp in x_points:
            m = 2 * xp
            b = -(xp ** 2)
            model.under_estimators.add(model.y['A'] >= m * model.x + b)

        model.con = pyo.Constraint(expr=model.y['A'] == 1 + model.y['B'])

        solver = appsi.solvers.Ipopt()
        lower, upper = coramin.domain_reduction.perform_obbt(model=model, solver=solver, varlist=model.y, update_bounds=True)
        self.assertAlmostEqual(pyo.value(model.x.lb), -5.0, delta=1e-6)
        self.assertAlmostEqual(pyo.value(model.x.ub), 5.0, delta=1e-6)
        self.assertAlmostEqual(pyo.value(model.y['A'].lb), -25.0, delta=1e-6)
        self.assertAlmostEqual(pyo.value(model.y['A'].ub), 100.0, delta=1e-6)
        self.assertAlmostEqual(pyo.value(model.y['B'].lb), -26.0, delta=1e-6)
        self.assertAlmostEqual(pyo.value(model.y['B'].ub), 99.0, delta=1e-6)
        self.assertAlmostEqual(lower[0], -25.0, delta=1e-6)
        self.assertAlmostEqual(upper[0], 100.0, delta=1e-6)
        self.assertAlmostEqual(lower[1], -26.0, delta=1e-6)
        self.assertAlmostEqual(upper[1], 99.0, delta=1e-6)
        
    def test_too_many_obj(self):
        model = pyo.ConcreteModel()
        model.x = pyo.Var(bounds=(-5.0, 5.0))
        model.y = pyo.Var(bounds=(-100.0, 100.0))

        model.obj1 = pyo.Objective(expr=model.x + model.y)
        model.obj2 = pyo.Objective(expr=model.x - model.y)

        solver = pyo.SolverFactory('ipopt')
        with self.assertRaises(ValueError):
            coramin.domain_reduction.perform_obbt(model=model, solver=solver, varlist=[model.x, model.y],
                                                  objective_bound=0.0, update_bounds=True)


if __name__ == '__main__':
    TestBoundsTightener()
