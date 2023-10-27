import unittest
import itertools
import math
import pyomo.environ as pe
import coramin
from coramin.relaxations.alphabb import AlphaBBRelaxation


class TestAlphaBBRelaxation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model = pe.ConcreteModel()
        cls.model = model
        model.x = pe.Var(bounds=(-2, 1))
        model.y = pe.Var(bounds=(-1, 1))
        model.w = pe.Var()

        model.f_x = pe.cos(model.x)*pe.sin(model.y) - model.x/(model.y**2 + 1)

        model.obj = pe.Objective(expr=model.w)
        model.abb = AlphaBBRelaxation()
        model.abb.build(
            aux_var=model.w, f_x_expr=model.f_x,
            relaxation_side=coramin.RelaxationSide.UNDER,
            eigenvalue_bounder=coramin.EigenValueBounder.GershgorinWithSimplification
        )

    def test_nonlinear(self):
        model = self.model.clone()
        model.abb.use_linear_relaxation = False
        model.abb.rebuild()

        model.w.value = 0.0

        for x_v in [model.x.lb, model.x.ub]:
            for y_v in [model.y.lb, model.y.ub]:
                model.x.value = x_v
                model.y.value = y_v
                f_x_v = pe.value(model.f_x)
                abb_v = pe.value(model.abb._nonlinear.body)
                self.assertAlmostEqual(f_x_v, abb_v)

        solver = pe.SolverFactory('ipopt')
        solver.solve(model)
        self.assertLessEqual(model.w.value, pe.value(model.f_x))

    def test_linear(self):
        model = self.model.clone()
        model.abb.use_linear_relaxation = True

        model.x.value = 0.0
        model.y.value = 0.0

        for _ in range(5):
            model.abb.add_oa_point()
            model.abb.rebuild()
            solver = pe.SolverFactory('appsi_gurobi')
            solver.solve(model)
            self.assertLessEqual(model.w.value, pe.value(model.f_x))
