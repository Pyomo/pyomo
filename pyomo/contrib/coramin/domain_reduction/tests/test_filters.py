import pyomo.environ as pe
import coramin
import unittest
from pyomo.contrib import appsi


class TestFilters(unittest.TestCase):
    def test_basic_filter(self):
        m = pe.ConcreteModel()
        m.y = pe.Var()
        m.x = pe.Var(bounds=(-2, -1))
        m.obj = pe.Objective(expr=m.y)
        m.c = pe.Constraint(expr=m.y == -m.x**2)
        coramin.relaxations.relax(m, in_place=True)
        opt = appsi.solvers.Gurobi()
        res = opt.solve(m)
        vars_to_min, vars_to_max = coramin.domain_reduction.filter_variables_from_solution([m.x])
        self.assertIn(m.x, vars_to_max)
        self.assertNotIn(m.x, vars_to_min)

    def test_aggressive_filter(self):
        m = pe.ConcreteModel()
        m.y = pe.Var()
        m.x = pe.Var(bounds=(-2, -1))
        m.obj = pe.Objective(expr=m.y)
        m.c = pe.Constraint(expr=m.y == -m.x**2)
        coramin.relaxations.relax(m, in_place=True)
        opt = appsi.solvers.Gurobi()
        vars_to_min, vars_to_max = coramin.domain_reduction.aggressive_filter(candidate_variables=[m.x], relaxation=m,
                                                                              solver=opt)
        self.assertNotIn(m.x, vars_to_max)
        self.assertNotIn(m.x, vars_to_min)
