#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

# DeerProblem.py
import pyomo.environ as pyo

model = pyo.AbstractModel()

model.p1 = pyo.Param()
model.p2 = pyo.Param()
model.p3 = pyo.Param()
model.p4 = pyo.Param()
model.p5 = pyo.Param()
model.p6 = pyo.Param()
model.p7 = pyo.Param()
model.p8 = pyo.Param()
model.p9 = pyo.Param()
model.ps = pyo.Param()

model.f = pyo.Var(initialize=20, within=pyo.PositiveReals)
model.d = pyo.Var(initialize=20, within=pyo.PositiveReals)
model.b = pyo.Var(initialize=20, within=pyo.PositiveReals)

model.hf = pyo.Var(initialize=20, within=pyo.PositiveReals)
model.hd = pyo.Var(initialize=20, within=pyo.PositiveReals)
model.hb = pyo.Var(initialize=20, within=pyo.PositiveReals)

model.br = pyo.Var(initialize=1.5, within=pyo.PositiveReals)

model.c = pyo.Var(initialize=500000, within=pyo.PositiveReals)


def obj_rule(m):
    return 10 * m.hb + m.hd + m.hf


model.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)


def f_bal_rule(m):
    return m.f == m.p1 * m.br * (m.p2 / 10.0 * m.f + m.p3 * m.d) - m.hf


model.f_bal = pyo.Constraint(rule=f_bal_rule)


def d_bal_rule(m):
    return m.d == m.p4 * m.d + m.p5 / 2.0 * m.f - m.hd


model.d_bal = pyo.Constraint(rule=d_bal_rule)


def b_bal_rule(m):
    return m.b == m.p6 * m.b + m.p5 / 2.0 * m.f - m.hb


model.b_bal = pyo.Constraint(rule=b_bal_rule)


def food_cons_rule(m):
    return m.c == m.p7 * m.b + m.p8 * m.d + m.p9 * m.f


model.food_cons = pyo.Constraint(rule=food_cons_rule)


def supply_rule(m):
    return m.c <= m.ps


model.supply = pyo.Constraint(rule=supply_rule)


def birth_rule(m):
    return m.br == 1.1 + 0.8 * (m.ps - m.c) / m.ps


model.birth = pyo.Constraint(rule=birth_rule)


def minbuck_rule(m):
    return m.b >= 1.0 / 5.0 * (0.4 * m.f + m.d)


model.minbuck = pyo.Constraint(rule=minbuck_rule)

# create the ConcreteModel
instance = model.create_instance('DeerProblem.dat')
status = pyo.SolverFactory('ipopt').solve(instance)
pyo.assert_optimal_termination(status)

instance.pprint()
