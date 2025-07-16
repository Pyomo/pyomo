#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.environ as pyo

#
# Model
#

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


def obj_rule(amodel):
    return 10 * amodel.hb + amodel.hd + amodel.hf


model.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)


def f_bal_rule(amodel):
    return (
        amodel.f
        == amodel.p1 * amodel.br * (amodel.p2 / 10.0 * amodel.f + amodel.p3 * amodel.d)
        - amodel.hf
    )


model.f_bal = pyo.Constraint(rule=f_bal_rule)


def d_bal_rule(amodel):
    return amodel.d == amodel.p4 * amodel.d + amodel.p5 / 2.0 * amodel.f - amodel.hd


model.d_bal = pyo.Constraint(rule=d_bal_rule)


def b_bal_rule(amodel):
    return amodel.b == amodel.p6 * amodel.b + amodel.p5 / 2.0 * amodel.f - amodel.hb


model.b_bal = pyo.Constraint(rule=b_bal_rule)


def food_cons_rule(amodel):
    return (
        amodel.c == amodel.p7 * amodel.b + amodel.p8 * amodel.d + amodel.p9 * amodel.f
    )


model.food_cons = pyo.Constraint(rule=food_cons_rule)


def supply_rule(amodel):
    return amodel.c <= amodel.ps


model.supply = pyo.Constraint(rule=supply_rule)


def birth_rule(amodel):
    return amodel.br == 1.1 + 0.8 * (amodel.ps - amodel.c) / amodel.ps


model.birth = pyo.Constraint(rule=birth_rule)


def proc_rule(amodel):
    return amodel.b >= 1.0 / 5.0 * (0.4 * amodel.f + amodel.d)


model.proc = pyo.Constraint(rule=proc_rule)
