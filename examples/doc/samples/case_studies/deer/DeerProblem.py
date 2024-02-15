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

from pyomo.core import *

#
# Model
#

model = AbstractModel()

model.p1 = Param()
model.p2 = Param()
model.p3 = Param()
model.p4 = Param()
model.p5 = Param()
model.p6 = Param()
model.p7 = Param()
model.p8 = Param()
model.p9 = Param()
model.ps = Param()

model.f = Var(initialize=20, within=PositiveReals)
model.d = Var(initialize=20, within=PositiveReals)
model.b = Var(initialize=20, within=PositiveReals)

model.hf = Var(initialize=20, within=PositiveReals)
model.hd = Var(initialize=20, within=PositiveReals)
model.hb = Var(initialize=20, within=PositiveReals)

model.br = Var(initialize=1.5, within=PositiveReals)

model.c = Var(initialize=500000, within=PositiveReals)


def obj_rule(amodel):
    return 10 * amodel.hb + amodel.hd + amodel.hf


model.obj = Objective(rule=obj_rule, sense=maximize)


def f_bal_rule(amodel):
    return (
        amodel.f
        == amodel.p1 * amodel.br * (amodel.p2 / 10.0 * amodel.f + amodel.p3 * amodel.d)
        - amodel.hf
    )


model.f_bal = Constraint(rule=f_bal_rule)


def d_bal_rule(amodel):
    return amodel.d == amodel.p4 * amodel.d + amodel.p5 / 2.0 * amodel.f - amodel.hd


model.d_bal = Constraint(rule=d_bal_rule)


def b_bal_rule(amodel):
    return amodel.b == amodel.p6 * amodel.b + amodel.p5 / 2.0 * amodel.f - amodel.hb


model.b_bal = Constraint(rule=b_bal_rule)


def food_cons_rule(amodel):
    return (
        amodel.c == amodel.p7 * amodel.b + amodel.p8 * amodel.d + amodel.p9 * amodel.f
    )


model.food_cons = Constraint(rule=food_cons_rule)


def supply_rule(amodel):
    return amodel.c <= amodel.ps


model.supply = Constraint(rule=supply_rule)


def birth_rule(amodel):
    return amodel.br == 1.1 + 0.8 * (amodel.ps - amodel.c) / amodel.ps


model.birth = Constraint(rule=birth_rule)


def proc_rule(amodel):
    return amodel.b >= 1.0 / 5.0 * (0.4 * amodel.f + amodel.d)


model.proc = Constraint(rule=proc_rule)
