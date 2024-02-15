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

# wl_abstract.py: AbstractModel version of warehouse location determination problem
import pyomo.environ as pyo

model = pyo.AbstractModel(name="(WL)")
# @setdecl:
model.N = pyo.Set()
model.M = pyo.Set()
# @:setdecl
# @paramdecl:
model.d = pyo.Param(model.N, model.M)
model.P = pyo.Param()
# @:paramdecl
# @vardecl:
model.x = pyo.Var(model.N, model.M, bounds=(0, 1))
model.y = pyo.Var(model.N, within=pyo.Binary)
# @:vardecl


def obj_rule(model):
    return sum(model.d[n, m] * model.x[n, m] for n in model.N for m in model.M)


model.obj = pyo.Objective(rule=obj_rule)


# @deliver:
def one_per_cust_rule(model, m):
    return sum(model.x[n, m] for n in model.N) == 1


model.one_per_cust = pyo.Constraint(model.M, rule=one_per_cust_rule)
# @:deliver


def warehouse_active_rule(model, n, m):
    return model.x[n, m] <= model.y[n]


model.warehouse_active = pyo.Constraint(model.N, model.M, rule=warehouse_active_rule)


def num_warehouses_rule(model):
    return sum(model.y[n] for n in model.N) <= model.P


model.num_warehouses = pyo.Constraint(rule=num_warehouses_rule)
