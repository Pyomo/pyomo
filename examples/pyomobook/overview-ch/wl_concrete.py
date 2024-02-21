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

# wl_concrete.py
# ConcreteModel version of warehouse location problem
import pyomo.environ as pyo


def create_warehouse_model(N, M, d, P):
    model = pyo.ConcreteModel(name="(WL)")

    model.x = pyo.Var(N, M, bounds=(0, 1))
    model.y = pyo.Var(N, within=pyo.Binary)

    def obj_rule(mdl):
        return sum(d[n, m] * mdl.x[n, m] for n in N for m in M)

    model.obj = pyo.Objective(rule=obj_rule)

    # @deliver:
    def demand_rule(mdl, m):
        return sum(mdl.x[n, m] for n in N) == 1

    model.demand = pyo.Constraint(M, rule=demand_rule)
    # @:deliver

    def warehouse_active_rule(mdl, n, m):
        return mdl.x[n, m] <= mdl.y[n]

    model.warehouse_active = pyo.Constraint(N, M, rule=warehouse_active_rule)

    def num_warehouses_rule(mdl):
        return sum(mdl.y[n] for n in N) <= P

    model.num_warehouses = pyo.Constraint(rule=num_warehouses_rule)

    return model
