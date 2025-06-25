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

import random

import pyomo.environ as pyo


def multi_knapsack(
    num_items: int = 20, item_ub: int = 10, num_cons: int = 10, max_weight: int = 100
) -> pyo.ConcreteModel:
    "Creates a random instance of Knapsack with multiple capacity constraints."
    mod = pyo.ConcreteModel()
    mod.I = pyo.Set(initialize=range(num_items), name="I")
    mod.J = pyo.Set(initialize=range(num_cons), name="I")

    rng = random.Random(0)
    weight = [[rng.randint(0, max_weight) for _ in mod.I] for _ in mod.J]
    cost = [rng.random() for _ in mod.I]
    capacity = 0.1 * num_items * item_ub * max_weight

    mod.x = pyo.Var(mod.I, domain=pyo.Integers, bounds=(0, item_ub), name="x")
    mod.cap = pyo.Constraint(
        mod.J,
        rule=lambda m, j: sum(weight[j][i] * m.x[i] for i in m.I) <= capacity,
        name="cap",
    )
    mod.obj = pyo.Objective(
        rule=lambda m: sum(cost[i] * m.x[i] for i in m.I), sense=pyo.maximize
    )
    return mod
