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

# IC_model_dict.py - Implement a particular instance of (H)

# @fct:
import pyomo.environ as pyo


def IC_model_dict(ICD):
    # ICD is a dictionary with the data for the problem

    model = pyo.ConcreteModel(name="(H)")

    model.A = pyo.Set(initialize=ICD["A"])

    model.h = pyo.Param(model.A, initialize=ICD["h"])
    model.d = pyo.Param(model.A, initialize=ICD["d"])
    model.c = pyo.Param(model.A, initialize=ICD["c"])
    model.b = pyo.Param(initialize=ICD["b"])
    model.u = pyo.Param(model.A, initialize=ICD["u"])

    def xbounds_rule(model, i):
        return (0, model.u[i])

    model.x = pyo.Var(model.A, bounds=xbounds_rule)

    def obj_rule(model):
        return sum(
            model.h[i] * (model.x[i] - (model.x[i] / model.d[i]) ** 2) for i in model.A
        )

    model.z = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

    def budget_rule(model):
        return sum(model.c[i] * model.x[i] for i in model.A) <= model.b

    model.budgetconstr = pyo.Constraint(rule=budget_rule)

    return model


# @:fct

if __name__ == "__main__":
    D = dict()
    D["A"] = ['I_C_Scoops', 'Peanuts']
    D["h"] = {'I_C_Scoops': 1, 'Peanuts': 0.1}
    D["d"] = {'I_C_Scoops': 5, 'Peanuts': 27}
    D["c"] = {'I_C_Scoops': 3.14, 'Peanuts': 0.2718}
    D["b"] = 12
    D["u"] = {'I_C_Scoops': 100, 'Peanuts': 40.6}

    model = IC_model_dict(D)
    model.pprint()
