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

model = pyo.AbstractModel()

model.foods = pyo.Set()
model.nutrients = pyo.Set()
model.costs = pyo.Param(model.foods)
model.min_nutrient = pyo.Param(model.nutrients)
model.max_nutrient = pyo.Param(model.nutrients)
model.volumes = pyo.Param(model.foods)
model.max_volume = pyo.Param()
model.nutrient_value = pyo.Param(model.nutrients, model.foods)
model.amount = pyo.Var(model.foods, within=pyo.NonNegativeReals)


def costRule(model):
    return sum(model.costs[n] * model.amount[n] for n in model.foods)


model.cost = pyo.Objective(rule=costRule)


def volumeRule(model):
    return (
        sum(model.volumes[n] * model.amount[n] for n in model.foods) <= model.max_volume
    )


model.volume = pyo.Constraint(rule=volumeRule)


def nutrientRule(model, n):
    value = sum(model.nutrient_value[n, f] * model.amount[f] for f in model.foods)
    return (model.min_nutrient[n], value, model.max_nutrient[n])


model.nutrientConstraint = pyo.Constraint(model.nutrients, rule=nutrientRule)
