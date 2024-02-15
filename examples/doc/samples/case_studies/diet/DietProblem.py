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

model = AbstractModel()

model.foods = Set()
model.nutrients = Set()
model.costs = Param(model.foods)
model.min_nutrient = Param(model.nutrients)
model.max_nutrient = Param(model.nutrients)
model.volumes = Param(model.foods)
model.max_volume = Param()
model.nutrient_value = Param(model.nutrients, model.foods)
model.amount = Var(model.foods, within=NonNegativeReals)


def costRule(model):
    return sum(model.costs[n] * model.amount[n] for n in model.foods)


model.cost = Objective(rule=costRule)


def volumeRule(model):
    return (
        sum(model.volumes[n] * model.amount[n] for n in model.foods) <= model.max_volume
    )


model.volume = Constraint(rule=volumeRule)


def nutrientRule(model, n):
    value = sum(model.nutrient_value[n, f] * model.amount[f] for f in model.foods)
    return (model.min_nutrient[n], value, model.max_nutrient[n])


model.nutrientConstraint = Constraint(model.nutrients, rule=nutrientRule)
