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

model.ITEMS = Set()

model.v = Param(model.ITEMS, within=PositiveReals)

model.w = Param(model.ITEMS, within=PositiveReals)

model.limit = Param(within=PositiveReals)

model.x = Var(model.ITEMS, within=Binary)


def value_rule(model):
    return sum(model.v[i] * model.x[i] for i in model.ITEMS)


model.value = Objective(sense=maximize, rule=value_rule)


def weight_rule(model):
    return sum(model.w[i] * model.x[i] for i in model.ITEMS) <= model.limit


model.weight = Constraint(rule=weight_rule)
