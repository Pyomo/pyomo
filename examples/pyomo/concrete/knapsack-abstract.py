#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

#
# Abstract Knapsack Problem
#

from pyomo.core import *

model = AbstractModel()

model.ITEMS = Set()

model.v = Param(model.ITEMS, within=PositiveReals)

model.w = Param(model.ITEMS, within=PositiveReals)

model.limit = Param(within=PositiveReals)

model.x = Var(model.ITEMS, within=Binary)

def value_rule(model):
    return sum(model.v[i]*model.x[i] for i in model.ITEMS)
model.value = Objective(sense=maximize)

def weight_rule(model):
    return sum(model.w[i]*model.x[i] for i in model.ITEMS) <= model.limit
model.weight = Constraint()
