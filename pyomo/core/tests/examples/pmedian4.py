#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

from pyomo.environ import *
import math

N = 5
M = 6
P = 3

model = ConcreteModel()

model.Locations = RangeSet(1,N)

model.Customers = RangeSet(1,M)

def d_rule(model, n, m):
    return math.sin(n*2.33333+m*7.99999)
model.d = Param(model.Locations, model.Customers, initialize=d_rule, within=Reals)

model.x = Var(model.Locations, model.Customers, bounds=(0.0,1.0))

model.y = Var(model.Locations, within=Binary)

def rule(model):
    return sum( [model.d[n,m]*model.x[n,m] for n in model.Locations for m in model.Customers] )
model.obj = Objective(rule=rule)

model.single_x = ConstraintList()
for m in model.Customers:
    model.single_x.add( sum( [model.x[n,m] for n in model.Locations]) == 1.0 )

model.bound_y = ConstraintList()
for n in model.Locations:
    for m in model.Customers:
        model.bound_y.add( model.x[n,m] <= model.y[n] )

model.num_facilities = Constraint(expr=sum( [model.y[n] for n in model.Locations] ) == P)
