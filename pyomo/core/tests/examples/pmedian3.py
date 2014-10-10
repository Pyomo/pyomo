
from pyomo.core import *
import math

model = AbstractModel()

model.N = Param(within=PositiveIntegers)

def Locations_rule(model):
    return range(1,value(model.N)+1)
model.Locations = Set()

model.P = Param(within=RangeSet(1,model.N))

model.M = Param(within=PositiveIntegers)

model.Customers = RangeSet(1,model.M)

def d_rule(model, n, m):
    return math.sin(n*2.33333+m*7.99999)
model.d = Param(model.Locations, model.Customers, within=Reals)

model.x = Var(model.Locations, model.Customers, bounds=(0.0,1.0))

model.y = Var(model.Locations, within=Binary)

def obj_rule(model):
    return sum( [model.d[n,m]*model.x[n,m] for n in model.Locations for m in model.Customers] )
model.obj = Objective()

def single_x_rule(model, m):
    return (sum( [model.x[n,m] for n in model.Locations]), 1.0)
model.single_x = Constraint(model.Customers)

def bound_y_rule(model, n, m):
    return (None, model.x[n,m] - model.y[n], 0.0)
model.bound_y = Constraint(model.Locations, model.Customers)

def num_facilities_rule(model):
    return (sum( [model.y[n] for n in model.Locations] ) - model.P, 0.0)
model.num_facilities = Constraint()
