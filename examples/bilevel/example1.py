import pyomo.environ
from pyomo.core import *
from pyomo.bilevel import *

# A bilevel program with no solution
# Bard, 1998
#
# min [x1 x2] [2 3] [y1]
#             [4 1] [y2]
#
# x1+x2=1
# x1 >= 0
# x2 >= 0
#
# y = argmin(...)

model = ConcreteModel()
model.x = Var([1,2], within=NonNegativeReals)
model.y = Var([1,2], within=NonNegativeReals)
model.o = Objective(expr=model.x[1]*(2*model.y[1]+3*model.y[2]) + model.x[2]*(4*model.y[1]+3*model.y[2]))
model.c = Constraint(expr=model.x[1]+model.x[2] == 1)

# Create a submodel
# The argument indicates the lower-level decision variables
model.sub = SubModel(fixed=model.x)
model.sub.o = Objective(expr=model.x[1]*(-1*model.y[1]-4*model.y[2]) + model.x[2]*(-3*model.y[1]-2*model.y[2]))
model.sub.c = Constraint(expr=model.y[1]+model.y[2] == 1)

