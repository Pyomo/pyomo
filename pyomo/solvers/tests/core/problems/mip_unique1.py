from pyomo.core import *

model = ConcreteModel()

model.x = Var(within=Binary)
model.y = Var(within=Binary)
model.z = Var(within=Binary)

model.o = Objective(expr=model.x+2*model.y+3*model.z)

model.c1 = Constraint(expr=model.x+model.y >= 1)
model.c2 = Constraint(expr=model.x+model.z >= 1)
model.c3 = Constraint(expr=model.y+model.z >= 1)

