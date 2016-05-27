from pyomo.core import *

model = ConcreteModel()

model.x = Var(bounds=(1,None), within=Integers)
model.y = Var(bounds=(1,None), within=Integers)

model.o = Objective(expr=model.x-model.x)

model.c = Constraint(expr=model.x+model.y >= 3)
