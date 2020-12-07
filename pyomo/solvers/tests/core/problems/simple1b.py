from pyomo.core import *

model = ConcreteModel()

model.x = Var(bounds=(1,None))
model.y = Var(bounds=(1,None))

model.o = Objective(expr=-model.x-model.y, sense=maximize)
