from pyomo.environ import *

model = ConcreteModel()
model.x = Var([1,2], initialize=1.0, bounds=(0,1))
model.o = Objective(expr=(model.x[1]+1)**2 + (model.x[2]+1)**2)

