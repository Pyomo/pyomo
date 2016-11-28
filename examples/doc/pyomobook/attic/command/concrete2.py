from pyomo.environ import *

model = ConcreteModel()
model.x = Var([1,2], within=NonNegativeReals)
model.obj = Objective(expr=model.x[1] + 2*model.x[2])
model.con1 = Constraint(expr=3*model.x[1] + 4*model.x[2]>=1)
model.con2 = Constraint(expr=2*model.x[1] + 5*model.x[2]>=2)
