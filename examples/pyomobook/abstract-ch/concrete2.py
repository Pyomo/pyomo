import pyomo.environ as pyo

model = pyo.ConcreteModel()
model.x = pyo.Var([1,2], within=pyo.NonNegativeReals)
model.obj = pyo.Objective(expr=model.x[1] + 2*model.x[2])
model.con1 = pyo.Constraint(expr=3*model.x[1] + 4*model.x[2]>=1)
model.con2 = pyo.Constraint(expr=2*model.x[1] + 5*model.x[2]>=2)

