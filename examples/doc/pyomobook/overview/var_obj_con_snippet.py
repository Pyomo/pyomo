from pyomo.environ import *

model = ConcreteModel()
# @body:
model.x = Var()
model.y = Var(bounds=(-2,4))
model.z = Var(initialize=1.0, within=NonNegativeReals)

model.obj = Objective(expr=model.x**2 + model.y + model.z)

model.eq_con = Constraint(expr=model.x + model.y + model.z == 1)
model.ineq_con = Constraint(expr=model.x + model.y <= 0)
# @:body
