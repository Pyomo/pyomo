from pyomo.environ import *

print("indexed1")
# --------------------------------------------------
# @indexed1:
model = ConcreteModel()
model.A = Set(initialize=[1,2,3])
model.B = Set(initialize=['Q', 'R'])
model.x = Var()
model.y = Var(model.A, model.B)
model.o = Objective(expr=model.x)
model.c = Constraint(expr = model.x >= 0)
def d_rule(model, a):
    return a*model.x <= 0
model.d = Constraint(model.A, rule=d_rule)
# @:indexed1

model.pprint()
