from pyomo.environ import *

model = ConcreteModel()

# @decl1:
model.e = Expression()
# @:decl1

model.pprint()
model = None
model = ConcreteModel()

# @decl2:
model.x = Var()
model.e1 = Expression(expr=model.x + 1)
def e2_rule(model):
    return model.x + 2
model.e2 = Expression(rule=e2_rule)
# @:decl2

model.pprint()
del e2_rule
model = None
model = ConcreteModel()

# @decl3:
N = [1,2,3]
model.x = Var(N)
def e_rule(model, i):
    if i == 1:
        return Expression.Skip
    else:
        return model.x[i]**2
model.e = Expression(N, rule=e_rule)
# @:decl3

model.pprint()
del e_rule
model = None
model = ConcreteModel()

# @decl4:
model.x = Var()
model.e = Expression(expr=(model.x - 1.0)**2)
model.o = Objective(expr=0.1*model.e + model.x)
model.c = Constraint(expr=model.e <= 1.0)
# @:decl4

model.pprint()

# @decl5:
model.x.set_value(2.0)
print(value(model.e))       # 1.0
print(value(model.o))       # 2.1
print(value(model.c.body))  # 1.0

model.e.set_value((model.x - 2.0)**2)
print(value(model.e))       # 0.0
print(value(model.o))       # 2.0
print(value(model.c.body))  # 0.0
# @:decl5

model.pprint()
