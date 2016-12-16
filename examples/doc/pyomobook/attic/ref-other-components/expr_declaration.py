from pyomo.environ import *


print("*"*5 + " decl1 ")
# @decl1:
model = ConcreteModel()
model.e = Expression(initialize=0)
model.o = Objective(expr=1.0+model.e)

print(value(model.o))   # 1.0
model.e.set_value(1.0)
print(value(model.o))   # 2.0
# @:decl1

print("*"*5 + " decl2 ")
# @decl2:
model = ConcreteModel()
model.x = Var(initialize=1.0)
model.e = Expression(initialize=model.x)
model.o = Objective(expr=1.0+model.e)

print(value(model.o))   # 2.0
print(value(model.x))   # 1.0
model.x.value = 2.0
print(value(model.o))   # 3.0
print(value(model.x))   # 2.0
# @:decl2

print("*"*5 + " decl3 ")
# @decl3:
model = ConcreteModel()
model.e = Expression([1,2,3], initialize=0)
model.o = Objective(expr=1.0+summation(model.e))

print(value(model.o))   # 1.0
model.e[2] = 1.0
model.e[3] = 1.0
print(value(model.o))   # 3.0
# @:decl3

print("*"*5 + " decl4 ")
# @decl4:
model = ConcreteModel()
model.x = Var([1,2,3], initialize=1.0)
def e_rule(model, i):
    return i*model.x[i]
model.e = Expression([1,2,3], rule=e_rule)
model.o = Objective(expr=1.0+summation(model.e))

print(value(model.o))   #   7.0
model.x[2] = 2.0
model.x[3] = 2.0
print(value(model.o))   #  12.0
model.e[1] = 100*model.x[1]
print(value(model.o))   # 111.0
# @:decl4

print("*"*5 + " decl5 ")
# @decl5:
model = ConcreteModel()
model.x = Var([1,2,3], initialize=1.0)
e_data = {1: model.x[1], 2:2*model.x[2], 3:3*model.x[3]}
model.e = Expression([1,2,3], initialize=e_data)
model.o = Objective(expr=1.0+summation(model.e))

print(value(model.o))   #   7.0
model.x[2] = 2.0
model.x[3] = 2.0
print(value(model.o))   #  12.0
model.e[1] = 100*model.x[1]
print(value(model.o))   # 111.0
# @:decl5

