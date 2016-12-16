from pyomo.core import *

# @Model:
model = ConcreteModel()

model.x = Var(initialize=1.0, bounds=(0,1))
model.y = Var(initialize=3.0, bounds=(2,4))
model.o = Objective(expr=model.x+model.y)
# @:Model


print("# @display1:")
display(model)
print("# @:display1")

print("# @display2:")
display(model.x)
print("# @:display2")

print("# @value1:")
print(value(model.o))
print("# @:value1")

print("# @value2:")
print(value(model.x))
print("# @:value2")
