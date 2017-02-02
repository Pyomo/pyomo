from pyomo.environ import *

model = ConcreteModel()
model.u = Var(initialize=1.0)
model.v = Var(initialize=1.0)

# comparing values
print(value(model.u) == value(model.v)) # True

# comparing variables
print(model.u == model.v) # "u  ==  v"

# following prints "Same"
if model.u == model.v:
    print('Same')
else:
    print('Different')
