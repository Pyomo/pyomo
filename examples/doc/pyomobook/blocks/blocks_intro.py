from pyomo.environ import *

# @hierarchy:
model = ConcreteModel()
model.x = Var()
model.NumVars = Param(initialize=5)
model.b = Block()
model.b.I = RangeSet(model.NumVars)
model.b.x = Var(model.b.I)
model.b.b = Block()
model.b.b.x = Var()
# @:hierarchy

# @hierarchyprint:
print(model.x.local_name)     # x
print(model.x.name)           # x
print(model.b.x.local_name)   # x
print(model.b.x.name)         # b.x
print(model.b.b.x.local_name) # x
print(model.b.b.x.name)       # b.b.x
# @:hierarchyprint
