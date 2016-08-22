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
print(model.x.name(fully_qualified=False))     # x
print(model.x.name(fully_qualified=True))      # x
print(model.b.x.name(fully_qualified=False))   # x
print(model.b.x.name(fully_qualified=True))    # b.x
print(model.b.b.x.name(fully_qualified=False)) # x
print(model.b.b.x.name(fully_qualified=True))  # b.b.x
# @:hierarchyprint
