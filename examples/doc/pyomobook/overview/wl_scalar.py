# wl_scalar.py: snippets that show the warehouse location problem implemented as scalar quantities
from pyomo.environ import *

model = ConcreteModel()

# @vars:
model.x_Harlingen_NYC = Var(bounds=(0,1))
model.x_Harlingen_LA = Var(bounds=(0,1))
model.x_Harlingen_Chicago = Var(bounds=(0,1))
model.x_Harlingen_Houston = Var(bounds=(0,1))
model.x_Memphis_NYC = Var(bounds=(0,1))
model.x_Memphis_LA = Var(bounds=(0,1))
#...
# @:vars
model.x_Memphis_Chicago = Var(bounds=(0,1))
model.x_Memphis_Houston = Var(bounds=(0,1))
model.x_Ashland_NYC = Var(bounds=(0,1))
model.x_Ashland_LA = Var(bounds=(0,1))
model.x_Ashland_Chicago = Var(bounds=(0,1))
model.x_Ashland_Houston = Var(bounds=(0,1))

# @cons:
model.one_warehouse_for_NYC = Constraint(expr=model.x_Harlingen_NYC + model.x_Memphis_NYC + model.x_Ashland_NYC == 1)

model.one_warehouse_for_LA = Constraint(expr=model.x_Harlingen_LA + model.x_Memphis_LA + model.x_Ashland_LA == 1)
#...
# @:cons

# @maxY:
model.maxY = Constraint(expr=model.y_Harlingen + model.y_Memphis + model.y_Ashland <= P)
# @:maxY
model.pprint()
