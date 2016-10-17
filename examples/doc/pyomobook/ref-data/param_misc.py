from pyomo.environ import *

# @mutable1:
model = ConcreteModel()
p = {1:1, 2:4, 3:9}

model.A = Set(initialize=[1,2,3])
model.p = Param(model.A, initialize=p)
model.x = Var(model.A, within=NonNegativeReals)

model.o = Objective(expr=summation(model.p, model.x))
# @:mutable1
model.pprint()

# @mutable2:
model = ConcreteModel()
p = {1:1, 2:4, 3:9}

model.A = Set(initialize=[1,2,3])
model.p = Param(model.A, initialize=p, mutable=True)
model.x = Var(model.A, within=NonNegativeReals)

model.o = Objective(expr=summation(model.p, model.x))

model.p[2] = 1
model.p[3] = 1
# @:mutable2
model.pprint()

