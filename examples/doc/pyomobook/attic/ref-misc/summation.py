import pyomo.environ
from pyomo.core import *

model = ConcreteModel()

# @components:
model.N = Set(initialize=[1,2,3])
model.M = Set(initialize=[1,3])

model.a = Param(model.N, initialize={1:1, 2:3.1, 3:4.5})

model.x = Var(model.N, within=NonNegativeReals)
model.y = Var(model.N, within=NonNegativeReals)
model.z = Var(model.M, within=NonNegativeReals)
#@:components

# @sum1:
summation(model.x)
# @:sum1

# @c1:
model.c1 = Constraint(expr=
                sum(model.x[i] for i in model.N) <= 0)
# @:c1

# @c2:
model.c2 = Constraint(expr=summation(model.x) <= 0)
# @:c2

# @o1:
model.o1 = Objective(expr=summation(model.a, model.x))
# @:o1

# @o2:
model.o2 = Objective(expr=summation(model.x, denom=model.y))
# @:o2

# @o3:
model.o3 = Objective(expr=
                summation(model.x, denom=(model.a, model.y)))
# @:o3

# @o4:
model.o4 = Objective(expr=
                summation(model.x, model.z, denom=model.a))
# @:o4

# @o5:
model.o5 = Objective(expr=
                summation(denom=(model.x, model.z)))
# @:o5

# @o6:
model.o6 = Objective(expr=
                summation(model.x, model.y, index=model.M))
# @:o6

instance = model
model.pprint()
