# scont.py
from pyomo.environ import *
from pyomo.gdp import *

L = [1,2,3]
U = [2,4,6]
index = [0,1,2]

model = ConcreteModel()
model.x = Var(index, within=Reals, bounds=(0,20))

# Each disjunct is a semi-continuous variable
#    x[k] == 0 or L[k] <= x[k] <= U[k]
def d_rule(block, k, i):
    m = block.model()
    if i == 0:
        block.c = Constraint(expr=m.x[k] == 0)
    else:
        block.c = Constraint(expr=L[k] <= m.x[k] <= U[k])
model.d = Disjunct(index, [0,1], rule=d_rule)

# There are three disjunctions
def D_rule(block, k):
    model = block.model()
    return [model.d[k,0], model.d[k,1]]
model.D = Disjunction(index, rule=D_rule)

# Minimize the number of x variables that are nonzero
model.o = Objective(expr=sum(model.d[k,1].indicator_var for k in index))

# Satisfy a demand that is met by these variables
model.c = Constraint(expr=sum(model.x[k] for k in index) >= 7)
