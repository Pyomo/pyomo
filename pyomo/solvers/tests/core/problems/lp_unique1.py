# This problem has a unique primal and dual solution.

from pyomo.core import *

model = ConcreteModel()

model.n = Param(default=7)
model.m = Param(default=7)

model.N = RangeSet(1,model.n)
model.M = RangeSet(1,model.m)

def c_rule(model, j):
    return 5 if j<5 else 9.0/2
model.c = Param(model.N)

def b_rule(model, i):
    if i == 4:
        i = 5
    elif i == 5:
        i = 4
    return 5 if i<5 else 7.0/2
model.b = Param(model.M)

def A_rule(model, i, j):
    if i == 4:
        i = 5
    elif i == 5:
        i = 4
    return 2 if i==j else 1
model.A = Param(model.M, model.N)

model.x = Var(model.N, within=NonNegativeReals)
model.y = Var(model.M, within=NonNegativeReals)

model.cost = Objective(expr=sum_product(model.c, model.x))

def primalcon_rule(model, i):
    return sum(model.A[i,j]*model.x[j] for j in model.N) >= model.b[i]
model.primalcon = Constraint(model.M)

model.dual = Suffix(direction=Suffix.IMPORT)
model.rc = Suffix(direction=Suffix.IMPORT)
model.slack = Suffix(direction=Suffix.IMPORT)
model.urc = Suffix(direction=Suffix.IMPORT)
model.lrc = Suffix(direction=Suffix.IMPORT)

