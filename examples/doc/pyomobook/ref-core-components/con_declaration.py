import logging
from pyomo.environ import *

model = ConcreteModel()

model.x = Var([1,2], initialize=1.0)

# @decl1:
model.Diff= Constraint(expr=model.x[2]-model.x[1] <= 7.5)
# @:decl1
model.del_component('Diff')

# @decl2:
def Diff_rule(model):
    return model.x[2] - model.x[1] <= 7.5
model.Diff = Constraint(rule=Diff_rule)
# @:decl2

# @decl3:
N = [1,2,3]

a = {1:1, 2:3.1, 3:4.5}
b = {1:1, 2:2.9, 3:3.1}

model.y = Var(N, within=NonNegativeReals, initialize=0.0)

def CoverConstr_rule(model, i):
    return a[i] * model.y[i] >= b[i]
model.CoverConstr= Constraint(N, rule=CoverConstr_rule)
# @:decl3

# @decl4:
def CapacityIneq_rule(model, i):
    return (0.25, (a[i] * model.y[i])/b[i], 1.0)
model.CapacityIneq = Constraint(N, rule=CapacityIneq_rule)
# @:decl4

# @decl5:
def CapacityEq_rule(model, i):
    return (0, a[i] * model.y[i] - b[i])
model.CapacityEq = Constraint(N, rule=CapacityEq_rule)
# @:decl5

# @decl6:
TimePeriods = [1,2,3,4,5]
LastTimePeriod = 5

model.StartTime = Var(TimePeriods, initialize=1.0)

def Pred_rule(model, t):
    if t == LastTimePeriod:
        return Constraint.Skip
    else:
        return model.StartTime[t] <= model.StartTime[t+1]

model.Pred = Constraint(TimePeriods, rule=Pred_rule)
# @:decl6

logging.disable(logging.ERROR)
# @decl7a:
model.EMPTY = Set()
model.z = Var(model.EMPTY)

@simple_constraint_rule
def C2_rule(model, i):
    if i == 1:
        return summation(model.z) < 1
    if i == 2:
        return summation(model.z) < -1
    return None

try:
    model.C2 = Constraint([1,2,3], rule=C2_rule)
except:
    pass
# @:decl7a

# @decl7b:
def C1_rule(model, i):
    if i == 1:
        return Constraint.Feasible
    if i == 2:
        return Constraint.Infeasible
    return Constraint.Skip

try:
    model.C1 = Constraint([1,2,3], rule=C1_rule)
except:
    pass
# @:decl7b
logging.disable(logging.NOTSET)

# @clist1:
model.c1 = ConstraintList()
model.c1.add(expr=model.x[2] -   model.x[1] <= 7.5)
model.c1.add(expr=model.x[2] - 2*model.x[1] <= 7.5)
print(value(model.c1[1].body))      #  0.0
print(value(model.c1[2].body))      # -1.0
# @:clist1

# @clist2:
def c2_rule(model):
    yield model.x[2] -   model.x[1] <= 7.5
    yield model.x[2] - 2*model.x[1] <= 7.5
    yield ConstraintList.End
model.c2 = ConstraintList(rule=c2_rule)
print(value(model.c2[1].body))      #  0.0
print(value(model.c2[2].body))      # -1.0
# @:clist2

model.display()


# @slack:
model = ConcreteModel()
model.x = Var(initialize=1.0)
model.y = Var(initialize=1.0)

model.c1 = Constraint(expr=        model.y - model.x <= 7.5)
model.c2 = Constraint(expr=-2.5 <= model.y - model.x)
model.c3 = Constraint(expr=-3.0 <= model.y - model.x <= 7.0)

print(value(model.c1.body))     #  0.0

print(model.c1.lslack())        # -inf
print(model.c1.uslack())        #  7.5
print(model.c2.lslack())        # -2.5
print(model.c2.uslack())        #  inf
print(model.c3.lslack())        # -3.0
print(model.c3.uslack())        #  7.0
# @:slack

model.display()
