from pyomo.environ import *

model = ConcreteModel()

# @decl1:
model.x = Var([1,2], initialize=1.0)
model.diff = Constraint(expr=model.x[2]-model.x[1] <= 7.5)
# @:decl1

model.pprint()
model = None
model = ConcreteModel()

# @decl2:
model.x = Var([1,2], initialize=1.0)
def diff_rule(model):
    return model.x[2] - model.x[1] <= 7.5
model.diff = Constraint(rule=diff_rule)
# @:decl2

model.pprint()
model = None
model = ConcreteModel()

# @decl3:
N = [1,2,3]

a = {1:1, 2:3.1, 3:4.5}
b = {1:1, 2:2.9, 3:3.1}

model.y = Var(N, within=NonNegativeReals, initialize=0.0)

def CoverConstr_rule(model, i):
    return a[i] * model.y[i] >= b[i]
model.CoverConstr = Constraint(N, rule=CoverConstr_rule)
# @:decl3

model.pprint()
model = None
model = ConcreteModel()

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

model.pprint()
model = None

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
print(model.c2.lslack())        #  2.5
print(model.c2.uslack())        #  inf
print(model.c3.lslack())        #  3.0
print(model.c3.uslack())        #  7.0
# @:slack

model.display()
