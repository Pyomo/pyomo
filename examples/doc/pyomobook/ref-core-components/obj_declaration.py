from pyomo.environ import *

model = ConcreteModel()

model.x = Var([1,2], initialize=1.0)

# @decl1:
model.a = Objective()
# @:decl1

# @decl2:
model.b = Objective()
model.c = Objective([1,2,3])
# @:decl2

# @decl3:
model.d = Objective(expr=model.x[1] + 2*model.x[2])
# @:decl3

# @decl4:
model.e = Objective(expr=model.x[1], sense=maximize)
# @:decl4

# @decl5:
model.f = Objective(expr=model.x[1] + 2*model.x[2])

def gg_rule(model):
    return model.x[1] + 2*model.x[2]
model.gg = Objective(rule=gg_rule)
# @:decl5

# @decl6:
def h_rule(model, i):
    return i*model.x[1] + i*i*model.x[2]
model.h = Objective([1, 2, 3, 4], rule=h_rule)
# @:decl6

# @decl6a:
def hh_rule(model, i):
    if i == 3:
        return Objective.Skip
    return i*model.x[1] + i*i*model.x[2]
model.hh = Objective([1, 2, 3, 4], rule=hh_rule)
# @:decl6a

# @decl7:
def m_rule(model):
    expr = model.x[1]
    expr += 2*model.x[2]
    return expr
model.m = Objective(rule=m_rule)
# @:decl7

# @decl8:
p = 0.6
def n_rule(model):
    if p > 0.5:
        return model.x[1] + 2*model.x[2]
    else:
        return model.x[1] + 3*model.x[2]
    return expr
model.n = Objective(rule=n_rule)
# @:decl8

# @decl9:
p = 0.6
if p > 0.5:
    model.p = Objective(expr=model.x[1] + 2*model.x[2])
else:
    model.p = Objective(expr=model.x[1] + 3*model.x[2])
# @:decl9

# @value:
model.o = Objective(expr=model.x[1] + 2*model.x[2])
print(value(model.o))       # 3
print(model.o())            # 3
# @:value

# @olist1:
model.r = ObjectiveList()
model.r.add(expr=model.x[1] + 2*model.x[2])
model.r.add(expr=model.x[1] + 3*model.x[2])
print(value(model.r[1]))    # 3
print(value(model.r[2]))    # 4
# @:olist1

# @olist2:
def rr_rule(model):
    yield model.x[1] + 2*model.x[2]
    yield model.x[1] + 3*model.x[2]
    yield ObjectiveList.End
model.rr = ObjectiveList(rule=rr_rule)
print(value(model.rr[1]))    # 3
print(value(model.rr[2]))    # 4
# @:olist2

model.display()
