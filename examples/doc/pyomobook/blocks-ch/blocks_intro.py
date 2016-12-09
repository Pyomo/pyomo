from pyomo.environ import *

# @hierarchy:
model = ConcreteModel()
model.x = Var()
model.P = Param(initialize=5)
model.S = RangeSet(model.P)
model.b = Block()
model.b.I = RangeSet(model.P)
model.b.x = Var(model.b.I)
model.b.y = Var(model.S)
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

model = None
# @assignment:
new_b = Block()
new_b.x = Var()
new_b.P = Param(initialize=5)
new_b.I = RangeSet(10)

model = ConcreteModel()
model.b = new_b
model.x = Var(model.b.I)
# @:assignment
model.pprint()

model = None
# @blockrule:
model = ConcreteModel()
model.P = Param(initialize=3)
model.T = RangeSet(model.P)

def xyb_rule(b, t):
    b.x = Var()
    b.I = RangeSet(t)
    b.y = Var(b.I)
    b.c = Constraint(expr = b.x == 1.0 - sum(b.y[i] for i in b.I))
model.xyb = Block(model.T, rule=xyb_rule)
# @:blockrule
model.pprint()

model = None
model = ConcreteModel()
model.P = Param(initialize=3)
model.T = RangeSet(model.P)
# @blockrule2:
def xyb_rule(b, t):
    b.x = Var()
    b.I = RangeSet(t)
    b.y = Var(b.I, initialize=1.0)
    def _b_c_rule(_b):
        return _b.x == 1.0 - sum(_b.y[i] for i in _b.I)
    b.c = Constraint(rule=_b_c_rule)
model.xyb = Block(model.T, rule=xyb_rule)
# @:blockrule2
model.pprint()

# @blockruleprint:
for t in model.T:
    print(model.xyb[t].c.body)
# @:blockruleprint

# @blockvalues1:
for t in model.xyb:
  for i in model.xyb[t].y:
    print("%s %f" % (model.xyb[t].y[i], value(model.xyb[t].y[i])))
# @:blockvalues1

# @blockvalues2:
for y in model.xyb[:].y[:]:
    print("%s %f" % (y, value(y)))
# @:blockvalues2
