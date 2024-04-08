#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.environ as pyo

# @hierarchy:
model = pyo.ConcreteModel()
model.x = pyo.Var()
model.P = pyo.Param(initialize=5)
model.S = pyo.RangeSet(model.P)
model.b = pyo.Block()
model.b.I = pyo.RangeSet(model.P)
model.b.x = pyo.Var(model.b.I)
model.b.y = pyo.Var(model.S)
model.b.b = pyo.Block([1, 2])
model.b.b[1].x = pyo.Var()
model.b.b[2].x = pyo.Var()
# @:hierarchy

# @hierarchyprint:
print(model.x.local_name)  # x
print(model.x.name)  # x
print(model.b.x.local_name)  # x
print(model.b.x.name)  # b.x
print(model.b.b[1].x.local_name)  # x
print(model.b.b[1].x.name)  # b.b[1].x
# @:hierarchyprint

# @hierarchymove:
model.b.b[1].x.parent_component()  # is model.b.b[1].x
model.b.b[1].x.parent_block()  # is model.b.b[1]
model.b.b[1].x.model()  # is model
model.b.b[1].component('x')  # is model.b.b[1].x
model.b.x[1].parent_component()  # is model.b.x
model.b.x[1].parent_block()  # is model.b
model.b.x[1].model()  # is model
model.b.component('x')  # is model.b.x
# @:hierarchymove

model = None
# @assignment:
new_b = pyo.Block()
new_b.x = pyo.Var()
new_b.P = pyo.Param(initialize=5)
new_b.I = pyo.RangeSet(10)

model = pyo.ConcreteModel()
model.b = new_b
model.x = pyo.Var(model.b.I)
# @:assignment
model.pprint()

model = None
# @blockrule:
model = pyo.ConcreteModel()
model.P = pyo.Param(initialize=3)
model.T = pyo.RangeSet(model.P)


def xyb_rule(b, t):
    b.x = pyo.Var()
    b.I = pyo.RangeSet(t)
    b.y = pyo.Var(b.I)
    b.c = pyo.Constraint(expr=b.x == 1 - sum(b.y[i] for i in b.I))


model.xyb = pyo.Block(model.T, rule=xyb_rule)
# @:blockrule
model.pprint()

model = None
model = pyo.ConcreteModel()
model.P = pyo.Param(initialize=3)
model.T = pyo.RangeSet(model.P)


# @blockrule2:
def xyb_rule(b, t):
    b.x = pyo.Var()
    b.I = pyo.RangeSet(t)
    b.y = pyo.Var(b.I, initialize=1.0)

    def _b_c_rule(_b):
        return _b.x == 1.0 - sum(_b.y[i] for i in _b.I)

    b.c = pyo.Constraint(rule=_b_c_rule)


model.xyb = pyo.Block(model.T, rule=xyb_rule)
# @:blockrule2
model.pprint()

# @blockruleprint:
for t in model.T:
    print(model.xyb[t].c.body)
# @:blockruleprint

# @blockvalues1:
for t in model.xyb:
    for i in model.xyb[t].y:
        print("%s %f" % (model.xyb[t].y[i], pyo.value(model.xyb[t].y[i])))
# @:blockvalues1

# @blockvalues2:
for y in model.xyb[:].y[:]:
    print("%s %f" % (y, pyo.value(y)))
# @:blockvalues2
