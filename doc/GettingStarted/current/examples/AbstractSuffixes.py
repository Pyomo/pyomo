from pyomo.environ import *

model = AbstractModel()
model.I = RangeSet(1,4)
model.x = Var(model.I)
def c_rule(m, i):
    return m.x[i] >= i
model.c = Constraint(model.I, rule=c_rule)

def foo_rule(m):
    return ((m.x[i], 3.0*i) for i in m.I)
model.foo = Suffix(rule=foo_rule)

# instantiate the model
inst = model.create_instance()
for i in inst.I:
    print (i, inst.foo[inst.x[i]])
