from pyomo.environ import *

model = AbstractModel()

# @decl1:
model.A = Set()
# @:decl1

# @decl2:
model.B = Set()
model.C = Set(model.A)
model.D = Set(model.A,model.B)
# @:decl2

# @decl6:
model.E = Set([1,2,3])
f = set([1,2,3])
model.F = Set(f)
# @:decl6

# @decl3:
model.G = model.A | model.B    # set union
model.H = model.B & model.A    # set intersection
model.I = model.A - model.B    # set difference
model.J = model.A ^ model.B    # set exclusive-or
# @:decl3

# @decl4:
model.K = model.A * model.B
# @:decl4
# Removing this component, which we're going to read again
model.del_component('K')

# @decl5:
model.L = Set(within=model.A * model.B)
# @:decl5

# @decl7:
model.K = model.A * model.B
model.K2 = Set(dimen=2)

model.DFirst = Set(model.A, model.B)
model.DSecond = Set(model.K)
model.D2 = Set(model.K2)
# @:decl7

instance = model.create_instance('set_declaration.dat')
instance.pprint(verbose=True)
