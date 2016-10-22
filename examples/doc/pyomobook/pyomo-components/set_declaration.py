from pyomo.environ import *

model = AbstractModel()

# @decl1:
model.A = Set()
# @:decl1

instance = model.create_instance('set_declaration.dat')
instance.pprint()
model = AbstractModel()

# @decl2:
model.A = Set()
model.B = Set()
model.C = Set(model.A)
model.D = Set(model.A,model.B)
# @:decl2

model = AbstractModel()
instance = model.create_instance('set_declaration.dat')
instance.pprint()
model = AbstractModel()

# @decl6:
model.E = Set([1,2,3])
f = set([1,2,3])
model.F = Set(f)
# @:decl6

instance = model.create_instance('set_declaration.dat')
instance.pprint()
model = AbstractModel()
model = AbstractModel()

# @decl3:
model.A = Set()
model.B = Set()
model.G = model.A | model.B    # set union
model.H = model.B & model.A    # set intersection
model.I = model.A - model.B    # set difference
model.J = model.A ^ model.B    # set exclusive-or
# @:decl3

instance = model.create_instance('set_declaration.dat')
instance.pprint()
model = AbstractModel()

# @decl4:
model.A = Set()
model.B = Set()
model.K = model.A * model.B
# @:decl4
# Removing this component, which we're going to read again
model.del_component('K')

instance = model.create_instance('set_declaration.dat')
instance.pprint(verbose=True)
