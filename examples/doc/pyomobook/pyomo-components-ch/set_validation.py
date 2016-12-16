from pyomo.environ import *

model = AbstractModel()

model.A = Set()
# @decl1:
model.B = Set(within=model.A)
# @:decl1

# @decl2:
def C_validate(model, value):
    return value in model.A
model.C = Set(validate=C_validate)
# @:decl2

# @decl5:
model.F = Set([1,2,3], within=model.A)
# @:decl5

# @decl6:
def G_validate(model, value):
    return value in model.A
model.G = Set([1,2,3], validate=G_validate)
# @:decl6

instance = model.create_instance('set_validation.dat')
instance.pprint(verbose=True)
