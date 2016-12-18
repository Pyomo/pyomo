from pyomo.environ import *

model = AbstractModel()

# @decl1:
model.Z = Param(within=Reals)
# @:decl1

# @decl2:
def Y_validate(model, value):
    return value in Reals
model.Y = Param(validate=Y_validate)
# @:decl2

# @decl3:
model.A = Set(initialize=[1,2,3])
def X_validate(model, value, i):
    return value > i
model.X = Param(model.A, validate=X_validate)
# @:decl3

instance = model.create_instance('param_validation.dat')
instance.pprint()
