import pyomo.environ as pyo

model = pyo.AbstractModel()

model.A = pyo.Set()
# @decl1:
model.B = pyo.Set(within=model.A)
# @:decl1

# @decl2:
def C_validate(model, value):
    return value in model.A
model.C = pyo.Set(validate=C_validate)
# @:decl2

instance = model.create_instance('set_validation.dat')
instance.pprint(verbose=True)
