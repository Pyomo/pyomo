import pyomo.environ as pyo

model = pyo.AbstractModel()

# @decl:
model.A = pyo.Set(dimen=2)
# @:decl

instance = model.create_instance('set4.dat')

print(sorted(list(instance.A.data())))
