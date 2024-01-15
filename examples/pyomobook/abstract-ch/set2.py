import pyomo.environ as pyo

model = pyo.AbstractModel()

# @decl:
model.A = pyo.Set(dimen=3)
# @:decl

instance = model.create_instance('set2.dat')

print(sorted(list(instance.A.data())))
