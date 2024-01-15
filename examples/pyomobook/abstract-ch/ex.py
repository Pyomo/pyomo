import pyomo.environ as pyo

model = pyo.AbstractModel()

# @decl:
model.z = pyo.Param()
# @:decl

instance = model.create_instance('ex.dat')

print(pyo.value(instance.z))
