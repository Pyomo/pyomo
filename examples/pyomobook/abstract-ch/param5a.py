import pyomo.environ as pyo

model = pyo.AbstractModel()

# @decl:
model.A = pyo.Set(dimen=2)
model.B = pyo.Param(model.A)
# @:decl

instance = model.create_instance('param5a.dat')

keys = instance.B.keys()
for key in sorted(keys):
    print(str(key)+" "+str(pyo.value(instance.B[key])))
