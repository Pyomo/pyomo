import pyomo.environ as pyo

model = pyo.AbstractModel()

# @decl:
model.A = pyo.Set()
model.B = pyo.Param(model.A)
# @:decl

instance = model.create_instance('param4.dat')

print('B')
keys = instance.B.keys()
for key in sorted(keys):
    print(str(key)+" "+str(pyo.value(instance.B[key])))
