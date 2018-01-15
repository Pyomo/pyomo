from pyomo.environ import *

model = AbstractModel()

# @decl
model.A = Set()
model.B = Param(model.A)
# @decl

instance = model.create_instance('param2.dat')

keys = instance.B.keys()
for key in sorted(keys):
    print(str(key)+" "+str(value(instance.B[key])))
