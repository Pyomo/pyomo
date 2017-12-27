from pyomo.environ import *

model = AbstractModel()

# @decl
model.A = Set()
model.B = Param(model.A)
model.C = Param(model.A)
model.D = Param(model.A)
# @decl

instance = model.create_instance('param3a.dat')

print('B')
keys = instance.B.keys()
for key in sorted(keys):
    print(str(key)+" "+str(value(instance.B[key])))
print('C')
keys = instance.C.keys()
for key in sorted(keys):
    print(str(key)+" "+str(value(instance.C[key])))
print('D')
keys = instance.D.keys()
for key in sorted(keys):
    print(str(key)+" "+str(value(instance.D[key])))
