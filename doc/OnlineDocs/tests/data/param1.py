from pyomo.environ import *

model = AbstractModel()

# @decl
model.A = Param()
model.B = Param()
model.C = Param()
model.D = Param()
model.E = Param()
# @decl

instance = model.create_instance('param1.dat')

print(value(instance.A))
print(value(instance.B))
print(value(instance.C))
print(value(instance.D))
print(value(instance.E))
