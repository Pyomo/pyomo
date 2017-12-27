from pyomo.environ import *

model = AbstractModel()

# @decl
model.z = Param()
# @decl

instance = model.create_instance('ex.dat')

print(value(instance.z))
