from pyomo.environ import *

model = AbstractModel()

# @decl
model.A = Set(dimen=2)
# @decl

instance = model.create_instance('set4.dat')

print(sorted(list(instance.A.data())))
