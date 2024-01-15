from pyomo.environ import *

model = AbstractModel()

# @decl
model.A = Set(dimen=3)
# @decl

instance = model.create_instance('set2a.dat')

print(sorted(list(instance.A.data())))
