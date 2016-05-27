from pyomo.environ import *

model = AbstractModel()

model.Z = Set(dimen=4)

instance = model.create_instance('ABCD1.dat')

print(sorted(list(instance.Z.data())))
