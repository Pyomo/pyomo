from pyomo.environ import *
import pyomo.common
import sys

model = AbstractModel()

model.Z = Set(dimen=3)
model.Y = Param(model.Z)

try:
    instance = model.create_instance('ABCD7.dat')
except pyomo.common.errors.ApplicationError as e:
    print("ERROR " + str(e))
    sys.exit(1)

print('Z ' + str(sorted(list(instance.Z.data()))))
print('Y')
for key in sorted(instance.Y.keys()):
    print(name(instance.Y, key) + " " + str(value(instance.Y[key])))
