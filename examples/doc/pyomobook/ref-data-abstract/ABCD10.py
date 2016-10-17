from pyomo.environ import *
import pyutilib.common
import sys

model = AbstractModel()

model.Z = Set(dimen=3)
model.Y = Param(model.Z)

try:
    instance = model.create_instance('ABCD10.dat')
except pyutilib.common.ApplicationError, e:
    print("ERROR "+str(e))
    sys.exit(1)

print('Z '+sorted(list(instance.Z.data())))
print('Y')
for key in sorted(instance.Y.keys()):
    print(instance.Y[key]+" "+value(instance.Y[key]))
