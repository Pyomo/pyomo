from pyomo.environ import *

model = AbstractModel()

model.c1 = Set(dimen=3)
model.Z = Param(model.c1)

instance = model.create_instance('example1.tab.dat')

print("c1 "+" "+str(sorted(list(instance.c1.data()))))
print('Z')
keys = instance.Z.keys()
for key in sorted(keys):
    print(str(key)+" "+str(value(instance.Z[key])))
