from pyomo.environ import *

model = AbstractModel()

model.A = Set()
model.Y = Param(model.A)

instance = model.create_instance('import2.tab.dat')

print('A '+str(sorted(list(instance.A.data()))))
print('Y')
keys = instance.Y.keys()
for key in sorted(keys):
    print(str(key)+" "+str(value(instance.Y[key])))
