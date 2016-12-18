from pyomo.environ import *

model = AbstractModel()

model.A = Set(initialize=['A1', 'A2', 'A3', 'A4'])
model.Y = Param(model.A)

instance = model.create_instance('import1.tab.dat')

print('Y')
keys = instance.Y.keys()
for key in sorted(keys):
    print(str(key)+" "+str(value(instance.Y[key])))
