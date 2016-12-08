from pyomo.environ import *

model = AbstractModel()

model.C = Set(dimen=2)

instance = model.create_instance('import4.tab.dat')

print('C '+str(sorted(list(instance.C.data()))))
