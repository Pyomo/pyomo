from pyomo.environ import *

model = AbstractModel()

model.A = Set()
model.B = Set()
model.C = Set()

instance = model.create_instance('set1.dat')

print(sorted(list(instance.A.data())))
print(sorted((instance.B.data())))
print(sorted(list((instance.C.data())), key=lambda x:x if type(x) is str else str(x)))
