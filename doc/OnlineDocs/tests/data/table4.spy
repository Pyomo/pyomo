from pyomo.environ import *

model = AbstractModel()

model.A = Set()
model.Z = Set(dimen=2)

model.M = Param(model.A)
model.N = Param(model.Z)

instance = model.create_instance('table4.dat')
instance.pprint()
