from pyomo.environ import *

model = AbstractModel()

model.Z = Set(dimen=2)
model.Y = Set(dimen=2)

instance = model.create_instance('table5.dat')
instance.pprint()
