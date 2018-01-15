from pyomo.environ import *

model = AbstractModel()

model.A = Set(initialize=['A1','A2','A3'])
model.M = Param(model.A)
model.Z = Set(dimen=2)

instance = model.create_instance('table7.dat')
instance.pprint()
