from pyomo.environ import *

model = AbstractModel()

model.A = Set(initialize=['A1','A2','A3'])
model.M = Param(model.A)

instance = model.create_instance('table0.dat')
instance.pprint()
