from pyomo.environ import *

model = AbstractModel()

model.A = Set(initialize=['A1','A2','A3'])
model.B = Set(initialize=['B1','B2','B3'])

model.M = Param(model.A)
model.N = Param(model.A, model.B)

instance = model.create_instance('table2.dat')
instance.pprint()
