from pyomo.environ import *

model = AbstractModel()

model.A = Set()
model.B = Set(initialize=['B1','B2','B3'])
model.Z = Set(dimen=2)

model.M = Param(model.A)
model.N = Param(model.A, model.B)


instance = model.create_instance('table3.ul.dat')
instance.pprint()
