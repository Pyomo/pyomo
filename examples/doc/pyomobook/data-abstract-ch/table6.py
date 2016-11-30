from pyomo.environ import *

model = AbstractModel()

model.pi = Param()

instance = model.create_instance('table6.dat')
instance.pprint()
