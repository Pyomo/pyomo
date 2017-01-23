from pyomo.environ import *

model = AbstractModel()

# @decl1:
model.A = Set(ordered=True)
# @:decl1

instance = model.create_instance('set_options.dat')
instance.pprint()
