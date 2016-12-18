from pyomo.environ import *

model = AbstractModel()

# @decl1:
model.A = RangeSet(10)
# @:decl1

# @decl3:
model.C = RangeSet(5,10)
# @:decl3

# @decl4:
model.D = RangeSet(2.5,11,1.5)
# @:decl4

instance = model.create_instance()
instance.pprint(verbose=True)
