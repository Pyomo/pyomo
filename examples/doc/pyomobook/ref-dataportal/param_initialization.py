from pyomo.environ import *
import numpy

model = ConcreteModel()

# @decl1:
model.a = Param(initialize=1.1)
# @:decl1

# Initialize with a dictionary
# @decl2:
model.b = Param([1,2,3], initialize={1:1, 2:2, 3:3})
# @:decl2

# Initialize with a function that returns native Python data
# @decl3:
def c(model):
    return {1:1, 2:2, 3:3}
model.c = Param([1,2,3], initialize=c)
# @:decl3

model.pprint(verbose=True)
