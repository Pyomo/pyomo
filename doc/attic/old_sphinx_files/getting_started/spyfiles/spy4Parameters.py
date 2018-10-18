"""
David L. Woodruff and Mingye Yang, Spring 2018
Code snippets for Paramters.rst in testable form
"""

from pyomo.environ import *

model = AbstractModel()

# @ABPSets
model.A = Set()
model.B = Set()
model.P = Param(model.A, model.B)
# @ABPSets

# @Param_python
v={}
v[1,1] = 9
v[2,2] = 16
v[3,3] = 25
model.S = Param(model.A, model.A, initialize=v, default=0)
# @Param_python

# @Param_def
def s_init(model, i, j):
    if i == j:
        return i*i
    else:
        return 0.0
model.S1 = Param(model.A, model.A, initialize=s_init)
# @Param_def

# @Valuecheck_validation
def s_validate(model, v, i):
    return v > 3.14159
model.s = Param(model.A, validate=s_validate)
# @Valuecheck_validation

# to make it testable
instance = model.create_instance("spy4Parameters.dat")
instance.pprint()
