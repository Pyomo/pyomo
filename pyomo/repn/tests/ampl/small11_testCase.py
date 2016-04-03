#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________
#
# Author:  Gabe Hackebeil
# Purpose: For regression testing to ensure that the Pyomo
#          NL writer properly labels constraint ids in the "J"
#          section of the NL file when trivial constraints exist.
#          At the creation of this test, trivial constraints 
#          (constraints with no variables) are being written to 
#          the nl file as a feasibility check for the user.
#
#          This test model relies on the asl_test executable. It
#          will not solve if sent to a real optimizer.
#

from pyomo.environ import *
model = ConcreteModel()

n=3

model.x = Var([(k,i) for k in range(1,n+1) for i in range(k,n+1)])

def obj_rule(model):
	return model.x[n,n]
model.obj = Objective(rule=obj_rule)

def var_bnd_rule(model,i):
	return -1.0 <= model.x[1,i] <= 1.0
model.var_bnd = Constraint(RangeSet(1,n),rule=var_bnd_rule)

model.x[1,1] = 1.0
model.x[1,1].fixed = True
