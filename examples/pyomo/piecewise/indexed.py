#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

# A piewise approximiation of a nonconvex objective function.

from pyomo.core import *

# Define the function
# Just like in Pyomo constraint rules, a Pyomo model object
# must be the first argument for the function rule
def f(model,t1,t2,x):
    return 0.1*x - cos(5.0*x)


model = ConcreteModel()

# Note we can use an arbitrary number of index sets of 
# arbitrary dimension as the first arguments to the
# Piecewise component.
model.INDEX1 = Set(dimen=2, initialize=[(0,1),(8,3)])
model.X = Var(model.INDEX1, bounds=(-2,2))
model.Z = Var(model.INDEX1)

# For indexed variables, pw_pts must be a
# python dictionary with keys the same as the variable index
PW_PTS = {}

# Increase n to see the solution approach:
# Z[i]=1.19, X[i]=1.89, obj=7.13
n = 3
# Using piecewise representations with a logarithmic number of
# binary variables ('LOG', 'DLOG') requires that pw_pts lists
# must have 2^n + 1 breakpoints.
num_points = 1 + 2**n
step = (2.0 - (-2.0))/(num_points-1)
for idx in model.X.index_set():
    PW_PTS[idx] = [-2.0 + i*step for i in range(num_points)]   # [-2.0, ..., 2.0]

model.linearized_constraint = Piecewise(model.INDEX1,        # indexing sets
                                        model.Z,model.X,     # range and domain variables
                                        pw_pts=PW_PTS,
                                        pw_constr_type='EQ',
                                        pw_repn='LOG',
                                        f_rule=f,
                                        force_pw=True)

# maximize the sum of Z over its index
# This is just a simple example of how to implement indexed variables. All indices
# of Z will have the same solution.
model.obj = Objective(expr= sum_product(model.Z) , sense=maximize)
