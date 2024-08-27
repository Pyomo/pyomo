#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

# driveabs2.py

import pyomo.environ as pyo
from pyomo.opt import SolverFactory

# Create a solver
opt = SolverFactory('cplex')

# get the model from another file
from abstract2 import model

# Create a model instance and optimize
instance = model.create_instance('abstract2.dat')

# @Create_dual_suffix_component
# Create a 'dual' suffix component on the instance
# so the solver plugin will know which suffixes to collect
instance.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
# @Create_dual_suffix_component

results = opt.solve(instance)
# also puts the results back into the instance for easy access

# @Access_all_dual
# display all duals
print("Duals")
for c in instance.component_objects(pyo.Constraint, active=True):
    print("   Constraint", c)
    for index in c:
        print("      ", index, instance.dual[c[index]])
# @Access_all_dual

# @Access_one_dual
# access one dual
print("Dual for Film=", instance.dual[instance.AxbConstraint['Film']])
# @Access_one_dual
