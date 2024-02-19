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

from pyomo.core import *
import pyomo.opt
import pyomo.environ

#
# Import model
import knapsack

#
# Create the model instance
instance = knapsack.model.create_instance("knapsack.dat")
#
# Setup the optimizer
opt = pyomo.opt.SolverFactory("glpk")
#
# Optimize
results = opt.solve(instance, symbolic_solver_labels=True)
instance.solutions.store_to(results)
#
# Write the output
results.write(num=1)
