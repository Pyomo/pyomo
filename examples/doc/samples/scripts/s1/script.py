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
results = opt.solve(instance,symbolic_solver_labels=True)
instance.solutions.store_to(results)
#
# Write the output
results.write(num=1)
