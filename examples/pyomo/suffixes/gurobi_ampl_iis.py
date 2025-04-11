#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

# A Suffix example for the gurobi_ampl solver.
#
# This Pyomo example is formulated as a python script.
# To run this script execute the following command:
#
# $ pyomo_python gurobi_ampl_iis.py
#
# Execution of this script requires that the gurobi_ampl
# solver is in the current search path for executables
# on this system. This example was tested using Gurobi
# Solver 5.0.0

import pyomo.environ as pyo

### Create the gurobi_ampl solver plugin using the ASL interface
solver = 'gurobi_ampl'
solver_io = 'nl'
stream_solver = False  # True prints solver output to screen
keepfiles = False  # True prints intermediate file names (.nl,.sol,...)
opt = pyo.SolverFactory(solver, solver_io=solver_io)

if opt is None:
    print("")
    print(
        "ERROR: Unable to create solver plugin for %s "
        "using the %s interface" % (solver, solver_io)
    )
    print("")
    exit(1)

# tell gurobi to be verbose with output
opt.options['outlev'] = 1

# tell gurobi to find an iis table for the infeasible model
opt.options['iisfind'] = 1  # tell gurobi to be verbose with output

### Create a trivial and infeasible example model
model = pyo.ConcreteModel()
model.x = pyo.Var(within=pyo.NonNegativeReals)
model.obj = pyo.Objective(expr=model.x)
model.con = pyo.Constraint(expr=model.x <= -1)
###

# Create an IMPORT Suffix to store the iis information that will
# be returned by gurobi_ampl
model.iis = pyo.Suffix(direction=pyo.Suffix.IMPORT)

### Send the model to gurobi_ampl and collect the solution
# The solver plugin will scan the model for all active suffixes
# valid for importing, which it will store into the results object
results = opt.solve(model, keepfiles=keepfiles, tee=stream_solver)

print("")
print("IIS Results")
for component, value in model.iis.items():
    print(component.name + " " + str(value))
