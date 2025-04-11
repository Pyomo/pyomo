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

# A Suffix example for ipopt.
# Translated to Pyomo from AMPL model source at:
#   https://projects.coin-or.org/Ipopt/wiki/IpoptAddFeatures
#
# This Pyomo example is formulated as a python script.
# To run this script execute the following command:
#
# $ pyomo_python ipopt_warmstart.py
#
# Execution of this script requires that the ipopt
# solver is in the current search path for executables
# on this system. This example was tested using Ipopt
# version 3.10.2

import pyomo.environ as pyo

### Create the ipopt solver plugin using the ASL interface
solver = 'ipopt'
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
###

### Create the example model
model = pyo.ConcreteModel()
model.x1 = pyo.Var(bounds=(1, 5), initialize=1.0)
model.x2 = pyo.Var(bounds=(1, 5), initialize=5.0)
model.x3 = pyo.Var(bounds=(1, 5), initialize=5.0)
model.x4 = pyo.Var(bounds=(1, 5), initialize=1.0)
model.obj = pyo.Objective(
    expr=model.x1 * model.x4 * (model.x1 + model.x2 + model.x3) + model.x3
)
model.inequality = pyo.Constraint(
    expr=model.x1 * model.x2 * model.x3 * model.x4 >= 25.0
)
model.equality = pyo.Constraint(
    expr=model.x1**2 + model.x2**2 + model.x3**2 + model.x4**2 == 40.0
)
###

### Declare all suffixes
# Ipopt bound multipliers (obtained from solution)
model.ipopt_zL_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
model.ipopt_zU_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
# Ipopt bound multipliers (sent to solver)
model.ipopt_zL_in = pyo.Suffix(direction=pyo.Suffix.EXPORT)
model.ipopt_zU_in = pyo.Suffix(direction=pyo.Suffix.EXPORT)
# Obtain dual solutions from first solve and send to warm start
model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)
###

### Send the model to ipopt and collect the solution
print("")
print("INITIAL SOLVE")
results = opt.solve(model, keepfiles=keepfiles, tee=stream_solver)
###

### Print Solution
print("   %7s %12s %12s" % ("Value", "ipopt_zL_out", "ipopt_zU_out"))
for v in [model.x1, model.x2, model.x3, model.x4]:
    print(
        "%s %7g %12g %12g"
        % (v, pyo.value(v), model.ipopt_zL_out[v], model.ipopt_zU_out[v])
    )
print("inequality.dual = " + str(model.dual[model.inequality]))
print("equality.dual   = " + str(model.dual[model.equality]))
###


### Set Ipopt options for warm-start
# The current values on the ipopt_zU_out and
# ipopt_zL_out suffixes will be used as initial
# conditions for the bound multipliers to solve
# the new problem
model.ipopt_zL_in.update(model.ipopt_zL_out)
model.ipopt_zU_in.update(model.ipopt_zU_out)
opt.options['warm_start_init_point'] = 'yes'
opt.options['warm_start_bound_push'] = 1e-6
opt.options['warm_start_mult_bound_push'] = 1e-6
opt.options['mu_init'] = 1e-6
###

### Send the model and suffix data to ipopt and collect the solution
print("")
print("WARM-STARTED SOLVE")
# The solver plugin will scan the model for all active suffixes
# valid for importing, which it will store into the results object
results = opt.solve(model, keepfiles=keepfiles, tee=stream_solver)
###

### Print Solution
print("   %7s %12s %12s" % ("Value", "ipopt_zL_out", "ipopt_zU_out"))
for v in [model.x1, model.x2, model.x3, model.x4]:
    print(
        "%s %7g %12g %12g"
        % (v, pyo.value(v), model.ipopt_zL_out[v], model.ipopt_zU_out[v])
    )
print("inequality.dual = " + str(model.dual[model.inequality]))
print("equality.dual   = " + str(model.dual[model.equality]))
###
