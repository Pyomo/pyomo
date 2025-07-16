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

# Author: Hans Pirnay, 2012-11-27
# The parametric.mod example from sIPOPT
#
# This Pyomo example is formulated as a python script.
# To run this script execute the following command:
#
# $ pyomo_python sipopt_parametric.py
#
# Execution of this script requires that the ipopt_sens
# solver (distributed with Ipopt) is in the current search
# path for executables on this system.

import pyomo.environ as pyo

### Create the ipopt_sens solver plugin using the ASL interface
solver = 'ipopt_sens'
solver_io = 'nl'
stream_solver = True  # True prints solver output to screen
keepfiles = False  # True prints intermediate file names (.nl,.sol,...)
opt = pyo.SolverFactory(solver, solver_io=solver_io)
###

if opt is None:
    print("")
    print("ERROR: Unable to create solver plugin for 'ipopt_sens'")
    print("")
    exit(1)

### Set this data
nominal_eta1 = 4.5
perturbed_eta1 = 4.0
nominal_eta2 = 1.0
perturbed_eta2 = 1.0

### Create the model
model = pyo.ConcreteModel()
# variables
model.x1 = pyo.Var(initialize=0.15, within=pyo.NonNegativeReals)
model.x2 = pyo.Var(initialize=0.15, within=pyo.NonNegativeReals)
model.x3 = pyo.Var(initialize=0.0, within=pyo.NonNegativeReals)
# parameters
model.eta1 = pyo.Var()
model.eta2 = pyo.Var()
# constraints + objective
model.const1 = pyo.Constraint(
    expr=6 * model.x1 + 3 * model.x2 + 2 * model.x3 - model.eta1 == 0
)
model.const2 = pyo.Constraint(expr=model.eta2 * model.x1 + model.x2 - model.x3 - 1 == 0)
model.cost = pyo.Objective(expr=model.x1**2 + model.x2**2 + model.x3**2)
model.consteta1 = pyo.Constraint(expr=model.eta1 == nominal_eta1)
model.consteta2 = pyo.Constraint(expr=model.eta2 == nominal_eta2)
###

### declare suffixes
model.sens_state_0 = pyo.Suffix(direction=pyo.Suffix.EXPORT)
model.sens_state_1 = pyo.Suffix(direction=pyo.Suffix.EXPORT)
model.sens_state_value_1 = pyo.Suffix(direction=pyo.Suffix.EXPORT)
model.sens_sol_state_1 = pyo.Suffix(direction=pyo.Suffix.IMPORT)
model.sens_init_constr = pyo.Suffix(direction=pyo.Suffix.EXPORT)
###

### set sIPOPT data
opt.options['run_sens'] = 'yes'
model.sens_state_0[model.eta1] = 1
model.sens_state_1[model.eta1] = 1
model.sens_state_value_1[model.eta1] = perturbed_eta1
model.sens_state_0[model.eta2] = 2
model.sens_state_1[model.eta2] = 2
model.sens_state_value_1[model.eta2] = perturbed_eta2
model.sens_init_constr[model.consteta1] = 1
model.sens_init_constr[model.consteta2] = 2
###

### Send the model to ipopt_sens and collect the solution
results = opt.solve(model, keepfiles=keepfiles, tee=stream_solver)
###

### Print Solution
print("Nominal and perturbed solution:")
for v in [model.x1, model.x2, model.x3, model.eta1, model.eta2]:
    print("%5s %14g %14g" % (v, pyo.value(v), model.sens_sol_state_1[v]))
###
