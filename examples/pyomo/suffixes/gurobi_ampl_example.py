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

# A Suffix example for the gurobi_ampl solver.
#
# This Pyomo example is formulated as a python script.
# To run this script execute the following command:
#
# $ pyomo_python gurobi_ampl_example.py
#
# Execution of this script requires that the gurobi_ampl
# solver is in the current search path for executables
# on this system. This example was tested using Gurobi
# Solver 5.0.0

import pyomo.environ
from pyomo.core import *
from pyomo.opt import SolverFactory

### Create the gurobi_ampl solver plugin using the ASL interface
solver = 'gurobi_ampl'
solver_io = 'nl'
stream_solver = False  # True prints solver output to screen
keepfiles = False  # True prints intermediate file names (.nl,.sol,...)
opt = SolverFactory(solver, solver_io=solver_io)

if opt is None:
    print("")
    print(
        "ERROR: Unable to create solver plugin for %s "
        "using the %s interface" % (solver, solver_io)
    )
    print("")
    exit(1)

opt.options['outlev'] = 1  # tell gurobi to be verbose with output
###

### Create a trivial example model
model = ConcreteModel()
model.s = Set(initialize=[1, 2, 3])
model.x = Var(model.s, within=NonNegativeReals)
model.obj = Objective(expr=sum_product(model.x))
model.con = Constraint(model.s, rule=lambda model, i: model.x[i] >= i - 1)
###

### Declare all suffixes
# The variable solution status suffix
# (this suffix can be sent to the solver and loaded from the solution)
sstatus_table = {
    'bas': 1,  # basic
    'sup': 2,  # superbasic
    'low': 3,  # nonbasic <= (normally =) lower bound
    'upp': 4,  # nonbasic >= (normally =) upper bound
    'equ': 5,  # nonbasic at equal lower and upper bounds
    'btw': 6,
}  # nonbasic between bounds
model.sstatus = Suffix(direction=Suffix.IMPORT_EXPORT, datatype=Suffix.INT)
model.dual = Suffix(direction=Suffix.IMPORT_EXPORT)

# Report the best known bound on the objective function
model.bestbound = Suffix(direction=Suffix.IMPORT)

# A few Gurobi variable solution sensitivity suffixes
model.senslblo = Suffix(direction=Suffix.IMPORT)  # smallest variable lower bound
model.senslbhi = Suffix(direction=Suffix.IMPORT)  # greatest variable lower bound
model.sensublo = Suffix(direction=Suffix.IMPORT)  # smallest variable upper bound
model.sensubhi = Suffix(direction=Suffix.IMPORT)  # greatest variable upper bound

# A Gurobi constraint solution sensitivity suffix
model.sensrhshi = Suffix(direction=Suffix.IMPORT)  # greatest right-hand side value
###

# Tell gurobi_ampl to report solution sensitivities
# and bestbound via suffixes in the solution file
opt.options['solnsens'] = 1
opt.options['bestbound'] = 1

# Set one of the sstatus suffix values, which will be sent to the solver
model.sstatus[model.x[1]] = sstatus_table['low']


def print_model_suffixes(model):
    # print all suffix values for all model components in a nice table
    print("\t", end='')
    for name, suffix in active_import_suffix_generator(model):
        print("%10s" % (name), end='')
    print("")
    for i in model.s:
        print(model.x[i].name + "\t", end='')
        for name, suffix in active_import_suffix_generator(model):
            print("%10s" % (suffix.get(model.x[i])), end='')
        print("")
    for i in model.s:
        print(model.con[i].name + "\t", end='')
        for name, suffix in active_import_suffix_generator(model):
            print("%10s" % (suffix.get(model.con[i])), end='')
        print("")
    print(model.obj.name + "\t", end='')
    for name, suffix in active_import_suffix_generator(model):
        print("%10s" % (suffix.get(model.obj)), end='')
    print("")
    print("")


print("")
print("Suffixes Before Solve:")
print_model_suffixes(model)

### Send the model to gurobi_ampl and collect the solution
# The solver plugin will scan the model for all active suffixes
# valid for importing, which it will store into the results object
results = opt.solve(model, keepfiles=keepfiles, tee=stream_solver)
###

print("")
print("Suffixes After Solve:")
print_model_suffixes(model)
