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

# A Suffix example for the gurobi_ampl solver that uses
# basis information from a previous solve to warmstart
# another solve.
#
# This Pyomo example is formulated as a python script.
# To run this script execute the following command:
#
# $ pyomo_python gurobi_ampl_basis.py
#
# Execution of this script requires that the gurobi_ampl
# solver is in the current search path for executables
# on this system. This example was tested using
# gurobi_ampl version 6.5.0.
from pyomo.environ import *


#
# Create the gurobi_ampl solver plugin using the ASL interface
#
solver = 'gurobi_ampl'
solver_io = 'nl'
stream_solver = True  # True prints solver output to screen
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

#
# Set a few solver options that make the effect of
# warmstarting more obvious for this trivial example model
#

# tell gurobi to be verbose with output
opt.options['outlev'] = 1
# disable presolve so we can see the effect
# that warmstarting will have on simplex iterations
opt.options['presolve'] = 0
# make sure gurobi_ampl returns and uses the sstatus suffix
# (solution basis information)
opt.options['basis'] = 3
# use primal simplex
opt.options['method'] = 0

#
# Create a trivial example model
#
model = ConcreteModel()
model.s = Set(initialize=[1, 2, 3])
model.x = Var(model.s, within=NonNegativeReals)
model.obj = Objective(expr=sum_product(model.x))
model.con = Constraint(model.s, rule=lambda model, i: model.x[i] >= i - 1)
###

#
# Declare all suffixes
#

# According to the Gurobi documentation, sstatus suffix
# values can be interpreted as:
#  - 1: basic
#  - 2: superbasic
#  - 3: nonbasic <= (normally =) lower bound
#  - 4: nonbasic >= (normally =) upper bound
#  - 5: nonbasic at equal lower and upper bounds
#  - 6: nonbasic between bounds

model.sstatus = Suffix(direction=Suffix.IMPORT_EXPORT, datatype=Suffix.INT)
model.dual = Suffix(direction=Suffix.IMPORT_EXPORT)


#
# Send the model to gurobi_ampl and collect basis and dual
# information through the suffixes that have been declared
# on the model. Sometimes it is necessary to alert the
# solver that certain suffixes are requested by setting a
# solver option (see the solver documentation).
#
results = opt.solve(model, keepfiles=keepfiles, tee=stream_solver)

#
# Print the suffix values that were imported
#
print("")
print("Suffixes After First Solve:")
for i in model.s:
    print("%s.sstatus: %s" % (model.x[i].name, model.sstatus.get(model.x[i])))
for i in model.s:
    print("%s.sstatus: %s" % (model.con[i].name, model.sstatus.get(model.con[i])))
    print("%s.dual: %s" % (model.con[i].name, model.dual.get(model.con[i])))
print("")

#
# Send the model to gurobi_ampl with the previously
# collected basis and dual suffix information. The solver
# plugin will detect when there are suffixes to export to
# the solver. There should be a noticeable decrease in
# iterations shown by the solver output that is due to the
# extra warmstart information.
#
results = opt.solve(model, keepfiles=keepfiles, tee=stream_solver)
