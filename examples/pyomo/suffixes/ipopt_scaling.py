# This Pyomo example is formulated as a python script.
# To run this script execute the following command:
#
# $ pyomo_python ipopt_scaling.py
#
# Execution of this script requires that the ipopt
# solver is in the current search path for executables
# on this system. This example was tested using Ipopt
# version 3.10.2

import pyomo.modeling
from pyomo.core import *
from pyomo.opt import SolverFactory

### Create the ipopt solver plugin using the ASL interface
solver = 'ipopt'
solver_io = 'nl'
stream_solver = False    # True prints solver output to screen
keepfiles =     False    # True prints intermediate file names (.nl,.sol,...) 
opt = SolverFactory(solver,solver_io=solver_io)

if opt is None:
    print("")
    print("ERROR: Unable to create solver plugin for %s "\
          "using the %s interface" % (solver, solver_io))
    print("")
    exit(1)
###

### Set Ipopt options to accept the scaling_factor suffix
opt.options['nlp_scaling_method'] = 'user-scaling'
###

### Create the example model
model = ConcreteModel()
model.s = Set(initialize=[1,2,3])
model.y = Var(bounds=(1,5),initialize=1.0)
model.x = Var(model.s,bounds=(1,5),initialize=5.0)
model.obj = Objective(expr=model.y*model.x[3]*(model.y+model.x[1]+model.x[2]) + model.x[2])
model.inequality = Constraint(expr=model.y*model.x[1]*model.x[2]*model.x[3] >= 25.0)
model.equality = Constraint(expr=model.y**2 + model.x[1]**2 + model.x[2]**2 + model.x[3]**2 == 40.0)
###

### Declare the scaling_factor suffix 
model.scaling_factor = Suffix(direction=Suffix.EXPORT)
# set objective scaling factor
model.scaling_factor[model.obj] = 4.23
# set variable scaling factor
model.scaling_factor[model.y] = 2.0
model.scaling_factor.setValue(model.x, 5.0)
model.scaling_factor[model.x[1]] = 1.5
# set constraint scaling factor
model.scaling_factor[model.inequality] = 2.0
model.scaling_factor[model.equality] = 2.0
###

### Generate the constraint expression trees if necessary
if solver_io != 'nl':
    # only required when not using the ASL interface
    model.preprocess()
###

### Send the model to ipopt and collect the solution
results = opt.solve(model,keepfiles=keepfiles,tee=stream_solver)
# load the results (including any values for previously declared
# IMPORT / IMPORT_EXPORT Suffix components)
model.load(results)
###
