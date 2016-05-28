from pyomo.environ import *
from pyomo.opt import SolverFactory

### Create the ipopt solver plugin using the ASL interface
solver = 'ipopt'
solver_io = 'nl'
stream_solver = True     # True prints solver output to screen
keepfiles =     False    # True prints intermediate file names (.nl,.sol,...) 
opt = SolverFactory(solver,solver_io=solver_io)

if opt is None:
    print("")
    print("ERROR: Unable to create solver plugin for %s "\
          "using the %s interface" % (solver, solver_io))
    print("")
    exit(1)
###

### Create the example model
model = ConcreteModel()
model.x1 = Var(bounds=(1,5),initialize=1.0)
model.x2 = Var(bounds=(1,5),initialize=5.0)
model.x3 = Var(bounds=(1,5),initialize=5.0)
model.x4 = Var(bounds=(1,5),initialize=1.0)
model.obj = Objective(expr=model.x1*model.x4*(model.x1+model.x2+model.x3) + model.x3)
model.inequality = Constraint(expr=model.x1*model.x2*model.x3*model.x4 >= 25.0)
model.equality = Constraint(expr=model.x1**2 + model.x2**2 + model.x3**2 + model.x4**2 == 40.0)
###

### Declare all suffixes 
# Ipopt bound multipliers (obtained from solution)
model.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
model.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
# Ipopt bound multipliers (sent to solver)
model.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
model.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)
# Obtain dual solutions from first solve and send to warm start
model.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
###

### Send the model to ipopt and collect the solution
print("")
print("INITIAL SOLVE")
results = opt.solve(model,keepfiles=keepfiles,tee=stream_solver)
# load the results (including any values for previously declared
# IMPORT / IMPORT_EXPORT Suffix components)
model.solutions.load_from(results)
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

### Send the model and suffix information to ipopt and collect the solution
print("") 
print("WARM-STARTED SOLVE")
# The solver plugin will scan the model for all active suffixes
# valid for importing, which it will store into the results object
results = opt.solve(model,keepfiles=keepfiles,tee=stream_solver)
# load the results (including any values for previously declared
# IMPORT / IMPORT_EXPORT Suffix components)
model.solutions.load_from(results)
###

