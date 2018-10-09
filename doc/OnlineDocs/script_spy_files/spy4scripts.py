###NOTE: as of May 16, this will not even come close to running. DLW
### and it is "wrong" in a lot of places.
### Someone should edit this file, then delete these comment lines. DLW may 16

"""
David L. Woodruff and Mingye Yang, Spring 2018
Code snippets for scripts.rst in testable form
"""
import pyomo.environ as pyo

instance = pyo.ConcreteModel()
instance.I = pyo.Set(initialize=[1,2,3])
instance.sigma = pyo.Param(mutable=True, initialize=2.3)
instance.Theta = pyo.Param(instance.I, mutable=True)
for i in instance.I:
    instance.Theta[i] = i
ParamName = "Theta"
idx = 1
NewVal = 1134

# @Assign_value_to_indexed_parametername
instance.ParamName[idx].value = NewVal
# @Assign_value_to_indexed_parametername

ParamName = "sigma"

# @Assign_value_to_unindexed_parametername_2
instance.ParamName.value = NewVal
# @Assign_value_to_unindexed_parametername_2

instance.x = pyo.Var([1,2,3], initialize=0)
instance.y = pyo.Var()
# @Set_upper&lower_bound

if instance.x[2] == 0:
    instance.x[2].setlb(1)
    instance.x[2].setub(1)
else:
    instance.x[2].setlb(0)
    instance.x[2].setub(0)
# @Set_upper&lower_bound

# @Equivalent_form_of_instance.x.fix(2)
instance.y.value = 2
instance.y.fixed = True
# @Equivalent_form_of_instance.x.fix(2)

model=ConcreteModel()
model.obj1 = pyo.Objective(expr = 0)
model.obj2 = pyo.Objective(expr = 0)

# @Pass_multiple_objectives_to_solver
model.obj1.deactivate()
model.obj2.activate()
# @Pass_multiple_objectives_to_solver

# @Listing_arguments
def pyomo_preprocess(options=None):
   if options == None:
      print ("No command line options were given.")
   else:
      print ("Command line arguments were: %s" % options)
# @Listing_arguments


# @Provide_dictionary_for_arbitrary_keywords
def pyomo_preprocess(**kwds):
   options = kwds.get('options',None)
   if options == None:
      print ("No command line options were given.")
   else:
      print ("Command line arguments were: %s" % options)
# @Provide_dictionary_for_arbitrary_keywords

# @Pyomo_preprocess_argument
def pyomo_preprocess(options=None):
    pass
# @Pyomo_preprocess_argument

# @Display_all_variables&values
for v in instance.component_objects(pyo.Var, active=True):
    print("Variable",v)
    for index in v:
        print ("   ",index, pyo.value(v[index]))
# @Display_all_variables&values

# @Display_all_variables&values_data
for v in instance.component_data_objects(pyo.Var, active=True):
    print(v, pyo.value(v))
# @Display_all_variables&values_data


instance.iVar = pyo.Var([1,2,3], initialize=1, domain=pyo.Boolean)
instance.sVar = pyo.Var(initialize=1, domain=pyo.Boolean)
# dlw may 2018: the next snippet does not trigger any fixing ("active?")
# @Fix_all_integers&values
for var in instance.component_data_objects(pyo.Var, active=True):
    if var.domain is pyo.IntegerSet or var.domain is pyo.BooleanSet:
        print ("fixing "+str(v))
        var.fixed = True # fix the current value
# @Fix_all_integers&values

# @Include_definition_in_modelfile
def pyomo_print_results(options, instance, results):
    for v in instance.component_objects(pyo.Var, active=True):
        print ("Variable "+str(v))
        varobject = getattr(instance, v)
        for index in varobject:
            print ("   ",index, varobject[index].value)
# @Include_definition_in_modelfile

# @Print_parameter_name&value
for parmobject in instance.component_objects(pyo.Param, active=True):
    print ("Parameter "+str(parmobject.name))
    for index in parmobject:
        print ("   ",index, parmobject[index].value)
# @Print_parameter_name&value

# @Include_definition_output_constraints&duals
def pyomo_print_results(options, instance, results):
    # display all duals
    print ("Duals")
    for c in instance.component_objects(pyo.Constraint, active=True):
        print ("   Constraint",c)
        cobject = getattr(instance, c)
        for index in cobject:
            print ("      ", index, instance.dual[cobject[index]])
# @Include_definition_output_constraints&duals

"""
xxxxxxxxxxxxxxxxxxxx high alert!!!! xxxxxx testing blocked from here to the end xxxxxxxxxxxxxx
# @Print_solver_status
results = opt.solve(instance)
#print ("The solver returned a status of:"+str(results.solver.status))
# @Print_solver_status

# @Pyomo_data_comparedwith_solver_status_1
from pyomo.opt import SolverStatus, TerminationCondition

#...

if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
     print ("this is feasible and optimal")
elif results.solver.termination_condition == TerminationCondition.infeasible:
     print ("do something about it? or exit?")
else:
     # something else is wrong
     print (str(results.solver))
# @Pyomo_data_comparedwith_solver_status_1

# @Pyomo_data_comparedwith_solver_status_2
from pyomo.opt import TerminationCondition

...

results = opt.solve(model, load_solutions=False)
if results.solver.termination_condition == TerminationCondition.optimal:
    model.solutions.load_from(results)
else:
    print ("Solution is not optimal")
    # now do something about it? or exit? ...
# @Pyomo_data_comparedwith_solver_status_2

# @See_solver_output
results = opt.solve(instance, tee=True)
# @See_solver_output

# @Add_option_to_solver
optimizer = pyo.SolverFactory['cbc']
optimizer.options["threads"] = 4
# @Add_option_to_solver

# @Add_multiple_options_to_solver
results = optimizer.solve(instance, options="threads=4", tee=True)
# @Add_multiple_options_to_solver

# @Set_path_to_solver_executable
opt = pyo.SolverFactory("ipopt", executable="../ipopt")
# @Set_path_to_solver_executable

# @Pass_warmstart_to_solver
instance = model.create()
instance.y[0] = 1
instance.y[1] = 0

opt = pyo.SolverFactory("cplex")

results = opt.solve(instance, warmstart=True)
# @Pass_warmstart_to_solver

# @Specify_temporary_directory_name
from pyutilib.services import TempfileManager
TempfileManager.tempdir = YourDirectoryNameGoesHere
# @Specify_temporary_directory_name
"""
