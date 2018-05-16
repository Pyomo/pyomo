###NOTE: as of May 16, this will not even come close to running. DLW
### and it is "wrong" in a lot of places.
### Someone should edit this file, then delete these comment lines. DLW may 16

"""
David L. Woodruff and Mingye Yang, Spring 2018
Code snippets for scripts.rst in testable form
"""


# @Assign_value_to_indexed_parametername
getattr(instance, ParamName)[idx] = NewVal
# @Assign_value_to_indexed_parametername

# @Assign_value_to_unindexed_parametername_1
getattr(instance, ParamName)[None] = NewVal
# @Assign_value_to_unindexed_parametername_1

# @Assign_value_to_unindexed_parametername_2
getattr(instance, ParamName).set_value(NewVal)
# @Assign_value_to_unindexed_parametername_2

# @Set_upper&lower_bound
instance.solutions.load_from(results)

if instance.x[2] == 0:
    instance.x[2].setlb(1)
    instance.x[2].setub(1)
else:
    instance.x[2].setlb(0)
    instance.x[2].setub(0)
results = opt.solve(instance)
# @Set_upper&lower_bound

# @Equivalent_form_of_instance.x.fix(2)
instance.x.value = 2
instance.x.fixed = True
# @Equivalent_form_of_instance.x.fix(2)

# @Pass_multiple_objectives_to_solver
model.obj1.deactivate()
model.obj2.activate()
# @Pass_multiple_objectives_to_solver

# @Listing_arguments
def pyomo_preprocess(options=None):
   if options == None:
      print "No command line options were given."
   else:
      print "Command line arguments were: %s" % options
# @Listing_arguments


# @Provide_dictionary_for_arbitrary_keywords
def pyomo_preprocess(**kwds):
   options = kwds.get('options',None)
   if options == None:
      print "No command line options were given."
   else:
      print "Command line arguments were: %s" % options
# @Provide_dictionary_for_arbitrary_keywords

# @Pyomo_preprocess_argument
def pyomo_preprocess(options=None):
# @Pyomo_preprocess_argument

# @Display_all_variables&values
for v in instance.component_objects(Var, active=True):
    print ("Variable",v)
    varobject = getattr(instance, str(v))
    for index in varobject:
        print ("   ",index, varobject[index].value)
# @Display_all_variables&values

# @Fix_all_integers&values
for v in instance.component_objects(Var, active=True):
    varobject = getattr(instance, v)
    if isinstance(varobject.domain, IntegerSet) or isinstance(varobject.domain, BooleanSet):
        print ("fixing "+str(v))
        for index in varobject:
            varobject[index].fixed = True # fix the current value
# @Fix_all_integers&values

# @Include_definition_in_modelfile
def pyomo_print_results(options, instance, results):
    from pyomo.core import Var
    for v in instance.component_objects(Var, active=True):
        print ("Variable "+str(v))
        varobject = getattr(instance, v)
        for index in varobject:
            print ("   ",index, varobject[index].value)
# @Include_definition_in_modelfile

# @Print_parameter_name&value
from pyomo.core import Param
for p in instance.component_objects(Param, active=True):
    print ("Parameter "+str(p))
    parmobject = getattr(instance, p)
    for index in parmobject:
        print ("   ",index, parmobject[index].value)
# @Print_parameter_name&value

# @Include_definition_output_constraints&duals
def pyomo_print_results(options, instance, results):
    # display all duals
    print ("Duals")
    from pyomo.core import Constraint
    for c in instance.component_objects(Constraint, active=True):
        print ("   Constraint",c)
        cobject = getattr(instance, c)
        for index in cobject:
            print ("      ", index, instance.dual[cobject[index]])
# @Include_definition_output_constraints&duals

# @Print_solver_status
instance = model.create()
results = opt.solve(instance)
print ("The solver returned a status of:"+str(results.Solution.Status))
# @Print_solver_status

# @Pyomo_data_comparedwith_solver_status_1
from pyomo.opt import SolverStatus, TerminationCondition

...

if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
     # this is feasible and optimal
elif results.solver.termination_condition == TerminationCondition.infeasible:
     # do something about it? or exit?
else:
     # something else is wrong
     print (results.solver)
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
optimizer = SolverFactory['cbc']
optimizer.options["threads"] = 4
# @Add_option_to_solver

# @Add_multiple_options_to_solver
results = optimizer.solve(instance, options="threads=4", tee=True)
# @Add_multiple_options_to_solver

# @Set_path_to_solver_executable
opt = SolverFactory("ipopt", executable="../ipopt")
# @Set_path_to_solver_executable

# @Pass_warmstart_to_solver
instance = model.create()
instance.y[0] = 1
instance.y[1] = 0

opt = SolverFactory("cplex")

results = opt.solve(instance, warmstart=True)
# @Pass_warmstart_to_solver

# @Specify_temporary_directory_name
from pyutilib.services import TempFileManager
TempfileManager.tempdir = YourDirectoryNameGoesHere
# @Specify_temporary_directory_name
