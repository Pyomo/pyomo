# @Import_symbols_for_pyomo
# iterative1.py
from pyomo.environ import *
from pyomo.opt import SolverFactory
# @Import_symbols_for_pyomo

# @Call_SolverFactory_with_argument
# Create a solver
opt = SolverFactory('glpk')
# @Call_SolverFactory_with_argument

#
# A simple model with binary variables and
# an empty constraint list.
#
# @Create_base_model
model = AbstractModel()
model.n = Param(default=4)
model.x = Var(RangeSet(model.n), within=Binary)
def o_rule(model):
    return summation(model.x)
model.o = Objective(rule=o_rule)
# @Create_base_model
# @Create_empty_constraint_list
model.c = ConstraintList()
# @Create_empty_constraint_list

# Create a model instance and optimize
# @Create_instantiated_model
instance = model.create_instance()
# @Create_instantiated_model
# @Solve_and_refer_to_results
results = opt.solve(instance)
# @Solve_and_refer_to_results
# @Display_updated_value
instance.display()
# @Display_updated_value

# Iterate to eliminate the previously found solution
# @Assign_integers
for i in range(5):
# @Assign_integers
# @Associate_results_with_instance
    instance.solutions.load_from(results)
# @Associate_results_with_instance
# @Iteratively_assign_and_test
    expr = 0
    for j in instance.x:
        if instance.x[j].value == 0:
            expr += instance.x[j]
        else:
            expr += (1-instance.x[j])
# @Iteratively_assign_and_test
# @Add_expression_constraint
    instance.c.add( expr >= 1 )
# @Add_expression_constraint
# @Find_and_display_solution
    results = opt.solve(instance)
    print ("\n===== iteration",i)
    instance.display()
# @Find_and_display_solution
