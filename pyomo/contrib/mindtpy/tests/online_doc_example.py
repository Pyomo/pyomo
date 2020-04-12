from pyomo.contrib.mindtpy.tests.eight_process_problem import \
    EightProcessFlowsheet
from pyomo.environ import *
model = ConcreteModel()
model.x = Var(bounds=(1.0, 10.0), initialize=5.0)
model.y = Var(within=Binary)
model.c1 = Constraint(expr=(model.x-3.0)**2 <= 50.0*(1-model.y))
model.c2 = Constraint(expr=model.x*log(model.x)+5.0 <= 50.0*(model.y))
model.objective = Objective(expr=model.x, sense=minimize)
# SolverFactory('mindtpy').solve(model, strategy='OA',
#                                init_strategy='max_binary', mip_solver='cplex', nlp_solver='ipopt')
SolverFactory('mindtpy').solve(model, strategy='OA',
                               mip_solver='cplex', nlp_solver='ipopt',
                               single_tree=True,
                               add_integer_cuts=False)

# SolverFactory('gams').solve(model, solver='baron', tee=True, keepfiles=True)
model.objective.display()
# model.objective.pprint()
# model.pprint()
# model = EightProcessFlowsheet()
# print('\n Solving problem with Outer Approximation')
# SolverFactory('mindtpy').solve(model, strategy='OA',
#                                init_strategy='rNLP',
#                                mip_solver='cplex',
#                                nlp_solver='ipopt',
#                                bound_tolerance=1E-5)
# print(value(model.cost.expr))
