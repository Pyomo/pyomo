"""Re-implementation of example 1 of Quesada and Grossmann.

Re-implementation of Quesada example 2 MINLP test problem in Pyomo
Author: David Bernal <https://github.com/bernalde>.

The expected optimal solution value is -5.512.

Ref:
    Quesada, Ignacio, and Ignacio E. Grossmann.
    "An LP/NLP based branch and bound algorithm
    for convex MINLP optimization problems."
    Computers & chemical engineering 16.10-11 (1992): 937-947.

    Problem type:    convex MINLP
            size:    1  binary variable
                     2  continuous variables
                     4  constraints


"""
from __future__ import division

from six import iteritems

from pyomo.environ import (Binary, ConcreteModel, Constraint, Reals,
                           Objective, Param, RangeSet, Var, exp, minimize, log)


class OnlineDocExample(ConcreteModel):

    def __init__(self, *args, **kwargs):
        """Create the problem."""
        kwargs.setdefault('name', 'OnlineDocExample')
        super(OnlineDocExample, self).__init__(*args, **kwargs)
        model = self
        model.x = Var(bounds=(1.0, 10.0), initialize=5.0)
        model.y = Var(within=Binary)
        model.c1 = Constraint(expr=(model.x-3.0)**2 <= 50.0*(1-model.y))
        model.c2 = Constraint(expr=model.x*log(model.x)+5.0 <= 50.0*(model.y))
        model.objective = Objective(expr=model.x, sense=minimize)
# SolverFactory('mindtpy').solve(model, strategy='OA',
#                                init_strategy='max_binary', mip_solver='cplex', nlp_solver='ipopt')
# SolverFactory('mindtpy').solve(model, strategy='OA',
#                                mip_solver='cplex', nlp_solver='ipopt',
#                                init_strategy='max_binary',
#                                #    single_tree=True,
#                                #   add_integer_cuts=True
#                                )

# # SolverFactory('gams').solve(model, solver='baron', tee=True, keepfiles=True)
# model.objective.display()
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
