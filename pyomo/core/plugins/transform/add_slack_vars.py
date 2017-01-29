import pyomo.environ
from pyomo.core import *
from pyomo.gdp import *
from pyomo.opt import SolverFactory

import pdb

from pyomo.util.plugin import alias
from pyomo.core.plugins.transform.hierarchy import NonIsomorphicTransformation

class AddSlackVariables(NonIsomorphicTransformation):
    """
    This plugin adds slack variables to every constraint
    """

    # TODO: is this right? this is a guess...
    alias('core.add_slack_variables', \
          doc="Create a model where we had slack variables to every constraint")

    def __init__(self, **kwds):
        kwds['name'] = "add_slack_vars"
        super(AddSlackVariables, self).__init__(**kwds)

    def _apply_to(self, instance, **kwds):
        # constriant types dict
        constraint_types = {'equality': pyomo.core.base.expr_coopr3._EqualityExpression,
                        'inequality': pyomo.core.base.expr_coopr3._InequalityExpression}

        obj_expr = 0
        sense = 1
        minimize = 1
        maximize = -1

        # deactivate the objective
        for o in instance.component_data_objects(Objective):
            if o.active: 
                obj_expr = o.expr 
                sense = o.sense
                o.deactivate()

            #iteration = 0
            #instance.pprint()
        for cons in instance.component_data_objects(Constraint, descend_into=(Block, Disjunct)):
            # don't want to do anything with constraints we've already added slacks to
            print cons.name
            if cons.name.startswith("_slackConstraint_"): continue
            #print(cons.name)
            expr = cons.expr
            #pdb.set_trace()
            cons.deactivate()
            # DEBUG
            #expr.to_string()
            lhs = cons.expr._args[0]
            rhs = cons.expr._args[1]
            # there are cases depending on what kind of expression this is.
            # TODO: I also know for sure I'm not covering all of them... The tuples are left out 
            #right now...
            # and the thing <= thing <= thing will come up in the disjunctions...
            exprType = type(expr)
            print exprType
            if (exprType == constraint_types['equality']):
                # we need to add two slack variables
                #print "equality"
                plusVarName = "_slack_plus_" + cons.name
                minusVarName = "_slack_minus_" + cons.name
                instance.add_component(plusVarName, Var(within=NonNegativeReals))
                instance.add_component(minusVarName, Var(within=NonNegativeReals))
                plusVar = getattr(instance, plusVarName)
                minusVar = getattr(instance, minusVarName)
                instance.add_component("_slackConstraint_" + cons.name, Constraint(
                    expr=lhs + plusVar - minusVar == rhs))
                # add slacks to objective:
                if sense == minimize:
                    obj_expr += plusVar + minusVar
                elif sense == maximize:
                    obj_expr -= plusVar + minusVar
                else:
                    raise RuntimeError("Unrecognized objective sense: %s" % sense)
            elif (exprType == constraint_types['inequality']):
                #print "inequality"
                varName = "_slack_" + cons.name
                instance.add_component(varName, Var(within=NonNegativeReals))
                slackVar = getattr(instance, varName)
                instance.add_component("_slackConstraint_" + cons.name, Constraint(
                    expr=lhs - slackVar <= rhs))
                # add slacks to objective:
                if sense == minimize:
                    obj_expr += slackVar
                elif sense == maximize:
                    obj_expr -= slackVar
                else:
                    raise RuntimeError("Unrecognized objective sense: %s" % sense)
            else:
                raise RuntimeError("Unrecognized constraint type: %s" % (exprType))

        # make a new objective that includes the slack variables
        instance.add_component("_slack_objective", Objective(expr=obj_expr, sense=sense))

        # TODO: I don't know what the plan should be in general... For now I am just going to 
        # do bigm and solve it. Does bigm hurt things that don't have disjunctions?

        bigMRelaxation = TransformationFactory('gdp.bigm')
        bigMRelaxation.apply_to(instance)
