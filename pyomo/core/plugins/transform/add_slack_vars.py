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

    minimize = 1
    maximize = -1

    # TODO: is this right? this is a guess...
    alias('core.add_slack_variables', \
          doc="Create a model where we had slack variables to every constraint")

    def __init__(self, **kwds):
        kwds['name'] = "add_slack_vars"
        super(AddSlackVariables, self).__init__(**kwds)

    @staticmethod
    def add_slacks_to_obj(obj_expr, sense, slackVar):
        if sense == minimize:
            obj_expr += slackVar
        elif sense == maximize:
            obj_expr -= slackVar
        else:
            raise RuntimeError("Unrecognized objective sense: %s" % sense)
        return obj_expr

    # helper function for constraints which require two slack variables (one added
    # and one subtracted). Right now these are equality constraints and a <= b <= c
    def add_two_slacks(self, instance, cons, numArgs, lhs, rhs, obj_expr, sense):
        plusVarName = "_slack_plus_" + cons.name
        minusVarName = "_slack_minus_" + cons.name
        instance.add_component(plusVarName, Var(within=NonNegativeReals))
        instance.add_component(minusVarName, Var(within=NonNegativeReals))
        plusVar = getattr(instance, plusVarName)
        minusVar = getattr(instance, minusVarName)
        # TODO: this ternary bit assumes that 2 and 3 arg inequalities are all we will 
        # ever get. If this isn't true, this should be a switch.
        expr = lhs + plusVar - minusVar == rhs if numArgs == 2 else \
               lhs <= cons.expr._args[1] + plusVar - minusVar <= rhs
        instance.add_component("_slackConstraint_" + cons.name, Constraint(expr=expr))
        # add slacks to objective:
        obj_expr = self.add_slacks_to_obj(obj_expr, sense, plusVar + minusVar)
        return (instance, obj_expr)

    def add_one_slack(self, instance, cons, lhs, rhs, obj_expr, sense):
        varName = "_slack_" + cons.name
        instance.add_component(varName, Var(within=NonNegativeReals))
        slackVar = getattr(instance, varName)
        instance.add_component("_slackConstraint_" + cons.name, Constraint(
            expr=lhs - slackVar <= rhs))
        obj_expr = self.add_slacks_to_obj(obj_expr, sense, slackVar)
        return (instance, obj_expr)
        

    def _apply_to(self, instance, **kwds):
        # constriant types dict
        constraint_types = {'equality': pyomo.core.base.expr_coopr3._EqualityExpression,
                        'inequality': pyomo.core.base.expr_coopr3._InequalityExpression}

        # TODO: should these be class variables? It seems to make sense for easy access
        # in the helper functions... I am so rusty on OO stuff...
        # obj_expr = 0
        # sense = 1

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
            if cons.name.startswith("_slackConstraint_"): continue
            expr = cons.expr
            exprType = type(expr)
            cons.deactivate()
            args = cons.expr._args
            lhs = args[0]
            rhs = args[-1]
            # there are cases depending on what kind of expression this is.
            # TODO: what cases haven't I covered??
            if (exprType == constraint_types['equality']):
                instance, obj_expr = self.add_two_slacks(instance, cons, 2,
                                                                lhs, rhs, obj_expr, sense)
                # we need to add two slack variables
                #print "equality"
                # plusVarName = "_slack_plus_" + cons.name
                # minusVarName = "_slack_minus_" + cons.name
                # instance.add_component(plusVarName, Var(within=NonNegativeReals))
                # instance.add_component(minusVarName, Var(within=NonNegativeReals))
                # plusVar = getattr(instance, plusVarName)
                # minusVar = getattr(instance, minusVarName)
                # instance.add_component("_slackConstraint_" + cons.name, Constraint(
                #     expr=lhs + plusVar - minusVar == rhs))

                
                # if sense == minimize:
                #     obj_expr += plusVar + minusVar
                # elif sense == maximize:
                #     obj_expr -= plusVar + minusVar
                # else:
                #     raise RuntimeError("Unrecognized objective sense: %s" % sense)
            elif (exprType == constraint_types['inequality']):
                # two cases (that I know of): either a <= b or a <= b <= c
                if len(args) == 2:
                    instance, obj_expr = self.add_one_slack(instance, cons, lhs, rhs,
                                                            obj_expr, sense)
                    
                    # add slacks to objective:
                    # if sense == minimize:
                    #     obj_expr += slackVar
                    # elif sense == maximize:
                    #     obj_expr -= slackVar
                    # else:
                    #     raise RuntimeError("Unrecognized objective sense: %s" % sense)
                elif len(args) == 3:
                    instance, obj_expr = self.add_two_slacks(instance, cons, 3, lhs, 
                                                             rhs, obj_expr, sense)
                    # TODO: this is identical code to the equality case except for the
                    # expression of the new constraint. Should be helper function.
                    # plusVarName = "_slack_plus_" + cons.name
                    # minusVarName = "_slack_minus_" + cons.name
                    # instance.add_component(plusVarName, Var(within=NonNegativeReals))
                    # instance.add_component(minusVarName, Var(within=NonNegativeReals))
                    # plusVar = getattr(instance, plusVarName)
                    # minusVar = getattr(instance, minusVarName)
                    # instance.add_component("_slackConstraint_" + cons.name, Constraint(
                    #     expr=lhs <= args[1] + plusVar - minusVar <= rhs))
                    # # TODO: this should be a helper function for sure.
                    # # add slacks to objective:
                    # if sense == minimize:
                    #     obj_expr += plusVar + minusVar
                    # elif sense == maximize:
                    #     obj_expr -= plusVar + minusVar
                    # else:
                    #     raise RuntimeError("Unrecognized objective sense: %s" % sense)
                else:
                    raise RuntimeError("""Unrecognized number of expressions in inequality
                                       constraint: %s args in constraint %s""" \
                                       % (len(args), cons.name))
            else:
                raise RuntimeError("Unrecognized constraint type: %s" % (exprType))

        # make a new objective that includes the slack variables
        instance.add_component("_slack_objective", Objective(expr=obj_expr, sense=sense))

        # TODO: I don't know what the plan should be in general... For now I am just going to 
        # do bigm and solve it. Does bigm hurt things that don't have disjunctions?

        bigMRelaxation = TransformationFactory('gdp.bigm')
        bigMRelaxation.apply_to(instance)
