import pyomo.environ
from pyomo.core import *
from pyomo.gdp import *
from pyomo.opt import SolverFactory

from pyomo.common.modeling import unique_component_name
from pyomo.core.plugins.transform.hierarchy import NonIsomorphicTransformation

from random import randint

import logging
logger = logging.getLogger('pyomo.core')

# DEBUG
from nose.tools import set_trace


@TransformationFactory.register('core.add_slack_variables', \
          doc="Create a model where we add slack variables to every constraint "
          "and add new objective penalizing the sum of the slacks")
class AddSlackVariables(NonIsomorphicTransformation):
    """
    This plugin adds slack variables to every constraint or to the constraints
    specified in targets
    """

    def __init__(self, **kwds):
        kwds['name'] = "add_slack_vars"
        super(AddSlackVariables, self).__init__(**kwds)

    def _apply_to(self, instance, **kwds):
        targets = kwds.pop('targets', None)

        if kwds:
            logger.warning("Unrecognized keyword arguments in add slack "
                           "variable transformation:\n%s"
                           % ( '\n'.join(iterkeys(kwds)), ))

        if targets is None:
            constraintDatas = instance.component_data_objects(
                Constraint, descend_into=True)
        else:
            constraintDatas = []
            for cuid in targets:
                cons = cuid.find_component(instance)
                if cons.is_indexed():
                    for i in cons:
                        constraintDatas.append(cons[i])
                else:
                    constraintDatas.append(cons)

        # deactivate the objective
        for o in instance.component_data_objects(Objective):
            o.deactivate()

        # create block where we can add slack variables safely
        xblockname = unique_component_name(instance, "_core_add_slack_variables")
        instance.add_component(xblockname, Block())
        xblock = instance.component(xblockname)

        obj_expr = 0
        for cons in constraintDatas:
            if (cons.lower is not None and cons.upper is not None) and \
               value(cons.lower) > value(cons.upper):
                # this is a structural infeasibility so slacks aren't going to
                # help:
                raise RuntimeError("Lower bound exceeds upper bound in "
                                   "constraint %s" % cons.name)
            if not cons.active: continue
            if cons.lower is not None:
                # we add positive slack variable to body:
                # declare positive slack
                varName = "_slack_plus_" + cons.name
                posSlack = Var(within=NonNegativeReals)
                xblock.add_component(varName, posSlack)
                # add positive slack to body expression
                cons._body += posSlack
                # penalize slack in objective
                obj_expr += posSlack
            if cons.upper is not None:
                # we subtract a positive slack variable from the body:
                # declare slack
                varName = "_slack_minus_" + cons.name
                negSlack = Var(within=NonNegativeReals)
                xblock.add_component(varName, negSlack)
                # add negative slack to body expression
                cons._body -= negSlack
                # add slack to objective
                obj_expr += negSlack

        # make a new objective that minimizes sum of slack variables
        xblock._slack_objective = Objective(expr=obj_expr)
