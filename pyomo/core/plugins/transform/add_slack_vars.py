#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core import TransformationFactory, Var, NonNegativeReals, Constraint, Objective, Block, value

from six import iterkeys

from pyomo.common.modeling import unique_component_name
from pyomo.core.plugins.transform.hierarchy import NonIsomorphicTransformation
from pyomo.common.config import ConfigBlock, ConfigValue
from pyomo.core.base import ComponentUID
from pyomo.core.base.constraint import _ConstraintData
from pyomo.common.deprecation import deprecation_warning

NAME_BUFFER = {}

def target_list(x):
    deprecation_msg = ("In future releases ComponentUID targets will no "
                      "longer be supported in the core.add_slack_variables "
                      "transformation. Specify targets as a Constraint or "
                      "list of Constraints.")
    if isinstance(x, ComponentUID):
        if deprecation_msg:
            deprecation_warning(deprecation_msg)
            # only emit the message once
            deprecation_msg = None
        # [ESJ 07/15/2020] We have to just pass it through because we need the
        # instance in order to be able to do anything about it...
        return [ x ]
    elif isinstance(x, (Constraint, _ConstraintData)):
        return [ x ]
    elif hasattr(x, '__iter__'):
        ans = []
        for i in x:
            if isinstance(i, ComponentUID):
                if deprecation_msg:
                    deprecation_warning(deprecation_msg)
                    deprecation_msg = None
                # same as above...
                ans.append(i)
            elif isinstance(i, (Constraint, _ConstraintData)):
                ans.append(i)
            else:
                raise ValueError(
                    "Expected Constraint or list of Constraints."
                    "\n\tRecieved %s" % (type(i),))
        return ans
    else:
        raise ValueError(
            "Expected Constraint or list of Constraints."
            "\n\tRecieved %s" % (type(x),))

import logging
logger = logging.getLogger('pyomo.core')


@TransformationFactory.register('core.add_slack_variables', \
          doc="Create a model where we add slack variables to every constraint "
          "and add new objective penalizing the sum of the slacks")
class AddSlackVariables(NonIsomorphicTransformation):
    """
    This plugin adds slack variables to every constraint or to the constraints
    specified in targets.
    """

    CONFIG = ConfigBlock("core.add_slack_variables")
    CONFIG.declare('targets', ConfigValue(
        default=None,
        domain=target_list,
        description="target or list of targets to which slacks will be added",
        doc="""

        This specifies the list of Constraints to add slack variables to.
        """
    ))

    def __init__(self, **kwds):
        kwds['name'] = "add_slack_vars"
        super(AddSlackVariables, self).__init__(**kwds)

    def _apply_to(self, instance, **kwds):
        try:
            assert not NAME_BUFFER
            self._apply_to_impl(instance, **kwds)
        finally:
            NAME_BUFFER.clear()

    def _apply_to_impl(self, instance, **kwds):
        config = self.CONFIG(kwds.pop('options', {}))
        config.set_value(kwds)
        targets = config.targets

        if targets is None:
            constraintDatas = instance.component_data_objects(
                Constraint, descend_into=True)
        else:
            constraintDatas = []
            for t in targets:
                if isinstance(t, ComponentUID):
                    cons = t.find_component(instance)
                    if cons.is_indexed():
                        for i in cons:
                            constraintDatas.append(cons[i])
                    else:
                        constraintDatas.append(cons)
                else:
                    # we know it's a constraint because that's all we let
                    # through the config block validation.
                    if t.is_indexed():
                        for i in t:
                            constraintDatas.append(t[i])
                    else:
                        constraintDatas.append(t)

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
            cons_name = cons.getname(fully_qualified=True,
                                     name_buffer=NAME_BUFFER)
            if cons.lower is not None:
                # we add positive slack variable to body:
                # declare positive slack
                varName = "_slack_plus_" + cons_name
                posSlack = Var(within=NonNegativeReals)
                xblock.add_component(varName, posSlack)
                # add positive slack to body expression
                cons._body += posSlack
                # penalize slack in objective
                obj_expr += posSlack
            if cons.upper is not None:
                # we subtract a positive slack variable from the body:
                # declare slack
                varName = "_slack_minus_" + cons_name
                negSlack = Var(within=NonNegativeReals)
                xblock.add_component(varName, negSlack)
                # add negative slack to body expression
                cons._body -= negSlack
                # add slack to objective
                obj_expr += negSlack

        # make a new objective that minimizes sum of slack variables
        xblock._slack_objective = Objective(expr=obj_expr)
