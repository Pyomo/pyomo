# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

from collections import defaultdict
from operator import attrgetter

from pyomo.core import (
    TransformationFactory,
    Var,
    NonNegativeReals,
    Constraint,
    Objective,
    Block,
    value,
    Expression,
    Param,
    Suffix,
)

from pyomo.common.autoslots import AutoSlots
from pyomo.common.collections import ComponentMap
from pyomo.common.modeling import unique_component_name
from pyomo.core.plugins.transform.hierarchy import NonIsomorphicTransformation
from pyomo.common.config import ConfigBlock, ConfigValue
from pyomo.core.base import ComponentUID, SortComponents
from pyomo.common.deprecation import deprecation_warning

from pyomo.repn.util import categorize_valid_components

### FIXME: Remove the following as soon as non-active components no
### longer report active==True
from pyomo.network import Port
from pyomo.core.base import RangeSet, Set

import logging

logger = logging.getLogger('pyomo.core')


def target_list(x):
    deprecation_msg = (
        "In future releases ComponentUID targets will no "
        "longer be supported in the core.add_slack_variables "
        "transformation. Specify targets as a Constraint or "
        "list of Constraints."
    )
    if isinstance(x, ComponentUID):
        if deprecation_msg:
            deprecation_warning(deprecation_msg, version='5.7.1')
            # only emit the message once
            deprecation_msg = None
        # [ESJ 07/15/2020] We have to just pass it through because we need the
        # instance in order to be able to do anything about it...
        return [x]
    elif getattr(x, 'ctype', None) is Constraint:
        return [x]
    elif hasattr(x, '__iter__'):
        ans = []
        for i in x:
            if isinstance(i, ComponentUID):
                if deprecation_msg:
                    deprecation_warning(deprecation_msg, version='5.7.1')
                    deprecation_msg = None
                # same as above...
                ans.append(i)
            elif getattr(i, 'ctype', None) is Constraint:
                ans.append(i)
            else:
                raise ValueError(
                    "Expected Constraint or list of Constraints."
                    "\n\tReceived %s" % (type(i),)
                )
        return ans
    else:
        raise ValueError(
            "Expected Constraint or list of Constraints.\n\tReceived %s" % (type(x),)
        )


class _AddSlackVariablesData(AutoSlots.Mixin):
    __slots__ = ('slack_variables', 'relaxed_constraint', 'summed_slacks_expr')

    def __init__(self):
        self.slack_variables = defaultdict(list)
        self.relaxed_constraint = ComponentMap()
        self.summed_slacks_expr = None


Block.register_private_data_initializer(_AddSlackVariablesData)


@TransformationFactory.register(
    'core.add_slack_variables',
    doc="Create a model where we add slack variables to every constraint "
    "and add new objective penalizing the sum of the slacks",
)
class AddSlackVariables(NonIsomorphicTransformation):
    """
    This plugin adds slack variables to every constraint or to the constraints
    specified in targets.
    """

    CONFIG = ConfigBlock("core.add_slack_variables")
    CONFIG.declare(
        'targets',
        ConfigValue(
            default=None,
            domain=target_list,
            description="target or list of targets to which slacks will be added",
            doc="This specifies the list of Constraints to add slack variables to.",
        ),
    )
    CONFIG.declare(
        'add_slack_objective',
        ConfigValue(
            default=True,
            domain=bool,
            description="Whether or not to change the model objective to minimizing "
            "the added slack variables.",
            doc="""
            Whether or not to change the problem objective to minimize the added slack
            variables. If True (the default), the original objective is deactivated
            and the transformation adds an objective to minimize the sum of the added
            (non-negative) slack variables. If False, the transformation does not
            change the model objective.
            """,
        ),
    )

    def __init__(self, **kwds):
        kwds['name'] = "add_slack_vars"
        super(AddSlackVariables, self).__init__(**kwds)

    def _apply_to(self, instance, **kwds):
        self._apply_to_impl(instance, **kwds)

    def _apply_to_impl(self, instance, **kwds):
        config = self.CONFIG(kwds.pop('options', {}))
        config.set_value(kwds)
        targets = config.targets

        trans_info = instance.private_data()

        if targets is None:
            constraintDatas = self._get_all_constraint_datas(instance)

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

        # create block where we can add slack variables safely
        xblockname = unique_component_name(instance, "_core_add_slack_variables")
        instance.add_component(xblockname, Block())
        xblock = instance.component(xblockname)

        obj_expr = 0
        for cons in constraintDatas:
            if (cons.lower is not None and cons.upper is not None) and value(
                cons.lower
            ) > value(cons.upper):
                # this is a structural infeasibility so slacks aren't going to
                # help:
                raise RuntimeError(
                    "Lower bound exceeds upper bound in constraint %s" % cons.name
                )
            if not cons.active:
                continue
            cons_name = cons.getname(fully_qualified=True)
            lower = cons.lower
            body = cons.body
            upper = cons.upper
            if lower is not None:
                # we add positive slack variable to body:
                # declare positive slack
                varName = "_slack_plus_" + cons_name
                posSlack = Var(within=NonNegativeReals)
                xblock.add_component(varName, posSlack)
                # add positive slack to body expression
                body += posSlack
                # penalize slack in objective
                obj_expr += posSlack
                trans_info.slack_variables[cons].append(posSlack)
                trans_info.relaxed_constraint[posSlack] = cons
            if upper is not None:
                # we subtract a positive slack variable from the body:
                # declare slack
                varName = "_slack_minus_" + cons_name
                negSlack = Var(within=NonNegativeReals)
                xblock.add_component(varName, negSlack)
                # add negative slack to body expression
                body -= negSlack
                # add slack to objective
                obj_expr += negSlack
                trans_info.slack_variables[cons].append(negSlack)
                trans_info.relaxed_constraint[negSlack] = cons

            cons.set_value((lower, body, upper))

        trans_info.summed_slacks_expr = obj_expr
        if config.add_slack_objective:
            # deactivate the objective
            for o in instance.component_data_objects(Objective):
                o.deactivate()

            # make a new objective that minimizes sum of slack variables
            xblock._slack_objective = Objective(expr=obj_expr)

    def _get_all_constraint_datas(self, model):
        components, unknown = categorize_valid_components(
            model,
            active=True,
            sort=SortComponents.deterministic,
            valid={
                Block,
                Expression,
                Var,
                Param,
                Suffix,
                Objective,
                # FIXME: Non-active components should not report as Active
                Set,
                RangeSet,
                Port,
            },
            targets={Constraint},
        )
        if unknown:
            raise ValueError(
                "The model ('%s') contains the following active components "
                "that the 'core.add_slack_variables' transformation does not "
                "know how to process:\n\t%s\nIf these components are Block-like "
                "(e.g., Disjuncts) and the intent is to add slacks on them, call "
                "the transformation on them directly."
                % (
                    model.name,
                    "\n\t".join(
                        sorted(
                            "%s:\n\t\t%s"
                            % (k, "\n\t\t".join(sorted(map(attrgetter('name'), v))))
                            for k, v in unknown.items()
                        )
                    ),
                )
            )
        if components[Constraint]:
            for block in components[Constraint]:
                for cons in block.component_data_objects(
                    Constraint,
                    active=True,
                    descend_into=False,
                    sort=SortComponents.deterministic,
                ):
                    yield cons

    def get_slack_variables(self, transformed_block, constraint):
        """Return the list of slack variables used to relax 'constraint.' Note
        that if 'constraint' is one-sided, there will be a single variable in
        the list, but if it is a ranged constraint (l <= expr <= u) or an
        equality, there will be two variables.

        Returns
        -------
        List of slack variables

        Parameters
        ----------
        transformed_block: ConcreteModel or Block
            The model or block that had the 'core.add_slack_variables'
            transformation applied to it
        constraint: Constraint
            A constraint that was relaxed by the transformation (either
            because no targets were specified or because it was a target)
        """
        slack_variables = transformed_block.private_data().slack_variables
        if constraint in slack_variables:
            return slack_variables[constraint]
        else:
            raise ValueError(
                f"It does not appear that {constraint.name} is a constraint "
                f"on model {transformed_block.name} that was relaxed by the "
                f"'core.add_slack_variables' transformation."
            )

    def get_relaxed_constraint(self, transformed_block, slack_var):
        """Return the constraint that 'slack_var' is used to relax.

        Returns
        -------
        Constraint

        Parameters
        -----------
        transformed_block: ConcreteModel or Block
            The model or block that had the 'core.add_slack_variables'
            transformation applied to it
        slack_var: Var
            A variable created by the 'core.add_slack_variables' transformation to
            relax a constraint.
        """
        relaxed_constraints = transformed_block.private_data().relaxed_constraint
        if slack_var in relaxed_constraints:
            return relaxed_constraints[slack_var]
        else:
            raise ValueError(
                f"It does not appear that {slack_var.name} is a slack variable "
                f"created by applying the 'core.add_slack_variables' transformation "
                f"to model {transformed_block.name}."
            )

    def get_summed_slacks_expr(self, transformed_block):
        """Return an expression summing all the slacks added to the model during the
        transformation. This would most commonly be used to add a penalty on non-zero
        slacks to an existing objective.

        Returns
        -------
        Expression

        Parameters
        ----------
        transformed_block: ConcreteModel or Block
            The model or block that had the 'core.add_slack_variables'
            transformation applied to it
        """
        expr = transformed_block.private_data().summed_slacks_expr
        if expr is None:
            raise ValueError(
                f"It does not appear that {transformed_block.name} is a model that "
                f"was transformed "
                f"by the 'core.add_slack_variables' transformation."
            )
        return expr
