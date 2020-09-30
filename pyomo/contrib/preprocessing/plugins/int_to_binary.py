"""Transformation to reformulate integer variables into binary."""
from __future__ import division

from math import floor, log
import logging

from pyomo.common.collections import ComponentSet
from pyomo.common.config import ConfigBlock, ConfigValue, In
from pyomo.core import TransformationFactory, Var, Block, Constraint, Any, Binary, value, RangeSet, \
    Reals
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.gdp import Disjunct
from pyomo.core.expr.current import identify_variables
from pyomo.common.modeling import unique_component_name

logger = logging.getLogger('pyomo.contrib.preprocessing')


@TransformationFactory.register(
    'contrib.integer_to_binary',
    doc="Reformulate integer variables into binary variables.")
class IntegerToBinary(IsomorphicTransformation):
    """Reformulate integer variables to binary variables and constraints.

    This transformation may be safely applied multiple times to the same model.
    """

    CONFIG = ConfigBlock("contrib.integer_to_binary")
    CONFIG.declare("strategy", ConfigValue(
        default='base2',
        domain=In('base2', ),
        description="Reformulation method",
        # TODO: eventually we will support other methods, but not yet.
    ))
    CONFIG.declare("ignore_unused", ConfigValue(
        default=False,
        domain=bool,
        description="Ignore variables that do not appear in (potentially) active constraints. "
        "These variables are unlikely to be passed to the solver."
    ))
    CONFIG.declare("relax_integrality", ConfigValue(
        default=True,
        domain=bool,
        description="Relax the integrality of the integer variables "
        "after adding in the binary variables and constraints."
    ))

    def _apply_to(self, model, **kwds):
        """Apply the transformation to the given model."""
        config = self.CONFIG(kwds.pop('options', {}))
        config.set_value(kwds)

        integer_vars = list(
            v for v in model.component_data_objects(
                ctype=Var, descend_into=(Block, Disjunct))
            if v.is_integer() and not v.is_binary() and not v.fixed)
        if len(integer_vars) == 0:
            logger.info(
                "Model has no free integer variables. No reformulation needed.")
            return

        vars_on_constr = ComponentSet()
        for c in model.component_data_objects(
                ctype=Constraint, descend_into=(Block, Disjunct), active=True):
            vars_on_constr.update(v for v in identify_variables(c.body, include_fixed=False)
                                  if v.is_integer())

        if config.ignore_unused:
            num_vars_not_on_constr = len(integer_vars) - len(vars_on_constr)
            if num_vars_not_on_constr > 0:
                logger.info(
                    "%s integer variables on the model are not attached to any constraints. "
                    "Ignoring unused variables."
                )
            integer_vars = list(vars_on_constr)

        logger.info(
            "Reformulating integer variables using the %s strategy."
            % config.strategy)

        # Set up reformulation block
        blk_name = unique_component_name(model, "_int_to_binary_reform")
        reform_block = Block(
            doc="Holds variables and constraints for reformulating "
                "integer variables to binary variables."
        )
        setattr(model, blk_name, reform_block)

        reform_block.int_var_set = RangeSet(0, len(integer_vars) - 1)

        reform_block.new_binary_var = Var(
            Any, domain=Binary, dense=False, initialize=0,
            doc="Binary variable with index (int_var_idx, idx)")
        reform_block.integer_to_binary_constraint = Constraint(
            reform_block.int_var_set,
            doc="Equality constraints mapping the binary variable values "
                "to the integer variable value.")

        # check that variables are bounded
        for idx, int_var in enumerate(integer_vars):
            if not (int_var.has_lb() and int_var.has_ub()):
                raise ValueError(
                    "Integer variable %s is missing an "
                    "upper or lower bound. LB: %s; UB: %s. "
                    "Integer to binary reformulation does not support unbounded integer variables."
                    % (int_var.name, int_var.lb, int_var.ub))
            # do the reformulation
            highest_power = int(floor(log(value(int_var.ub - int_var.lb), 2)))
            # TODO potentially fragile due to floating point

            reform_block.integer_to_binary_constraint.add(
                idx, expr=int_var == sum(
                    reform_block.new_binary_var[idx, pwr] * (2 ** pwr)
                    for pwr in range(0, highest_power + 1))
                + int_var.lb)

            # Relax the original integer variable
            if config.relax_integrality:
                int_var.domain = Reals

        logger.info(
            "Reformulated %s integer variables using "
            "%s binary variables and %s constraints."
            % (len(integer_vars), len(reform_block.new_binary_var),
               len(reform_block.integer_to_binary_constraint)))
