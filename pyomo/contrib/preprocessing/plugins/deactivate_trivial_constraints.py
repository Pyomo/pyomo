# -*- coding: utf-8 -*-
#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Transformation to deactivate trivial constraints."""
import logging

from pyomo.common.collections import ComponentSet
from pyomo.common.config import (
    ConfigBlock,
    ConfigValue,
    NonNegativeFloat,
    document_kwargs_from_configdict,
)
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.transformation import TransformationFactory
from pyomo.core.expr.numvalue import value
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.repn import generate_standard_repn

logger = logging.getLogger('pyomo.contrib.preprocessing')


@TransformationFactory.register(
    'contrib.deactivate_trivial_constraints', doc="Deactivate trivial constraints."
)
@document_kwargs_from_configdict('CONFIG')
class TrivialConstraintDeactivator(IsomorphicTransformation):
    """Deactivates trivial constraints.

    Trivial constraints take form :math:`k_1 = k_2` or :math:`k_1 \\leq k_2`,
    where :math:`k_1` and :math:`k_2` are constants. These constraints
    typically arise when variables are fixed.

    Keyword arguments below are specified for the ``apply_to`` and
    ``create_using`` functions.

    """

    CONFIG = ConfigBlock("TrivialConstraintDeactivator")
    CONFIG.declare(
        "tmp",
        ConfigValue(
            default=False,
            domain=bool,
            description="True to store a set of transformed constraints for future"
            " reversion of the transformation.",
        ),
    )
    CONFIG.declare(
        "ignore_infeasible",
        ConfigValue(
            default=False,
            domain=bool,
            description="True to skip over trivial constraints that are "
            "infeasible instead of raising an InfeasibleConstraintException.",
        ),
    )
    CONFIG.declare(
        "return_trivial",
        ConfigValue(
            default=[],
            description="a list to which the deactivated trivial"
            "constraints are appended (side effect)",
        ),
    )
    CONFIG.declare(
        "tolerance",
        ConfigValue(
            default=1e-13,
            domain=NonNegativeFloat,
            description="tolerance on constraint violations",
        ),
    )

    def _apply_to(self, instance, **kwargs):
        config = self.CONFIG(kwargs)
        if config.tmp and not hasattr(instance, '_tmp_trivial_deactivated_constrs'):
            instance._tmp_trivial_deactivated_constrs = ComponentSet()
        elif config.tmp:
            logger.warning(
                'Deactivating trivial constraints on the block {} for which '
                'trivial constraints were previously deactivated. '
                'Reversion will affect all deactivated constraints.'.format(
                    instance.name
                )
            )

        # Trivial constraints are those that do not contain any variables, ie.
        # the polynomial degree is 0
        for constr in instance.component_data_objects(
            ctype=Constraint, active=True, descend_into=True
        ):
            repn = generate_standard_repn(constr.body)
            if not repn.is_constant():
                # This constraint is not trivial
                continue

            # We need to check each constraint to sure that it is not violated.
            constr_lb = value(constr.lower) if constr.has_lb() else float('-inf')
            constr_ub = value(constr.upper) if constr.has_ub() else float('inf')
            constr_value = repn.constant

            # Check if the lower bound is violated outside a given tolerance
            if constr_value + config.tolerance <= constr_lb:
                if config.ignore_infeasible:
                    continue
                else:
                    raise InfeasibleConstraintException(
                        'Trivial constraint {} violates LB {} ≤ BODY {}.'.format(
                            constr.name, constr_lb, constr_value
                        )
                    )

            # Check if the upper bound is violated outside a given tolerance
            if constr_value >= constr_ub + config.tolerance:
                if config.ignore_infeasible:
                    continue
                else:
                    raise InfeasibleConstraintException(
                        'Trivial constraint {} violates BODY {} ≤ UB {}.'.format(
                            constr.name, constr_value, constr_ub
                        )
                    )

            # Constraint is not infeasible. Deactivate it.
            if config.tmp:
                instance._tmp_trivial_deactivated_constrs.add(constr)
            config.return_trivial.append(constr)
            constr.deactivate()

    def revert(self, instance):
        """Revert constraints deactivated by the transformation.

        Args:
            instance: the model instance on which trivial constraints were
                earlier deactivated.
        """
        for constr in instance._tmp_trivial_deactivated_constrs:
            constr.activate()
        del instance._tmp_trivial_deactivated_constrs
