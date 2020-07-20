# -*- coding: UTF-8 -*-
"""Transformation to fix and enforce disjunct True/False status."""

import logging
from math import fabs

from pyomo.common.config import ConfigBlock, ConfigValue, NonNegativeFloat
from pyomo.core.base import Transformation, TransformationFactory
from pyomo.core.base.block import Block
from pyomo.core.expr.numvalue import value
from pyomo.gdp import GDP_Error
from pyomo.gdp.disjunct import Disjunct, Disjunction

logger = logging.getLogger('pyomo.gdp.fix_disjuncts')


@TransformationFactory.register(
    'gdp.fix_disjuncts',
    doc="Fix disjuncts to their current Boolean values.")
class GDP_Disjunct_Fixer(Transformation):
    """Fix disjuncts to their current Boolean values.

    This reclassifies all disjuncts in the passed model instance as ctype Block and deactivates the
    constraints and disjunctions within inactive disjuncts.

    """

    def __init__(self, **kwargs):
        # TODO This uses an integer tolerance. At some point, these should be
        # standardized.
        super(GDP_Disjunct_Fixer, self).__init__(**kwargs)

    CONFIG = ConfigBlock("gdp.fix_disjuncts")
    CONFIG.declare('integer_tolerance', ConfigValue(
        default=1E-6,
        domain=NonNegativeFloat,
        description="tolerance on binary variable 0, 1 values"
    ))

    def _apply_to(self, model, **kwds):
        """Fix all disjuncts in the given model and reclassify them to Blocks."""
        config = self.config = self.CONFIG(kwds.pop('options', {}))
        config.set_value(kwds)

        self._transformContainer(model)

        # Reclassify all disjuncts
        for disjunct_object in model.component_objects(Disjunct, descend_into=(Block, Disjunct)):
            disjunct_object.parent_block().reclassify_component_type(disjunct_object, Block)

    def _transformContainer(self, obj):
        """Find all disjuncts in the container and transform them."""
        for disjunct in obj.component_data_objects(ctype=Disjunct, active=True, descend_into=True):
            if fabs(value(disjunct.indicator_var) - 1) <= self.config.integer_tolerance:
                disjunct.indicator_var.fix(1)
                self._transformContainer(disjunct)
            elif fabs(value(disjunct.indicator_var)) <= self.config.integer_tolerance:
                disjunct.deactivate()
            else:
                raise ValueError(
                    'Non-binary indicator variable value %s for disjunct %s'
                    % (disjunct.name, value(disjunct.indicator_var)))

        for disjunction in obj.component_data_objects(ctype=Disjunction, active=True, descend_into=True):
            self._transformDisjunctionData(disjunction)

    def _transformDisjunctionData(self, disjunction):
        # the sum of all the indicator variable values of disjuncts in the
        # disjunction
        logical_sum = sum(value(disj.indicator_var)
                          for disj in disjunction.disjuncts)

        # Check that the disjunctions are not being fixed to an infeasible
        # realization.
        if disjunction.xor and not logical_sum == 1:
            # for XOR disjunctions, the sum of all disjunct values should be 1
            raise GDP_Error(
                "Disjunction %s violated. "
                "Expected 1 disjunct to be active, but %s were active."
                % (disjunction.name, logical_sum))
        elif not logical_sum >= 1:
            # for non-XOR disjunctions, the sum of all disjunct values should
            # be at least 1
            raise GDP_Error(
                "Disjunction %s violated. "
                "Expected at least 1 disjunct to be active, "
                "but none were active.")
        else:
            # disjunction is in feasible realization. Deactivate it.
            disjunction.deactivate()
