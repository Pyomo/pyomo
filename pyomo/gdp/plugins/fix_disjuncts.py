# -*- coding: UTF-8 -*-
"""Transformation to fix disjuncts.

This transformation looks for active disjunctions in the passed model instance
and fixes the participating disjuncts based on their current indicator_var
values. Active disjuncts are transformed to Block and inactive disjuncts
(indicator_var = 0) are transformed to Block with all of their constituent
constraints and disjunctions deactivated.

"""

import logging
import textwrap
from math import fabs

from pyomo.core.base import Transformation
from pyomo.core.base.block import Block, _BlockData
from pyomo.core.base.constraint import Constraint
from pyomo.core.kernel.numvalue import value
from pyomo.core.kernel.component_set import ComponentSet
from pyomo.gdp import GDP_Error
from pyomo.gdp.disjunct import (Disjunct, Disjunction, _DisjunctData,
                                _DisjunctionData)
from pyomo.util.plugin import alias
from six import itervalues

logger = logging.getLogger('pyomo.gdp.fix_disjuncts')


class GDP_Disjunct_Fixer(Transformation):
    """Fix disjuncts to their current logical values.

    This reclassifies all disjuncts as ctype Block and deactivates the
    constraints and disjunctions within inactive disjuncts.

    """

    def __init__(self, *args, **kwargs):
        # TODO This uses an integer tolerance. At some point, these should be
        # standardized.
        self.integer_tolerance = kwargs.pop('int_tol', 1E-6)
        super(GDP_Disjunct_Fixer, self).__init__(*args, **kwargs)
        self._transformedDisjuncts = ComponentSet()

    alias('gdp.fix_disjuncts',
          doc=textwrap.fill(textwrap.dedent(__doc__.strip())))

    def _apply_to(self, instance):
        """Apply the transformation to the given instance.

        The instance ctype is expected to be Block, Disjunct, or Disjunction.
        For a Block or Disjunct, the transformation will fix all disjuncts
        found in disjunctions within the container.

        """
        t = instance
        if not t.active:
            return  # do nothing for inactive containers

        # screen for allowable instance types
        if (type(t) not in (_DisjunctData, _BlockData, _DisjunctionData) and
                t.type() not in (Disjunct, Block, Disjunction)):
            raise GDP_Error(
                "Target %s was not a Block, Disjunct, or Disjunction. "
                "It was of type %s and can't be transformed."
                % (t.name, type(t)))

        # if the object is indexed, transform all of its _ComponentData
        if t.is_indexed():
            for obj in itervalues(t):
                self._transformObject(obj)
        else:
            self._transformObject(t)

    def _transformObject(self, obj):
        # If the object is a disjunction, transform it.
        if obj.type() == Disjunction and not obj.is_indexed():
            self._transformDisjunctionData(obj)
        # Otherwise, treat it like a container and transform its contents.
        else:
            self._transformContainer(obj)

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

        # Process the disjuncts associatd with the disjunction that have not
        # already been transformed.
        for disj in ComponentSet(disjunction.disjuncts) - self._transformedDisjuncts:
            self._transformDisjunctData(disj)
        # Update the set of transformed disjuncts with those from this
        # disjunction
        self._transformedDisjuncts.update(disjunction.disjuncts)

    def _transformDisjunctData(self, obj):
        """Fix the disjunct either to a Block or a deactivated Block."""
        if fabs(value(obj.indicator_var) - 1) <= self.integer_tolerance:
            # Disjunct is active. Convert to Block.
            obj.parent_block().reclassify_component_type(obj, Block)
            obj.indicator_var.fix(1)
            # Process the components attached to this disjunct.
            self._transformContainer(obj)
        elif fabs(value(obj.indicator_var)) <= self.integer_tolerance:
            obj.parent_block().reclassify_component_type(obj, Block)
            obj.indicator_var.fix(0)
            # Deactivate all constituent constraints and disjunctions
            # HACK I do not deactivate the whole block because some writers
            # do not look for variables in deactivated blocks.
            for constr in obj.component_objects(
                    ctype=(Constraint, Disjunction),
                    active=True, descend_into=True):
                constr.deactivate()
        else:
            raise ValueError(
                'Non-binary indicator variable value %s for disjunct %s'
                % (obj.name, value(obj.indicator_var)))

    def _transformContainer(self, obj):
        """Find all disjunctions in the container and transform them."""
        for disjunction in obj.component_data_objects(
                ctype=Disjunction, active=True,
                descend_into=Block):
            self._transformDisjunctionData(disjunction)
