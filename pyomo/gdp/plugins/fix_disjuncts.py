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
from pyomo.core.base.block import Block, TraversalStrategy, _BlockData
from pyomo.core.base.constraint import Constraint
from pyomo.core.kernel.numvalue import value
from pyomo.core.kernel.component_set import ComponentSet
from pyomo.gdp import GDP_Error
from pyomo.gdp.disjunct import (Disjunct, Disjunction, _DisjunctData,
                                _DisjunctionData)
from pyomo.util.plugin import alias
from six import itervalues

__author__ = "Qi Chen <https://github.com/qtothec>"

logger = logging.getLogger('pyomo.core')


class GDP_Disjunct_Fixer(Transformation):
    """Fix disjuncts to Blocks."""

    def __init__(self, *args, **kwargs):
        self.integer_tolerance = kwargs.pop('int_tol', 1E-6)
        super(GDP_Disjunct_Fixer, self).__init__(*args, **kwargs)
        self._transformedDisjuncts = ComponentSet()

    alias('gdp.fix_disjuncts',
          doc=textwrap.fill(textwrap.dedent(__doc__.strip())))

    def _apply_to(self, instance):
        t = instance
        if not t.active:
            return

        if (type(t) not in (_DisjunctData, _BlockData, _DisjunctionData) and
                t.type() not in (Disjunct, Block, Disjunction)):
            raise GDP_Error(
                "Target %s was not a Block, Disjunct, or Disjunction. "
                "It was of type %s and can't be transformed."
                % (t.name, type(t)))

        if t.is_indexed():
            for obj in itervalues(t):
                self._transformObject(obj)
        else:
            self._transformObject(t)

    def _transformObject(self, obj):
        if type(obj) is _DisjunctionData:
            self._transformDisjunctionData(obj)
        self._transformContainer(obj)

    def _transformDisjunctionData(self, disjunction):
        xor = disjunction.parent_component().xor
        logical_sum = sum(value(disj.indicator_var)
                          for disj in disjunction.disjuncts)
        if xor and not logical_sum == 1:
            raise GDP_Error(
                "Disjunction %s violated. "
                "Expected 1 disjunct to be active, but %s were active."
                % (disjunction.name, logical_sum))
        elif not logical_sum >= 1:
            raise GDP_Error(
                "Disjunction %s violated. "
                "Expected at least 1 disjunct to be active, "
                "but none were active.")
        else:
            disjunction.deactivate()

        for disj in disjunction.disjuncts - self._transformedDisjuncts:
            self._transformDisjunctData(disj)
        self._transformedDisjuncts.update(disjunction.disjuncts)

    def _transformDisjunctData(self, obj):
        if fabs(value(obj.indicator_var) - 1) <= self.integer_tolerance:
            # Disjunct is active. Convert to Block.
            obj.parent_block().reclassify_component_type(obj, Block)
            obj.indicator_var.fix(1)
            self._transformContainer(obj)
        elif fabs(value(obj.indicator_var)) <= self.integer_tolerance:
            obj.parent_block().reclassify_component_type(obj, Block)
            obj.indicator_var.fix(0)
            # Deactivate all constituent constraints and disjunctions
            for constr in obj.component_objects(
                    ctype=(Constraint, Disjunction),
                    active=True, descend_into=True):
                constr.deactivate()
        else:
            raise ValueError(
                'Non-binary indicator variable value %s for disjunct %s'
                % obj.name, value(obj.indicator_var))

    def _transformContainer(self, obj):
        for disjunction in obj.component_data_objects(
                ctype=Disjunction, active=True,
                descend_into=Block):
            self._transformDisjunctionData(disjunction)
