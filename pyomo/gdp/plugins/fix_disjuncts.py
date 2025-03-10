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

# -*- coding: utf-8 -*-
"""Transformation to fix and enforce disjunct True/False status."""

import logging
from math import fabs

from pyomo.common.config import ConfigDict, ConfigValue, NonNegativeFloat
from pyomo.contrib.cp.transform.logical_to_disjunctive_program import (
    LogicalToDisjunctive,
)
from pyomo.core.base import Transformation, TransformationFactory
from pyomo.core.base.block import Block
from pyomo.core.expr.numvalue import value
from pyomo.gdp import GDP_Error
from pyomo.gdp.disjunct import Disjunct, Disjunction
from pyomo.gdp.plugins.bigm import BigM_Transformation

logger = logging.getLogger('pyomo.gdp.fix_disjuncts')


def _transformation_name_or_object(transformation_name_or_object):
    if isinstance(transformation_name_or_object, Transformation):
        return transformation_name_or_object
    xform = TransformationFactory(transformation_name_or_object)
    if xform is None:
        raise ValueError(
            "Expected valid name for a registered Pyomo transformation. "
            "\n\tRecieved: %s" % transformation_name_or_object
        )
    return xform


@TransformationFactory.register(
    'gdp.fix_disjuncts',
    doc="""Fix disjuncts to their current Boolean values and transforms any
    LogicalConstraints and BooleanVars so that the resulting model is a
    (MI)(N)LP.""",
)
class GDP_Disjunct_Fixer(Transformation):
    """Fix disjuncts to their current Boolean values.

    This reclassifies all disjuncts in the passed model instance as ctype Block
    and deactivates the constraints and disjunctions within inactive disjuncts.
    In addition, it transforms relevant LogicalConstraints and BooleanVars so
    that the resulting model is a (MI)(N)LP (where it is only mixed-integer
    if the model contains integer-domain Vars or BooleanVars which were not
    indicator_vars of Disjuncs.
    """

    def __init__(self, **kwargs):
        # TODO This uses an integer tolerance. At some point, these should be
        # standardized.
        super(GDP_Disjunct_Fixer, self).__init__(**kwargs)

    CONFIG = ConfigDict("gdp.fix_disjuncts")
    CONFIG.declare(
        "GDP_to_MIP_transformation",
        ConfigValue(
            default=BigM_Transformation(),
            domain=_transformation_name_or_object,
            description="The name of the transformation to call after the "
            "'logical_to_disjunctive' transformation in order to finish "
            "transforming to a MI(N)LP.",
            doc="""
        If there are no logical constraints on the model being transformed,
        this option is not used. However, if there are logical constraints
        that involve mixtures of Boolean and integer variables, this option
        specifies the transformation to use to transform the model with fixed
        Disjuncts to a MI(N)LP. Uses 'gdp.bigm' by default.
        """,
        ),
    )

    def _apply_to(self, model, **kwds):
        """Fix all disjuncts in the given model and reclassify them to
        Blocks."""
        config = self.config = self.CONFIG(kwds.pop('options', {}))
        config.set_value(kwds)

        self._transformContainer(model)

        # Reclassify all disjuncts
        for disjunct_object in model.component_objects(
            Disjunct, descend_into=(Block, Disjunct)
        ):
            disjunct_object.parent_block().reclassify_component_type(
                disjunct_object, Block
            )

        # Transform any remaining logical stuff
        TransformationFactory('contrib.logical_to_disjunctive').apply_to(model)
        # Transform anything disjunctive that the above created:
        config.GDP_to_MIP_transformation.apply_to(model)

    def _transformContainer(self, obj):
        """Find all disjuncts in the container and transform them."""
        for disjunct in obj.component_data_objects(
            ctype=Disjunct, active=True, descend_into=True
        ):
            _bool = disjunct.indicator_var
            if _bool.value is None:
                raise GDP_Error(
                    "The value of the indicator_var of "
                    "Disjunct '%s' is None. All indicator_vars "
                    "must have values before calling "
                    "'fix_disjuncts'." % disjunct.name
                )
            elif _bool.value:
                disjunct.indicator_var.fix(True)
                self._transformContainer(disjunct)
            else:
                # Deactivating fixes the indicator_var to False
                disjunct.deactivate()

        for disjunction in obj.component_data_objects(
            ctype=Disjunction, active=True, descend_into=True
        ):
            self._transformDisjunctionData(disjunction)

    def _transformDisjunctionData(self, disjunction):
        # the sum of all the indicator variable values of disjuncts in the
        # disjunction
        logical_sum = sum(
            value(disj.binary_indicator_var) for disj in disjunction.disjuncts
        )

        # Check that the disjunctions are not being fixed to an infeasible
        # realization.
        if disjunction.xor and not logical_sum == 1:
            # for XOR disjunctions, the sum of all disjunct values should be 1
            raise GDP_Error(
                "Disjunction %s violated. "
                "Expected 1 disjunct to be active, but %s were active."
                % (disjunction.name, logical_sum)
            )
        elif not logical_sum >= 1:
            # for non-XOR disjunctions, the sum of all disjunct values should
            # be at least 1
            raise GDP_Error(
                "Disjunction %s violated. "
                "Expected at least 1 disjunct to be active, "
                "but none were active."
            )
        else:
            # disjunction is in feasible realization. Deactivate it.
            disjunction.deactivate()
