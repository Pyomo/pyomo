#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Collection of GDP-related hacks.

Hacks for dealing with the fact that solver writers may sometimes fail to
detect variables inside of Disjuncts or deactivated Blocks.
"""

import logging
import textwrap
from pyomo.common.plugin import alias
from pyomo.core.base import Transformation, Block, Constraint, Reference
from pyomo.gdp import Disjunct, GDP_Error, Disjunction
from pyomo.core import TraversalStrategy, TransformationFactory, SimpleVar
from pyomo.core.base.indexed_component import ActiveIndexedComponent
from pyomo.common.deprecation import deprecated
from pyomo.common.config import ConfigBlock, ConfigValue

from six import itervalues

logger = logging.getLogger('pyomo.gdp')


@TransformationFactory.register('gdp.reference_indicator_vars', 
                                doc = "Create references to indicator_vars "
                                "on the Disjunct's transformation block "
                                "so that the writers will pick them up.")
class HACK_GDP_Reference_Indicator_Vars(Transformation):
    """
    Creates References for all indicator_vars on the transformation block of
    all transformed Disjuncts. This means that the writers will pick up these
    variables despite not looking for them on Disjuncts.
    """
    CONFIG = ConfigBlock("gdp.reference_indicator_vars")
    CONFIG.declare('check_model_algebraic', ConfigValue(
        default=False,
        domain=bool,
        description="Whether or not to check that the model is completely "
        "algebraic.",
        doc="""
        This specifies whether or not we should do any error checking to see 
        if the model is completely algebraic. Setting this to True provides
        backwards compatibility with the deprecated HACK_GDP_DisjunctReclassifier
        transformation because it will mean this transformation checks for 
        untransformed Disjuncts. Set to False if this transformation is called after
        using targets, when you do not expect the GDP model to be complete transformed
        yet.
        """
    ))

    def _apply_to(self, instance, **kwds):
        config = self.CONFIG(kwds.pop('options', {}))
        config.set_value(kwds)
        check_algebraic = config.check_model_algebraic

        # First, do a couple checks in order to give a more useful error
        # message. (This is all for the check-for-algebraic hack
        # below. Completely unncessary when the writers are fixed.)
        disjunction_set = {i for i in instance.component_data_objects(
            Disjunction, descend_into=(Block, Disjunct), active=None)}
        active_disjunction_set = {i for i in instance.component_data_objects(
            Disjunction, descend_into=(Block, Disjunct), active=True)}
        disjuncts_in_disjunctions = set()
        for i in disjunction_set:
            disjuncts_in_disjunctions.update(i.disjuncts)
        disjuncts_in_active_disjunctions = set()
        for i in active_disjunction_set:
            disjuncts_in_active_disjunctions.update(i.disjuncts)
        
        for disjunct in instance.component_data_objects(
                Disjunct, descend_into=(Block, Disjunct),
                descent_order=TraversalStrategy.PostfixDFS):
            # check if it's relaxed
            if disjunct.transformation_block is not None:
                disjBlock = disjunct.transformation_block()
                if disjBlock.component("indicator_var") is None:
                    disjBlock.indicator_var = Reference(disjunct.indicator_var)
            # It's not transformed, check if we should complain
            elif check_algebraic and disjunct.active and \
                 self._disjunct_not_fixed_true(disjunct) and \
                 self._disjunct_on_active_block(disjunct):
                # If someone thinks they've transformed the whole instance, but
                # there is still an active Disjunct on the model, we will warn
                # them. (Though this is a HACK becuase in the future this should
                # be the writers' job.)
                if disjunct not in disjuncts_in_disjunctions:
                    raise GDP_Error('Disjunct "%s" is currently active, '
                                    'but was not found in any Disjunctions. '
                                    'This is generally an error as the model '
                                    'has not been fully relaxed to a '
                                    'pure algebraic form.' % (disjunct.name,))
                elif disjunct not in disjuncts_in_active_disjunctions:
                    raise GDP_Error('Disjunct "%s" is currently active. While '
                                    'it participates in a Disjunction, '
                                    'that Disjunction is currently deactivated. '
                                    'This is generally an error as the '
                                    'model has not been fully relaxed to a pure '
                                    'algebraic form. Did you deactivate '
                                    'the Disjunction without addressing the '
                                    'individual Disjuncts?' % (disjunct.name,))
                else:
                    raise GDP_Error('Disjunct "%s" is currently active. It must be '
                                    'transformed or deactivated before solving the '
                                    'model.' % (disjunct.name,))

    def _disjunct_not_fixed_true(self, disjunct):
        # Return true if the disjunct indicator variable is not fixed to True
        return not (disjunct.indicator_var.fixed and
                    disjunct.indicator_var.value == 1)

    def _disjunct_on_active_block(self, disjunct):
        # Check first to make sure that the disjunct is not a
        # descendent of an inactive Block or fixed and deactivated
        # Disjunct, before raising a warning.
        parent_block = disjunct.parent_block()
        while parent_block is not None:
            # deactivated Block
            if parent_block.ctype is Block and not parent_block.active:
                return False
            # properly deactivated Disjunct
            elif (parent_block.ctype is Disjunct and not parent_block.active
                  and parent_block.indicator_var.value == 0
                  and parent_block.indicator_var.fixed):
                return False
            else:
                # Step up one level in the hierarchy
                parent_block = parent_block.parent_block()
                continue
        return True
                
@TransformationFactory.register('gdp.reclassify',
          doc="Reclassify Disjuncts to Blocks.")
class HACK_GDP_Disjunct_Reclassifier(Transformation):
    """Reclassify Disjuncts to Blocks.

    HACK: this will reclassify all Disjuncts to Blocks so the current writers
    can find the variables

    """
    @deprecated(msg="The gdp.reclasify transformation has been deprecated in "
                "favor of the gdp.reference_indicator_vars transformation.",
                version='5.6.10')
    def _apply_to(self, instance, **kwds):
        assert not kwds  # no keywords expected to the transformation
        disjunct_generator = instance.component_objects(
            Disjunct, descend_into=(Block, Disjunct),
            descent_order=TraversalStrategy.PostfixDFS)
        for disjunct_component in disjunct_generator:
            # Check that the disjuncts being reclassified are all relaxed or
            # are not on an active block.
            for disjunct in itervalues(disjunct_component):
                if (disjunct.active and
                        self._disjunct_not_relaxed(disjunct) and
                        self._disjunct_on_active_block(disjunct) and
                        self._disjunct_not_fixed_true(disjunct)):

                    # First, do a couple checks in order to give a more
                    # useful error message
                    disjunction_set = {i for i in
                                       instance.component_data_objects(
                                           Disjunction, descend_into=True,
                                           active=None)}
                    active_disjunction_set = {i for i in
                                              instance.component_data_objects(
                                                  Disjunction,
                                                  descend_into=True,
                                                  active=True)}
                    disjuncts_in_disjunctions = set()
                    for i in disjunction_set:
                        disjuncts_in_disjunctions.update(i.disjuncts)
                    disjuncts_in_active_disjunctions = set()
                    for i in active_disjunction_set:
                        disjuncts_in_active_disjunctions.update(i.disjuncts)

                    if disjunct not in disjuncts_in_disjunctions:
                        raise GDP_Error('Disjunct "%s" is currently active, '
                                        'but was not found in any Disjunctions. '
                                        'This is generally an error as the model '
                                        'has not been fully relaxed to a '
                                        'pure algebraic form.' % (disjunct.name,))
                    elif disjunct not in disjuncts_in_active_disjunctions:
                        raise GDP_Error('Disjunct "%s" is currently active. While '
                                        'it participates in a Disjunction, '
                                        'that Disjunction is currently deactivated. '
                                        'This is generally an error as the '
                                        'model has not been fully relaxed to a pure '
                                        'algebraic form. Did you deactivate '
                                        'the Disjunction without addressing the '
                                        'individual Disjuncts?' % (disjunct.name,))
                    else:
                        raise GDP_Error("""
                        Reclassifying active Disjunct "%s" as a Block.  This
                        is generally an error as it indicates that the model
                        was not completely relaxed before applying the
                        gdp.reclassify transformation""" % (disjunct.name,))

            # Reclassify this disjunct as a block
            disjunct_component.parent_block().reclassify_component_type(
                disjunct_component, Block)
            # HACK: activate teh block, but do not activate the
            # _BlockData objects
            super(ActiveIndexedComponent, disjunct_component).activate()

            # Deactivate all constraints.  Note that we only need to
            # descend into blocks: we will catch disjuncts in the outer
            # loop.
            #
            # Note that we defer this until AFTER we reactivate the
            # block, as the component_objects generator will not
            # return anything when active=True and the block is
            # deactivated.
            for disjunct in itervalues(disjunct_component._data):
                if self._disjunct_not_relaxed(disjunct):
                    disjunct._deactivate_without_fixing_indicator()
                else:
                    disjunct._activate_without_unfixing_indicator()

                cons_in_disjunct = disjunct.component_objects(
                    Constraint, descend_into=Block, active=True)
                for con in cons_in_disjunct:
                    con.deactivate()

    def _disjunct_not_fixed_true(self, disjunct):
        # Return true if the disjunct indicator variable is not fixed to True
        return not (disjunct.indicator_var.fixed and
                    disjunct.indicator_var.value == 1)

    def _disjunct_not_relaxed(self, disjunct):
        # Return True if the disjunct was not relaxed by a transformation.
        return disjunct.transformation_block is None
        
    def _disjunct_on_active_block(self, disjunct):
        # Check first to make sure that the disjunct is not a
        # descendent of an inactive Block or fixed and deactivated
        # Disjunct, before raising a warning.
        parent_block = disjunct.parent_block()
        while parent_block is not None:
            if parent_block.ctype is Block and not parent_block.active:
                return False
            elif (parent_block.ctype is Disjunct and not parent_block.active
                  and parent_block.indicator_var.value == 0
                  and parent_block.indicator_var.fixed):
                return False
            else:
                # Step up one level in the hierarchy
                parent_block = parent_block.parent_block()
                continue
        return True
