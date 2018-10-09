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
from pyomo.core.base import Transformation, Block, Constraint
from pyomo.gdp import Disjunct, GDP_Error
from pyomo.core import TraversalStrategy, TransformationFactory
from pyomo.common.deprecation import deprecated

from six import itervalues

logger = logging.getLogger('pyomo.gdp')



@TransformationFactory.register('gdp.varmover', doc="Move indicator vars to top block.")
class HACK_GDP_Var_Mover(Transformation):
    """Move indicator vars to top block.

    HACK: this will move all indicator variables on the model to the top block
    so the writers can find them.

    """

    @deprecated(msg="The gdp.varmover transformation has been deprecated in "
                "favor of the gdp.reclassify transformation.")
    def _apply_to(self, instance, **kwds):
        assert not kwds
        count = 0
        disjunct_generator = instance.component_data_objects(
            Disjunct, descend_into=(Block, Disjunct))
        for disjunct in disjunct_generator:
            count += 1
            var = disjunct.indicator_var
            var.doc = "%s(Moved from %s)" % (
                var.doc + " " if var.doc else "", var.name, )
            disjunct.del_component(var)
            instance.add_component("_gdp_moved_IV_%s" % (count,), var)


@TransformationFactory.register('gdp.reclassify',
          doc="Reclassify Disjuncts to Blocks.")
class HACK_GDP_Disjunct_Reclassifier(Transformation):
    """Reclassify Disjuncts to Blocks.

    HACK: this will reclassify all Disjuncts to Blocks so the current writers
    can find the variables

    """

    def _apply_to(self, instance, **kwds):
        assert not kwds  # no keywords expected to the transformation
        disjunct_generator = instance.component_objects(
            Disjunct, descend_into=(Block, Disjunct),
            descent_order=TraversalStrategy.PostfixDFS)
        for disjunct_component in disjunct_generator:
            # Check that the disjuncts being reclassified are all relaxed or
            # are not on an active block.
            for disjunct in itervalues(disjunct_component._data):
                if (disjunct.active and
                        self._disjunct_not_relaxed(disjunct) and
                        self._disjunct_on_active_block(disjunct) and
                        self._disjunct_not_fixed_true(disjunct)):
                    raise GDP_Error("""
                    Reclassifying active Disjunct "%s" as a Block.  This
                    is generally an error as it indicates that the model
                    was not completely relaxed before applying the
                    gdp.reclassify transformation""" % (disjunct.name,))

            # Reclassify this disjunct as a block
            disjunct_component.parent_block().reclassify_component_type(
                disjunct_component, Block)
            disjunct_component._activate_without_unfixing_indicator()

            # Deactivate all constraints.  Note that we only need to
            # descend into blocks: we will catch disjuncts in the outer
            # loop.
            #
            # Note that we defer this until AFTER we reactivate the
            # block, as the component_objects generator will not
            # return anything when active=True and the block is
            # deactivated.
            for disjunct in itervalues(disjunct_component._data):
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
        return not getattr(
            disjunct, '_gdp_transformation_info', {}).get('relaxed', False)

    def _disjunct_on_active_block(self, disjunct):
        # Check first to make sure that the disjunct is not a
        # descendent of an inactive Block or fixed and deactivated
        # Disjunct, before raising a warning.
        parent_block = disjunct.parent_block()
        while parent_block is not None:
            if parent_block.type() is Block and not parent_block.active:
                return False
            elif (parent_block.type() is Disjunct and not parent_block.active
                  and parent_block.indicator_var.value == 0
                  and parent_block.indicator_var.fixed):
                return False
            else:
                # Step up one level in the hierarchy
                parent_block = parent_block.parent_block()
                continue
        return True
