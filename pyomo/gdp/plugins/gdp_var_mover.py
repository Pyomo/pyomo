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
from pyomo.util.plugin import alias
from pyomo.core.base import Transformation, Block, Constraint
from pyomo.gdp import Disjunct

from six import itervalues

logger = logging.getLogger('pyomo.gdp')


class HACK_GDP_Var_Mover(Transformation):
    """Move indicator vars to top block.

    HACK: this will move all indicator variables on the model to the top block
    so the writers can find them.

    """

    alias('gdp.varmover', doc=textwrap.fill(textwrap.dedent(__doc__.strip())))

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


class HACK_GDP_Disjunct_Reclassifier(Transformation):
    """Reclassify Disjuncts to Blocks.

    HACK: this will reclassify all Disjuncts to Blocks so the current writers
    can find the variables

    """

    alias('gdp.reclassify',
          doc=textwrap.fill(textwrap.dedent(__doc__.strip())))

    def _apply_to(self, instance, **kwds):
        assert not kwds
        disjunct_generator = instance.component_objects(
            Disjunct, descend_into=(Block, Disjunct))
        for disjunct_component in disjunct_generator:
            for disjunct in itervalues(disjunct_component._data):
                if disjunct.active:
                    logger.error("""
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
