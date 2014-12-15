#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  _________________________________________________________________________

from pyomo.util.plugin import alias
from pyomo.core.base import Var
from pyomo.core.base.set_types import BooleanSet, IntegerSet, Reals, RealInterval
import pyomo.core.base
from pyomo.core.plugins.transform.hierarchy import NonIsomorphicTransformation


class RelaxIntegrality(NonIsomorphicTransformation):
    """
    This plugin relaxes integrality in a Pyomo model.
    """

    alias('base.relax_integrality',\
          doc="Create a model where integer variables are replaced with real variables.")

    def __init__(self, **kwds):
        kwds['name'] = "relax_integrality"
        super(RelaxIntegrality, self).__init__(**kwds)

    def apply(self, model, **kwds):
        #
        # Clone the model
        #
        M = model.clone()
        #
        # Iterate over all variables, replacing the domain with a real-valued domain
        # and setting appropriate bounds.
        #
        # Note: this technique is not extensible.  If a user creates a new integer variable
        # class then we'd need to add it here (unless it was a subset of one of the sets we
        # detect here.  Alternatively, we could make the variables responsible for managing
        # the generation of a relaxed variable.  But that would pollute the variable objects
        # with logic about generating a specific relaxation.  That's probably a worse alternative.
        #
        # TODO: rework this so it works with model instances
        #
        comp = M.components(Var)
        for var in comp.values():
            if isinstance(var.domain, BooleanSet):
                var.domain=Reals
                dbnd = ( 0.0, 1.0 )
            elif isinstance(var.domain, IntegerSet):
                dbnd = var.domain.bounds()
                if dbnd is None:
                    bnd = ( None, None )
                var.domain=Reals
            elif isinstance(var.domain, pyomo.core.base.RangeSet):
                dbnd = var.domain.bounds()
                if dbnd is None:
                    bnd = ( None, None )
                var.domain=Reals
            else:
                continue

            bnd = var.domain.bounds()
            if bnd is None:
                bnd = ( None, None )
            bnd = ( self._tightenBound(bnd[0], dbnd[0], max),
                    self._tightenBound(bnd[1], dbnd[1], min) )

            if bnd == (None, None):
                var.domain = Reals
            else:
                var.domain = RealInterval(bounds=bnd)
            #print("HERE %s %s %s %s %s" % (str(var), str(type(var.domain)), str(var.bounds), str(bnd), str(dbnd)))
            var._initialize_members(var._index)
            #print("HERE %s %s %s %s %s" % (str(var), str(type(var.domain)), str(var.bounds), str(bnd), str(dbnd)))

        return M

    def _tightenBound(self, a, b, comp):
        if a is None:
            return b
        if b is None:
            return a
        return comp(a,b)
