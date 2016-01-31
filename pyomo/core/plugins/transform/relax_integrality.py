#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
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

    alias('core.relax_integrality',\
          doc="Create a model where integer variables are replaced with real variables.")

    def __init__(self, **kwds):
        kwds['name'] = "relax_integrality"
        super(RelaxIntegrality, self).__init__(**kwds)

    def _apply_to(self, model, **kwds):
        #
        # Iterate over all variables, replacing the domain with a real-valued domain
        # and setting appropriate bounds.
        #
        for var in model.component_data_objects(Var):
            # var.bounds returns the tightest of the domain
            # vs user-supplied lower and upper bounds
            lb, ub = var.bounds
            var.domain = Reals
            var.setlb(lb)
            var.setub(ub)
