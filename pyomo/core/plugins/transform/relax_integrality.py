#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common import deprecated
from pyomo.core.base import TransformationFactory
from pyomo.core.plugins.transform.discrete_vars import RelaxIntegerVars


@TransformationFactory.register(
    'core.relax_integrality',
    doc="[DEPRECATED] Create a model where integer variables are replaced with "
    "real variables.")
class RelaxIntegrality(RelaxIntegerVars):
    """
    This plugin relaxes integrality in a Pyomo model.
    """

    @deprecated(
        "core.relax_integrality is deprecated.  Use core.relax_integer_vars",
        version='5.7')
    def __init__(self, **kwds):
        super(RelaxIntegrality, self).__init__(**kwds)
