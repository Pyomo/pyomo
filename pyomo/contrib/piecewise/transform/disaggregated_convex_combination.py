#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core.base import Transformation, TransformationFactory
import pyomo.gdp.plugins.hull

@TransformationFactory.register('contrib.disaggregated_convex_combination',
                                doc="Convert piecewise-linear model to a GDP "
                                "to 'Disaggregated Convex Combination' MIP "
                                "formulation.")
class DisaggregatedConvexCombinationTransformation(Transformation):
    def _apply_to(self, instance, **kwds):
        TransformationFactory('contrib.inner_repn_gdp').apply_to(instance)
        TransformationFactory('gdp.hull').apply_to(instance)
