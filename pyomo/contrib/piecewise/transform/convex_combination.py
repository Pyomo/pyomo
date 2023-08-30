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
import pyomo.gdp.plugins.multiple_bigm


@TransformationFactory.register(
    'contrib.piecewise.convex_combination',
    doc="Convert piecewise-linear model to a GDP "
    "to 'Convex Combination' MIP formulation.",
)
class ConvexCombinationTransformation(Transformation):
    """
    Converts a model containing PiecewiseLinearFunctions to a an equivalent
    MIP via the Convex Combination method from [1]. Note that,
    while this model probably resolves to the model described in [1] after
    presolve, the Pyomo version is not as simplified.

    References
    ----------
    [1] J.P. Vielma, S. Ahmed, and G. Nemhauser, "Mixed-integer models
        for nonseparable piecewise-linear optimization: unifying framework
        and extensions," Operations Research, vol. 58, no. 2, pp. 305-315,
        2010.
    """

    def _apply_to(self, instance, **kwds):
        TransformationFactory('contrib.piecewise.reduced_inner_repn_gdp').apply_to(
            instance
        )
        TransformationFactory('gdp.mbigm').apply_to(
            instance, reduce_bound_constraints=True
        )
