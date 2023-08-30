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


@TransformationFactory.register(
    'contrib.piecewise.multiple_choice',
    doc="Convert piecewise-linear model to a GDP "
    "to 'Multiple Choice' MIP "
    "formulation.",
)
class MultipleChoiceTransformation(Transformation):
    """
    Converts a model containing PiecewiseLinearFunctions to a an equivalent
    MIP via the Multiple Choice method from [1]. Note that,
    while this model probably resolves to the model described in [1] after
    presolve, the Pyomo version is not as simplified. Specifically, in [1], the
    the 'z' variables (representing the value of the piecewise-linear function
    in each Disjunct) are not disaggregated. In this transformation's output
    they will be, but a linear combination of inequalities yields a model
    equivalent to the Multiple Choice model in [1].

    References
    ----------
    [1] J.P. Vielma, S. Ahmed, and G. Nemhauser, "Mixed-integer models
        for nonseparable piecewise-linear optimization: unifying framework
        and extensions," Operations Research, vol. 58, no. 2, pp. 305-315,
        2010.
    """

    def _apply_to(self, instance, **kwds):
        TransformationFactory('contrib.piecewise.outer_repn_gdp').apply_to(instance)
        TransformationFactory('gdp.hull').apply_to(instance)
