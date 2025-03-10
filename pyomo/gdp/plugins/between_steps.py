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

"""
Between Steps (P-Split) reformulation for GDPs from:

J. Kronqvist, R. Misener, and C. Tsay, "Between Steps: Intermediate
Relaxations between big-M and Convex Hull Reformulations," 2021.
"""
from pyomo.core import Transformation, TransformationFactory


@TransformationFactory.register(
    'gdp.between_steps',
    doc="Reformulates a convex disjunctive model "
    "by splitting additively separable constraints"
    "on P sets of variables, then taking hull "
    "reformulation.",
)
class BetweenSteps_Transformation(Transformation):
    """
    Transform disjunctive model to equivalent MI(N)LP using the between steps
    transformation from Konqvist et al. 2021 [1].

    This transformation first calls the 'gdp.partition_disjuncts'
    transformation, resulting in an equivalent GDP with the constraints
    partitioned, and then takes the hull reformulation of that model to get
    an algebraic model.

    References
    ----------
        [1] J. Kronqvist, R. Misener, and C. Tsay, "Between Steps: Intermediate
            Relaxations between big-M and Convex Hull Reformulations," 2021.
    """

    def __init__(self):
        super(BetweenSteps_Transformation, self).__init__()

    def _apply_to(self, instance, **kwds):
        TransformationFactory('gdp.partition_disjuncts').apply_to(instance, **kwds)
        TransformationFactory('gdp.hull').apply_to(instance)
