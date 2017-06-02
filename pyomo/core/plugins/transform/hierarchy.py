#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core.base import Transformation


class AbstractTransformation(Transformation):
    """
    Base class for all model transformations that produce abstract
    models.
    """

    def __init__(self, **kwds):
        kwds["name"] = kwds.get("name", "abstract_transformation")
        super(AbstractTransformation, self).__init__(**kwds)


class ConcreteTransformation(Transformation):
    """
    Base class for all model transformations that produce concrete
    models.
    """

    def __init__(self, **kwds):
        kwds["name"] = kwds.get("name", "concrete_transformation")
        super(ConcreteTransformation, self).__init__(**kwds)


class IsomorphicTransformation(Transformation):
    """
    Base class for 'lossless' transformations for which a bijective
    mapping between optimal variable values and the optimal cost
    exists.
    """

    def __init__(self, **kwds):
        kwds["name"] = kwds.get("name", "isomorphic_transformation")
        super(IsomorphicTransformation, self).__init__(**kwds)


class LinearTransformation(Transformation):
    """ Base class for all linear model transformations. """

    def __init__(self, **kwds):
        kwds["name"] = kwds.get("name", "linear_transform")
        super(LinearTransformation, self).__init__(**kwds)


class NonIsomorphicTransformation(Transformation):
    """
    Base class for 'lossy' transformations for which a bijective
    mapping between optimal variable values and the optimal cost does
    not  exist.
    """

    def __init__(self, **kwds):
        kwds["name"] = kwds.get("name", "isomorphic_transformation")
        super(NonIsomorphicTransformation, self).__init__(**kwds)


class NonlinearTransformation(Transformation):
    """ Base class for all nonlinear model transformations. """

    def __init__(self, **kwds):
        kwds["name"] = kwds.get("name", "nonlinear_transform")
        super(NonlinearTransformation, self).__init__(**kwds)

