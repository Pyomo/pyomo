#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.kernel


# @Nonnegative
class NonNegativeVariable(pyomo.kernel.variable):
    """A non-negative variable."""

    __slots__ = ()

    def __init__(self, **kwds):
        if 'lb' not in kwds:
            kwds['lb'] = 0
        if kwds['lb'] < 0:
            raise ValueError("lower bound must be non-negative")
        super(NonNegativeVariable, self).__init__(**kwds)

    #
    # restrict assignments to x.lb to non-negative numbers
    #
    @property
    def lb(self):
        # calls the base class property getter
        return pyomo.kernel.variable.lb.fget(self)

    @lb.setter
    def lb(self, lb):
        if lb < 0:
            raise ValueError("lower bound must be non-negative")
        # calls the base class property setter
        pyomo.kernel.variable.lb.fset(self, lb)


# @Nonnegative


# @Point
class Point(pyomo.kernel.variable_tuple):
    """A 3-dimensional point in Cartesian space with the
    z coordinate restricted to non-negative values."""

    __slots__ = ()

    def __init__(self):
        super(Point, self).__init__(
            (pyomo.kernel.variable(), pyomo.kernel.variable(), NonNegativeVariable())
        )

    @property
    def x(self):
        return self[0]

    @property
    def y(self):
        return self[1]

    @property
    def z(self):
        return self[2]


# @Point


# @SOC
class SOC(pyomo.kernel.constraint):
    """A convex second-order cone constraint"""

    __slots__ = ()

    def __init__(self, point):
        assert isinstance(point.z, NonNegativeVariable)
        super(SOC, self).__init__(point.x**2 + point.y**2 <= point.z**2)


# @SOC

# @Usage
model = pyomo.kernel.block()
model.p = Point()
model.p.z.lb = 0
model.soc = SOC(model.p)
# @Usage
