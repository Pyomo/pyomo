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

import enum

from pyomo.common.deprecation import RenamedClass

class ObjectiveSense(enum.IntEnum):
    """Flag indicating if an objective is minimizing (1) or maximizing (-1).

    While the numeric values are arbitrary, there are parts of Pyomo
    that rely on this particular choice of value.  These values are also
    consistent with some solvers (notably Gurobi).

    """
    minimize = 1
    maximize = -1

    # Overloading __str__ is needed to match the behavior of the old
    # pyutilib.enum class (removed June 2020). There are spots in the
    # code base that expect the string representation for items in the
    # enum to not include the class name. New uses of enum shouldn't
    # need to do this.
    def __str__(self):
        return self.name

minimize = ObjectiveSense.minimize
maximize = ObjectiveSense.maximize


