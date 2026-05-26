# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

# -*- coding: utf-8 -*-
"""Example 2 in paper 'A Feasibility Pump for mixed integer nonlinear programs'

Ref:
    Bonami P, Cornuéjols G, Lodi A, et al. A feasibility pump for mixed integer nonlinear programs[J]. Mathematical Programming, 2009, 119(2): 331-352.

    Problem type:    convex MINLP
            size:    1  binary variables
                     2  continuous variables
                     3  constraints

"""

from math import pi

from pyomo.environ import (
    Binary,
    ConcreteModel,
    Constraint,
    Objective,
    Var,
    minimize,
    Reals,
    sin,
)
from pyomo.common.collections import ComponentMap


class FeasibilityPump2(ConcreteModel):
    """Feasibility Pump example 2"""

    def __init__(self, *args, **kwargs):
        """Create the problem.

        Parameters
        ----------
        *args
            Positional arguments forwarded to ``ConcreteModel``.
        **kwargs
            Keyword arguments forwarded to ``ConcreteModel``.
        """
        kwargs.setdefault('name', 'Feasibility Pump 2')
        super(FeasibilityPump2, self).__init__(*args, **kwargs)
        m = self

        m.x = Var(within=Binary)
        m.y = Var(within=Reals)

        m.objective = Objective(expr=-m.y, sense=minimize)

        m.c1 = Constraint(expr=m.y - sin(m.x * pi * (5 / 3)) <= 0)
        m.c2 = Constraint(expr=-m.y - sin(m.x * pi * (5 / 3)) <= 0)
        m.optimal_value = 0
        m.optimal_solution = ComponentMap()
        m.optimal_solution[m.x] = 0.0
        m.optimal_solution[m.y] = 0.0


# Backward-compatible alias.
FeasPump2 = FeasibilityPump2
