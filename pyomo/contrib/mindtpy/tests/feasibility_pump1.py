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

# -*- coding: utf-8 -*-
"""Example 1 in paper 'A Feasibility Pump for mixed integer nonlinear programs'

Ref:
    Bonami P, Cornuéjols G, Lodi A, et al. A feasibility pump for mixed integer nonlinear programs[J]. Mathematical Programming, 2009, 119(2): 331-352.

    Problem type:    convex MINLP
            size:    1  binary variables
                     2  continuous variables
                     3  constraints

"""

from pyomo.environ import (
    Binary,
    ConcreteModel,
    Constraint,
    Objective,
    Var,
    minimize,
    Reals,
)
from pyomo.common.collections import ComponentMap


class FeasPump1(ConcreteModel):
    """Feasibility Pump example 1"""

    def __init__(self, *args, **kwargs):
        """Create the problem."""
        kwargs.setdefault('name', 'Feasibility Pump 1')
        super(FeasPump1, self).__init__(*args, **kwargs)
        m = self

        m.x = Var(within=Binary)
        m.y1 = Var(within=Reals)
        m.y2 = Var(within=Reals)

        m.objective = Objective(expr=m.x, sense=minimize)

        m.c1 = Constraint(
            expr=(m.y1 - 0.5) * (m.y1 - 0.5) + (m.y2 - 0.5) * (m.y2 - 0.5) <= 0.25
        )
        m.c2 = Constraint(expr=m.x - m.y1 <= 3)
        m.c3 = Constraint(expr=m.y2 <= 0)
        m.optimal_value = 0
        m.optimal_solution = ComponentMap()
        m.optimal_solution[m.x] = 0.0
        m.optimal_solution[m.y1] = 0.5
        m.optimal_solution[m.y2] = 0.0
