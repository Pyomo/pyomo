# -*- coding: utf-8 -*-
"""Example 2 in paper 'A Feasibility Pump for mixed integer nonlinear programs'

Ref:
    Bonami P, Cornuéjols G, Lodi A, et al. A feasibility pump for mixed integer nonlinear programs[J]. Mathematical Programming, 2009, 119(2): 331-352.

    Problem type:    convex MINLP
            size:    1  binary variables
                     2  continuous variables
                     3  constraints

"""
from __future__ import division
from math import pi

from pyomo.environ import (Binary, ConcreteModel, Constraint,
                           NonNegativeReals, Objective, Param,
                           RangeSet, Var, exp, minimize, Reals, sin, cos)


class Feasibility_Pump2(ConcreteModel):
    """Feasibility_Pump2 example"""

    def __init__(self, *args, **kwargs):
        """Create the problem."""
        kwargs.setdefault('name', 'Feasibility_Pump2')
        super(Feasibility_Pump2, self).__init__(*args, **kwargs)
        m = self

        m.x = Var(within=Binary)
        m.y = Var(within=Reals)

        m.objective = Objective(expr=- m.y, sense=minimize)

        m.c1 = Constraint(expr=m.y - sin(m.x * pi * (5 / 3)) <= 0)
        m.c2 = Constraint(expr=- m.y - sin(m.x * pi * (5 / 3)) <= 0)
        m.optimal_value = 0
