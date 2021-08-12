# -*- coding: utf-8 -*-
"""Example 1 in paper 'A Feasibility Pump for mixed integer nonlinear programs'

Ref:
    Bonami P, Cornuéjols G, Lodi A, et al. A feasibility pump for mixed integer nonlinear programs[J]. Mathematical Programming, 2009, 119(2): 331-352.

    Problem type:    convex MINLP
            size:    1  binary variables
                     2  continuous variables
                     3  constraints

"""
from __future__ import division

from pyomo.environ import (Binary, ConcreteModel, Constraint,
                           NonNegativeReals, Objective, Param,
                           RangeSet, Var, exp, minimize, Reals)


class Feasibility_Pump1(ConcreteModel):
    """Feasibility_Pump1 example"""

    def __init__(self, *args, **kwargs):
        """Create the problem."""
        kwargs.setdefault('name', 'Feasibility_Pump1')
        super(Feasibility_Pump1, self).__init__(*args, **kwargs)
        m = self

        m.x = Var(within=Binary)
        m.y1 = Var(within=Reals)
        m.y2 = Var(within=Reals)

        m.objective = Objective(expr=m.x, sense=minimize)

        m.c1 = Constraint(expr=(m.y1-0.5) * (m.y1-0.5) +
                          (m.y2-0.5) * (m.y2-0.5) <= 0.25)
        m.c2 = Constraint(expr=m.x - m.y1 <= 3)
        m.c3 = Constraint(expr=m.y2 <= 0)
        m.optimal_value = 0
