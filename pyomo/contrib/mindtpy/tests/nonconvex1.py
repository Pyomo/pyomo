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
"""Problem A in paper 'Outer approximation algorithms for separable nonconvex mixed-integer nonlinear programs'

Ref:
Kesavan P, Allgor R J, Gatzke E P, et al. Outer approximation algorithms for separable nonconvex mixed-integer nonlinear programs[J]. Mathematical Programming, 2004, 100(3): 517-535.

Problem type:   nonconvex MINLP
        size:   3  binary variable
                2  continuous variables
                6  constraints

"""
from pyomo.environ import (
    ConcreteModel,
    Constraint,
    Reals,
    Binary,
    Objective,
    Var,
    minimize,
)
from pyomo.common.collections import ComponentMap


class Nonconvex1(ConcreteModel):
    def __init__(self, *args, **kwargs):
        """Create the problem."""
        kwargs.setdefault('name', 'Nonconvex1')
        super(Nonconvex1, self).__init__(*args, **kwargs)
        m = self

        m.x1 = Var(within=Reals, bounds=(0, 10))
        m.x2 = Var(within=Reals, bounds=(0, 10))
        m.y1 = Var(within=Binary, bounds=(0, 1), initialize=0)
        m.y2 = Var(within=Binary, bounds=(0, 1), initialize=0)
        m.y3 = Var(within=Binary, bounds=(0, 1), initialize=0)

        m.objective = Objective(
            expr=2 * m.x1 + 3 * m.x2 + 1.5 * m.y1 + 2 * m.y2 - 0.5 * m.y3,
            sense=minimize,
        )

        m.c1 = Constraint(expr=m.x1 * m.x1 + m.y1 == 1.25)
        m.c2 = Constraint(expr=m.x2**1.5 + 1.5 * m.y2 == 3)
        m.c4 = Constraint(expr=m.x1 + m.y1 <= 1.6)
        m.c5 = Constraint(expr=1.333 * m.x2 + m.y2 <= 3)
        m.c6 = Constraint(expr=-m.y1 - m.y2 + m.y3 <= 0)
        m.optimal_value = 7.667
        m.optimal_solution = ComponentMap()
        m.optimal_solution[m.x1] = 1.118033988749895
        m.optimal_solution[m.x2] = 1.3103706971044473
        m.optimal_solution[m.y1] = 0.0
        m.optimal_solution[m.y2] = 1.0
        m.optimal_solution[m.y3] = 1.0
