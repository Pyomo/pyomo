# -*- coding: utf-8 -*-
"""Problem C in paper 'Outer approximation algorithms for separable nonconvex mixed-integer nonlinear programs'.
The problem in the paper has two optimal solution. Variable y4 and y6 are symmetric. Therefore, we remove variable y6 for simplification.

Ref:
Kesavan P, Allgor R J, Gatzke E P, et al. Outer approximation algorithms for separable nonconvex mixed-integer nonlinear programs[J]. Mathematical Programming, 2004, 100(3): 517-535.

Problem type:   nonconvex MINLP
        size:   6  binary variable
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


class Nonconvex3(ConcreteModel):
    def __init__(self, *args, **kwargs):
        """Create the problem."""
        kwargs.setdefault('name', 'Nonconvex3')
        super(Nonconvex3, self).__init__(*args, **kwargs)
        m = self

        m.x1 = Var(within=Reals, bounds=(1, 5))
        m.x2 = Var(within=Reals, bounds=(1, 5))
        m.y1 = Var(within=Binary, bounds=(0, 1), initialize=0)
        m.y2 = Var(within=Binary, bounds=(0, 1), initialize=0)
        m.y3 = Var(within=Binary, bounds=(0, 1), initialize=0)
        m.y4 = Var(within=Binary, bounds=(0, 1), initialize=0)
        m.y5 = Var(within=Binary, bounds=(0, 1), initialize=0)

        m.objective = Objective(expr=7 * m.x1 + 10 * m.x2, sense=minimize)

        m.c1 = Constraint(
            expr=(m.x1**1.2) * (m.x2**1.7) - 7 * m.x1 - 9 * m.x2 <= -24
        )
        m.c2 = Constraint(expr=-m.x1 - 2 * m.x2 <= 5)
        m.c3 = Constraint(expr=-3 * m.x1 + m.x2 <= 1)
        m.c4 = Constraint(expr=4 * m.x1 - 3 * m.x2 <= 11)
        m.c5 = Constraint(expr=-m.x1 + m.y1 + 2 * m.y2 + 4 * m.y3 == 0)
        m.c6 = Constraint(expr=-m.x2 + m.y4 + 2 * m.y5 == 0)
        m.optimal_value = 31
        m.optimal_solution = ComponentMap()
        m.optimal_solution[m.x1] = 3.0
        m.optimal_solution[m.x2] = 1.0
        m.optimal_solution[m.y1] = 1.0
        m.optimal_solution[m.y2] = 1.0
        m.optimal_solution[m.y3] = 0.0
        m.optimal_solution[m.y4] = 1.0
        m.optimal_solution[m.y5] = 0.0
