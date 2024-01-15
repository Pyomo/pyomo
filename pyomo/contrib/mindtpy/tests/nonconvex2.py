# -*- coding: utf-8 -*-
"""Problem B in paper 'Outer approximation algorithms for separable nonconvex mixed-integer nonlinear programs'

Ref:
Kesavan P, Allgor R J, Gatzke E P, et al. Outer approximation algorithms for separable nonconvex mixed-integer nonlinear programs[J]. Mathematical Programming, 2004, 100(3): 517-535.

Problem type:   nonconvex MINLP
        size:   8  binary variable
                3  continuous variables
                7  constraints

"""
from pyomo.environ import (
    ConcreteModel,
    Constraint,
    Reals,
    Binary,
    Objective,
    Var,
    minimize,
    log,
)
from pyomo.common.collections import ComponentMap


class Nonconvex2(ConcreteModel):
    def __init__(self, *args, **kwargs):
        """Create the problem."""
        kwargs.setdefault('name', 'Nonconvex2')
        super(Nonconvex2, self).__init__(*args, **kwargs)
        m = self

        m.x1 = Var(within=Reals, bounds=(0, 0.9970))
        m.x2 = Var(within=Reals, bounds=(0, 0.9985))
        m.x3 = Var(within=Reals, bounds=(0, 0.9988))
        m.y1 = Var(within=Binary, bounds=(0, 1), initialize=0)
        m.y2 = Var(within=Binary, bounds=(0, 1), initialize=0)
        m.y3 = Var(within=Binary, bounds=(0, 1), initialize=0)
        m.y4 = Var(within=Binary, bounds=(0, 1), initialize=0)
        m.y5 = Var(within=Binary, bounds=(0, 1), initialize=0)
        m.y6 = Var(within=Binary, bounds=(0, 1), initialize=0)
        m.y7 = Var(within=Binary, bounds=(0, 1), initialize=0)
        m.y8 = Var(within=Binary, bounds=(0, 1), initialize=0)

        m.objective = Objective(expr=-m.x1 * m.x2 * m.x3, sense=minimize)

        m.c1 = Constraint(
            expr=-log(1 - m.x1) + log(0.1) * m.y1 + log(0.2) * m.y2 + log(0.15) * m.y3
            == 0
        )
        m.c2 = Constraint(
            expr=-log(1 - m.x2) + log(0.05) * m.y4 + log(0.2) * m.y5 + log(0.15) * m.y6
            == 0
        )
        m.c3 = Constraint(
            expr=-log(1 - m.x3) + log(0.02) * m.y7 + log(0.06) * m.y8 == 0
        )

        m.c4 = Constraint(expr=-m.y1 - m.y2 - m.y3 <= -1)
        m.c5 = Constraint(expr=-m.y4 - m.y5 - m.y6 <= -1)
        m.c6 = Constraint(expr=-m.y7 - m.y8 <= -1)
        m.c7 = Constraint(
            expr=3 * m.y1
            + m.y2
            + 2 * m.y3
            + 3 * m.y4
            + 2 * m.y5
            + m.y6
            + 3 * m.y7
            + 2 * m.y8
            <= 10
        )
        m.optimal_value = -0.94347
        m.optimal_solution = ComponentMap()
        m.optimal_solution[m.x1] = 0.97
        m.optimal_solution[m.x2] = 0.9925
        m.optimal_solution[m.x3] = 0.98
        m.optimal_solution[m.y1] = 0.0
        m.optimal_solution[m.y2] = 1.0
        m.optimal_solution[m.y3] = 1.0
        m.optimal_solution[m.y4] = 1.0
        m.optimal_solution[m.y5] = 0.0
        m.optimal_solution[m.y6] = 1.0
        m.optimal_solution[m.y7] = 1.0
        m.optimal_solution[m.y8] = 0.0
