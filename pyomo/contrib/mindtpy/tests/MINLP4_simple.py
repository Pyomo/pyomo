# -*- coding: utf-8 -*-
""" Example 1 in Paper 'Using regularization and second order information in outer approximation for convex MINLP'

The expected optimal solution value is -56.981.

Ref:
    Kronqvist J, Bernal D E, Grossmann I E. Using regularization and second order information in outer approximation for convex MINLP[J]. Mathematical Programming, 2020, 180(1): 285-310.

    Problem type:    convex MINLP
            size:    1  binary variable
                     1  continuous variables
                     3  constraints


"""

from pyomo.environ import (
    ConcreteModel,
    Constraint,
    Reals,
    Integers,
    Objective,
    Var,
    sqrt,
    minimize,
    exp,
)
from pyomo.common.collections import ComponentMap


class SimpleMINLP4(ConcreteModel):
    def __init__(self, *args, **kwargs):
        """Create the problem."""
        kwargs.setdefault('name', 'SimpleMINLP4')
        super(SimpleMINLP4, self).__init__(*args, **kwargs)
        m = self

        m.x = Var(domain=Reals, bounds=(1, 20), initialize=5.29)
        m.y = Var(domain=Integers, bounds=(1, 20), initialize=3)

        m.c1 = Constraint(
            expr=0.3 * (m.x - 8) ** 2
            + 0.04 * (m.y - 6) ** 4
            + 0.1 * exp(2 * m.x) * ((m.y) ** (-4))
            <= 56
        )
        m.c2 = Constraint(expr=1 / m.x + 1 / m.y - sqrt(m.x) * sqrt(m.y) <= -1)
        m.c3 = Constraint(expr=2 * m.x - 5 * m.y <= -1)

        m.objective = Objective(expr=-6 * m.x - m.y, sense=minimize)
        m.optimal_value = -56.981
        m.optimal_solution = ComponentMap()
        m.optimal_solution[m.x] = 7.663528589138092
        m.optimal_solution[m.y] = 11.0
