# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

# -*- coding: utf-8 -*-
"""Example in paper 'Using regularization and second order information in outer approximation for convex MINLP'

References
----------
Kronqvist, J., Bernal, D. E., and Grossmann, I. E. (2020). Using
regularization and second order information in outer approximation for convex
MINLP. Mathematical Programming, 180(1), 285-310.

Problem type:   convex MINLP
        size:   1  binary variable
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
    minimize,
    exp,
)
from pyomo.common.collections import ComponentMap


class Minlp5Simple(ConcreteModel):
    """Convex MINLP test instance used by regularization and utility tests."""

    def __init__(self, *args, **kwargs):
        """Create the problem.

        Parameters
        ----------
        *args
            Positional arguments forwarded to ``ConcreteModel``.
        **kwargs
            Keyword arguments forwarded to ``ConcreteModel``.
        """
        kwargs.setdefault('name', 'Minlp5Simple')
        super(Minlp5Simple, self).__init__(*args, **kwargs)
        m = self

        m.x = Var(within=Reals, bounds=(1, 20), initialize=5.29)
        m.y = Var(within=Integers, bounds=(1, 20), initialize=3)

        m.objective = Objective(
            expr=0.3 * (m.x - 8) ** 2
            + 0.04 * (m.y - 6) ** 4
            + 0.1 * exp(2 * m.x) * (m.y ** (-4)),
            sense=minimize,
        )

        m.c1 = Constraint(expr=6 * m.x + m.y <= 60)
        m.c2 = Constraint(expr=1 / m.x + 1 / m.x - m.x**0.5 * m.y**0.5 <= -1)
        m.c3 = Constraint(expr=2 * m.x - 5 * m.y <= -1)
        m.optimal_value = 3.6572
        m.optimal_solution = ComponentMap()
        m.optimal_solution[m.x] = 4.991797840270567
        m.optimal_solution[m.y] = 7.0
