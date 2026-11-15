# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

# -*- coding: utf-8 -*-
"""Convex MINLP test model based on Example 1 from a regularization OA study.

The expected optimal solution value is -56.981.

References
----------
Kronqvist, J., Bernal, D. E., and Grossmann, I. E. (2020). Using
regularization and second order information in outer approximation for convex
MINLP. Mathematical Programming, 180(1), 285-310.

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


class Minlp4Simple(ConcreteModel):
    """Convex MINLP benchmark instance used in MindtPy regression tests."""

    def __init__(self, *args, **kwargs):
        """Create the problem.

        Parameters
        ----------
        *args
            Positional arguments forwarded to ``ConcreteModel``.
        **kwargs
            Keyword arguments forwarded to ``ConcreteModel``.
        """
        kwargs.setdefault('name', 'Minlp4Simple')
        super(Minlp4Simple, self).__init__(*args, **kwargs)
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
