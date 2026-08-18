# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

# -*- coding: utf-8 -*-
"""Nonconvex MINLP test model based on problem D from a benchmark study.

The expected optimal solution value is -17.

References
----------
Kesavan, P., Allgor, R. J., Gatzke, E. P., et al. (2004). Outer approximation
algorithms for separable nonconvex mixed-integer nonlinear programs.
Mathematical Programming, 100(3), 517-535.

    Problem type:    nonconvex MINLP
        size:    3  binary variables
             2  continuous variables
             4  constraints

"""

from pyomo.environ import *
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


class Nonconvex4(ConcreteModel):
    """Nonconvex MINLP benchmark problem D for MindtPy tests."""

    def __init__(self, *args, **kwargs):
        """Create the problem.

        Parameters
        ----------
        *args
            Positional arguments forwarded to ``ConcreteModel``.
        **kwargs
            Keyword arguments forwarded to ``ConcreteModel``.
        """
        kwargs.setdefault('name', 'Nonconvex4')
        super(Nonconvex4, self).__init__(*args, **kwargs)
        m = self

        m.x1 = Var(within=Reals, bounds=(1, 10))
        m.x2 = Var(within=Reals, bounds=(1, 6))
        m.y1 = Var(within=Binary, bounds=(0, 1), initialize=0)
        m.y2 = Var(within=Binary, bounds=(0, 1), initialize=0)
        m.y3 = Var(within=Binary, bounds=(0, 1), initialize=0)

        m.objective = Objective(expr=-5 * m.x1 + 3 * m.x2, sense=minimize)

        m.c1 = Constraint(
            expr=2 * (m.x2**2)
            - 2 * (m.x2**0.5)
            - 2 * (m.x1**0.5) * (m.x2**2)
            + 11 * m.x2
            + 8 * m.x1
            <= 39
        )
        m.c2 = Constraint(expr=m.x1 - m.x2 <= 3)
        m.c3 = Constraint(expr=3 * m.x1 + 2 * m.x2 <= 24)
        m.c4 = Constraint(expr=-m.x1 + m.y1 + 2 * m.y2 + 4 * m.y3 == 0)
        m.optimal_value = -17
        m.optimal_solution = ComponentMap()
        m.optimal_solution[m.x1] = 4.0
        m.optimal_solution[m.x2] = 1.0
        m.optimal_solution[m.y1] = 0.0
        m.optimal_solution[m.y2] = 0.0
        m.optimal_solution[m.y3] = 1.0
