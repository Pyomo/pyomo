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
"""Example of constraint qualification.

The expected optimal solution value is 3.

    Problem type:    convex MINLP
            size:    1  binary variable
                     1  continuous variables
                     2  constraints

"""

from pyomo.environ import (
    Binary,
    ConcreteModel,
    Constraint,
    Objective,
    Var,
    minimize,
    log,
)
from pyomo.common.collections import ComponentMap


class ConstraintQualificationExample(ConcreteModel):
    def __init__(self, *args, **kwargs):
        """Create the problem."""
        kwargs.setdefault('name', 'ConstraintQualificationExample')
        super(ConstraintQualificationExample, self).__init__(*args, **kwargs)
        m = self
        m.x = Var(bounds=(1.0, 10.0), initialize=5.0)
        m.y = Var(within=Binary, initialize=1.0)
        m.c1 = Constraint(expr=(m.x - 3.0) ** 2 <= 50.0 * (1 - m.y))
        m.c2 = Constraint(expr=m.x * log(m.x) + 5.0 <= 50.0 * (m.y))
        m.objective = Objective(expr=m.x, sense=minimize)
        m.optimal_value = 3
        m.optimal_solution = ComponentMap()
        m.optimal_solution[m.x] = 3.0
        m.optimal_solution[m.y] = 1.0
