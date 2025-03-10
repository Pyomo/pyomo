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

"""Example in the online doc.

The expected optimal solution value is 2.438447187191098.

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


class OnlineDocExample(ConcreteModel):
    def __init__(self, *args, **kwargs):
        """Create the problem."""
        kwargs.setdefault('name', 'OnlineDocExample')
        super(OnlineDocExample, self).__init__(*args, **kwargs)
        m = self
        m.x = Var(bounds=(1.0, 10.0), initialize=5.0)
        m.y = Var(within=Binary)
        m.c1 = Constraint(expr=(m.x - 4.0) ** 2 - m.x <= 50.0 * (1 - m.y))
        m.c2 = Constraint(expr=m.x * log(m.x) + 5 <= 50.0 * (m.y))
        m.objective = Objective(expr=m.x, sense=minimize)
        m.optimal_value = 2.438447
        m.optimal_solution = ComponentMap()
        m.optimal_solution[m.x] = 2.4384471855377243
        m.optimal_solution[m.y] = 1.0
