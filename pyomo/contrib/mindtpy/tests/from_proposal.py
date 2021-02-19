"""
See David Bernal PhD proposal example.
"""

from __future__ import division
from pyomo.environ import (ConcreteModel, Constraint, Reals, Integers,
                           Objective, Var, sqrt, minimize)


class ProposalModel(ConcreteModel):
    def __init__(self, *args, **kwargs):
        """Create the problem."""
        kwargs.setdefault('name', 'DavidProposalExample')
        super(ProposalModel, self).__init__(*args, **kwargs)
        m = self

        m.x = Var(domain=Reals, bounds=(0, 20), initialize=4)
        m.y = Var(domain=Integers, bounds=(0, 20), initialize=2)

        m.c1 = Constraint(expr=m.x**2/20.0 + m.y <= 20)
        m.c2 = Constraint(expr=(m.x-1)**2/40.0 - m.y <= -4)
        m.c3 = Constraint(expr=m.y - 10*sqrt(m.x+0.1) <= 0)
        m.c4 = Constraint(expr=-m.x-m.y <= -5)

        m.obj = Objective(expr=m.x - m.y / 4.5 + 2, sense=minimize)
