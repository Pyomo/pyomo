"""Problem A in paper "Outer approximation algorithms for separable nonconvex mixed-integer nonlinear programs"

Ref:
Kesavan P, Allgor R J, Gatzke E P, et al. Outer approximation algorithms for separable nonconvex mixed-integer nonlinear programs[J]. Mathematical Programming, 2004, 100(3): 517-535.

Problem type:   nonconvex MINLP
        size:   3  binary variable
                2  continuous variables
                6  constraints

"""
from pyomo.environ import *


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

        m.objective = Objective(expr=2 * m.x1 + 3 * m.x2 + 1.5 * m.y1 +
                                2 * m.y2 - 0.5 * m.y3, sense=minimize)

        m.c1 = Constraint(expr=m.x1 * m.x1 + m.y1 == 1.25)
        m.c2 = Constraint(expr=m.x2 ** 1.5 + 1.5 * m.y2 == 3)
        m.c4 = Constraint(expr=m.x1 + m.y1 <= 1.6)
        m.c5 = Constraint(expr=1.333 * m.x2 + m.y2 <= 3)
        m.c6 = Constraint(expr=-m.y1 - m.y2 + m.y3 <= 0)
