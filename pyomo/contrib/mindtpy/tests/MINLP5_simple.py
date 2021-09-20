# -*- coding: utf-8 -*-
"""Example in paper 'Using regularization and second order information in outer approximation for convex MINLP'

Ref:
Kronqvist J, Bernal D E, Grossmann I E. Using regularization and second order information in outer approximation for convex MINLP[J]. Mathematical Programming, 2020, 180(1): 285-310.

Problem type:   nonconvex MINLP
        size:   1  binary variable
                1  continuous variables
                3  constraints

"""
from pyomo.environ import *


class SimpleMINLP5(ConcreteModel):
    def __init__(self, *args, **kwargs):
        """Create the problem."""
        kwargs.setdefault('name', 'SimpleMINLP5')
        super(SimpleMINLP5, self).__init__(*args, **kwargs)
        m = self

        m.x = Var(within=Reals, bounds=(1, 20), initialize=5.29)
        m.y = Var(within=Integers, bounds=(1, 20), initialize=3)

        m.objective = Objective(expr=0.3 * (m.x - 8)**2 + 0.04 *
                                (m.y - 6)**4 + 0.1*exp(2*m.x)*(m.y**(-4)), sense=minimize)

        m.c1 = Constraint(expr=6 * m.x + m.y <= 60)
        m.c2 = Constraint(expr=1/m.x + 1/m.x - m.x**0.5*m.y**0.5 <= -1)
        m.c3 = Constraint(expr=2 * m.x - 5 * m.y <= -1)
        m.optimal_value = 3.6572
