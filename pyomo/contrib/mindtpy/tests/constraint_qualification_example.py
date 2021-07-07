# -*- coding: utf-8 -*-
""" Example of constraint qualification.

The expected optimal solution value is 3.

    Problem type:    convex MINLP
            size:    1  binary variable
                     1  continuous variables
                     2  constraints

"""
from __future__ import division

from pyomo.environ import (Binary, ConcreteModel, Constraint, Reals,
                           Objective, Param, RangeSet, Var, exp, minimize, log)


class ConstraintQualificationExample(ConcreteModel):

    def __init__(self, *args, **kwargs):
        """Create the problem."""
        kwargs.setdefault('name', 'ConstraintQualificationExample')
        super(ConstraintQualificationExample, self).__init__(*args, **kwargs)
        m = self
        m.x = Var(bounds=(1.0, 10.0), initialize=5.0)
        m.y = Var(within=Binary)
        m.c1 = Constraint(expr=(m.x-3.0)**2 <= 50.0*(1-m.y))
        m.c2 = Constraint(expr=m.x*log(m.x)+5.0 <= 50.0*(m.y))
        m.objective = Objective(expr=m.x, sense=minimize)
        m.optimal_value = 3
