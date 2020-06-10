""" Example of constraint qualification.

The expected optimal solution value is 3.

    Problem type:    convex MINLP
            size:    1  binary variable
                     1  continuous variables
                     2  constraints

"""
from __future__ import division

from six import iteritems

from pyomo.environ import (Binary, ConcreteModel, Constraint, Reals,
                           Objective, Param, RangeSet, Var, exp, minimize, log)


class ConstraintQualificationExample(ConcreteModel):

    def __init__(self, *args, **kwargs):
        """Create the problem."""
        kwargs.setdefault('name', 'ConstraintQualificationExample')
        super(ConstraintQualificationExample, self).__init__(*args, **kwargs)
        model = self
        model.x = Var(bounds=(1.0, 10.0), initialize=5.0)
        model.y = Var(within=Binary)
        model.c1 = Constraint(expr=(model.x-3.0)**2 <= 50.0*(1-model.y))
        model.c2 = Constraint(expr=model.x*log(model.x)+5.0 <= 50.0*(model.y))
        model.objective = Objective(expr=model.x, sense=minimize)
