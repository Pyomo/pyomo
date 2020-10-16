# -*- coding: utf-8 -*-
""" Example in the online doc.

The expected optimal solution value is 2.438447187191098.

    Problem type:    convex MINLP
            size:    1  binary variable
                     1  continuous variables
                     2  constraints

"""
from __future__ import division

from six import iteritems

from pyomo.environ import (Binary, ConcreteModel, Constraint, Reals,
                           Objective, Param, RangeSet, Var, exp, minimize, log)


class OnlineDocExample(ConcreteModel):

    def __init__(self, *args, **kwargs):
        """Create the problem."""
        kwargs.setdefault('name', 'OnlineDocExample')
        super(OnlineDocExample, self).__init__(*args, **kwargs)
        model = self
        model.x = Var(bounds=(1.0, 10.0), initialize=5.0)
        model.y = Var(within=Binary)
        model.c1 = Constraint(expr=(model.x-4.0)**2 -
                              model.x <= 50.0*(1-model.y))
        model.c2 = Constraint(expr=model.x*log(model.x) + 5 <= 50.0*(model.y))
        model.objective = Objective(expr=model.x, sense=minimize)
