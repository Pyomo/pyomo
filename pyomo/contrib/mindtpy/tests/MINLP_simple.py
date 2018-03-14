"""Implementation of MINLP problem in Assignment 6 of the Advanced PSE lecture at CMU.

Author: David Bernal <https://github.com/bernalde>

The expected optimal solution is 3.5.

Ref:
    IGNACIO GROSSMANN.
    CARNEGIE-MELLON UNIVERSITY , PITTSBURGH , PA.

    Problem type:    convex MINLP
            size:    3  binary variables
                     3  continuous variables
                     7  constraints

"""
from __future__ import division

from six import iteritems

from pyomo.environ import (Binary, ConcreteModel, Constraint,
                           NonNegativeReals, Objective, Param,
                           RangeSet, Var, exp, minimize)


class SimpleMINLP(ConcreteModel):
    """Convex MINLP problem Assignment 6 APSE."""

    def __init__(self, *args, **kwargs):
        """Create the problem."""
        kwargs.setdefault('name', 'SimpleMINLP')
        super(SimpleMINLP, self).__init__(*args, **kwargs)
        m = self

        """Set declarations"""
        I = m.I = RangeSet(1, 2, doc="continuous variables")
        J = m.J = RangeSet(1, 3, doc="discrete variables")

        # initial point information for discrete variables
        initY = {
            'sub1': {1: 1, 2: 1, 3: 1},
            'sub2': {1: 0, 2: 1, 3: 1},
            'sub3': {1: 1, 2: 0, 3: 1},
            'sub4': {1: 1, 2: 1, 3: 0},
            'sub5': {1: 0, 2: 0, 3: 0}
        }
        # initial point information for continuous variables
        initX = {1: 0, 2: 0}

        """Variable declarations"""
        # DISCRETE VARIABLES
        Y = m.Y = Var(J, domain=Binary, initialize=initY['sub2'])
        # CONTINUOUS VARIABLES
        X = m.X = Var(I, domain=NonNegativeReals, initialize=initX)

        """Constraint definitions"""
        # CONSTRAINTS
        m.const1 = Constraint(expr=(m.X[1] - 2)**2 - m.X[2] <= 0)
        m.const2 = Constraint(expr=m.X[1] - 2 * m.Y[1] >= 0)
        m.const3 = Constraint(expr=m.X[1] - m.X[2] - 4 * (1 - m.Y[2]) <= 0)
        m.const4 = Constraint(expr=m.X[1] - (1 - m.Y[1]) >= 0)
        m.const5 = Constraint(expr=m.X[2] - m.Y[2] >= 0)
        m.const6 = Constraint(expr=m.X[1] + m.X[2] >= 3 * m.Y[3])
        m.const7 = Constraint(expr=m.Y[1] + m.Y[2] + m.Y[3] >= 1)

        """Cost (objective) function definition"""
        m.cost = Objective(expr=Y[1] + 1.5*Y[2] + 0.5*Y[3] + X[1]**2 + X[2]**2,
                             sense=minimize)

        """Bound definitions"""
        # x (continuous) upper bounds
        x_ubs = {1: 4, 2: 4}
        for i, x_ub in iteritems(x_ubs):
            X[i].setub(x_ub)
