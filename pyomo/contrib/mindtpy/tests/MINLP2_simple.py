#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Re-implementation of example 1 of Outer approximation and ECP.

Re-implementation of Duran example 1 as written by Westerlund
MINLP test problem in Pyomo
Author: David Bernal <https://github.com/bernalde>

The expected optimal solution value is 6.00976.

Ref:
    Duran, Marco A., and Ignacio E. Grossmann.
    "An outer-approximation algorithm for a class of mixed-integer nonlinear
    programs."
    Mathematical programming 36.3 (1986): 307-339.
    Westerlund, Tapio, and Frank Pettersson.
    "An extended cutting plane method for solving convex MINLP problems."
    Computers & Chemical Engineering 19 (1995): 131-136.
    Example 1

    Problem type:    convex MINLP
            size:    3  binary variables
                     4  continuous variables
                     11  constraints


"""
from __future__ import division

from six import iteritems

from pyomo.environ import (Binary, ConcreteModel, Constraint, NonNegativeReals,
                           Objective, RangeSet, Var, minimize, log)


class SimpleMINLP(ConcreteModel):
    """Example 1 Outer Approximation and Extended Cutting Planes."""

    def __init__(self, *args, **kwargs):
        """Create the problem."""
        kwargs.setdefault('name', 'DuranEx1')
        super(SimpleMINLP, self).__init__(*args, **kwargs)
        m = self

        """Set declarations"""
        I = m.I = RangeSet(1, 4, doc="continuous variables")
        J = m.J = RangeSet(1, 3, doc="discrete variables")

        # initial point information for discrete variables
        initY = {1: 1, 2: 0, 3: 1}
        # initial point information for continuous variables
        initX = {1: 0, 2: 0, 3: 0, 4: 0}

        """Variable declarations"""
        # DISCRETE VARIABLES
        Y = m.Y = Var(J, domain=Binary, initialize=initY)
        # CONTINUOUS VARIABLES
        X = m.X = Var(I, domain=NonNegativeReals, initialize=initX, bounds=(0, 2))

        """Constraint definitions"""
        # CONSTRAINTS
        m.const1 = Constraint(expr=0.8*log(X[2] + 1) + 0.96*log(X[1] - X[2] + 1)
         - 0.8*X[3] >= 0)
        m.const2 = Constraint(expr=log(X[2] + 1) + 1.2*log(X[1] - X[2] + 1)
          - X[3] - 2*Y[3] >= -2)
        m.const3 = Constraint(expr=10*X[1] - 7*X[3]
        - 18*log(X[2] + 1) - 19.2*log(X[1] - X[2] + 1) + 10 - X[4] <= 0)
        m.const4 = Constraint(expr=X[2] - X[1] <= 0)
        m.const5 = Constraint(expr=X[2] - 2*Y[1] <= 0)
        m.const6 = Constraint(expr=X[1] - X[2] - 2*Y[2] <= 0)
        m.const7 = Constraint(expr=Y[1] + Y[2] <= 1)

        """Cost (objective) function definition"""
        m.cost = Objective(expr=+5*Y[1] + 6*Y[2] + 8*Y[3] + X[4], sense=minimize)

        """Bound definitions"""
        # x (continuous) upper bounds
        x_ubs = {1: 2, 2: 2, 3: 1, 4: 100}
        for i, x_ub in iteritems(x_ubs):
            X[i].setub(x_ub)
