#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

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

from pyomo.environ import (
    Binary,
    ConcreteModel,
    Constraint,
    NonNegativeReals,
    Objective,
    RangeSet,
    Var,
    minimize,
    Block,
)
from pyomo.common.collections import ComponentMap
from pyomo.contrib.mindtpy.tests.MINLP_simple_grey_box import (
    GreyBoxModel,
    build_model_external,
)


class SimpleMINLP(ConcreteModel):
    """Convex MINLP problem Assignment 6 APSE."""

    def __init__(self, grey_box=False, *args, **kwargs):
        """Create the problem."""
        kwargs.setdefault('name', 'SimpleMINLP')
        if grey_box and GreyBoxModel is None:
            m = None
            return

        super(SimpleMINLP, self).__init__(*args, **kwargs)
        m = self

        """Set declarations"""
        I = m.I = RangeSet(1, 2, doc='continuous variables')
        J = m.J = RangeSet(1, 3, doc='discrete variables')

        # initial point information for discrete variables
        initY = {
            'sub1': {1: 1, 2: 1, 3: 1},
            'sub2': {1: 0, 2: 1, 3: 1},
            'sub3': {1: 1, 2: 0, 3: 1},
            'sub4': {1: 1, 2: 1, 3: 0},
            'sub5': {1: 0, 2: 0, 3: 0},
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
        m.const1 = Constraint(expr=(m.X[1] - 2) ** 2 - m.X[2] <= 0)
        m.const2 = Constraint(expr=m.X[1] - 2 * m.Y[1] >= 0)
        m.const3 = Constraint(expr=m.X[1] - m.X[2] - 4 * (1 - m.Y[2]) <= 0)
        m.const4 = Constraint(expr=m.X[1] - (1 - m.Y[1]) >= 0)
        m.const5 = Constraint(expr=m.X[2] - m.Y[2] >= 0)
        m.const6 = Constraint(expr=m.X[1] + m.X[2] >= 3 * m.Y[3])
        m.const7 = Constraint(expr=m.Y[1] + m.Y[2] + m.Y[3] >= 1)

        """Cost (objective) function definition"""
        m.objective = Objective(
            expr=Y[1] + 1.5 * Y[2] + 0.5 * Y[3] + X[1] ** 2 + X[2] ** 2, sense=minimize
        )

        if not grey_box:
            m.objective = Objective(
                expr=Y[1] + 1.5 * Y[2] + 0.5 * Y[3] + X[1] ** 2 + X[2] ** 2,
                sense=minimize,
            )
        else:

            def _model_i(b):
                build_model_external(b)

            m.my_block = Block(rule=_model_i)

            for i in m.I:

                def eq_inputX(m):
                    return m.X[i] == m.my_block.egb.inputs["X" + str(i)]

                con_name = "con_X_" + str(i)
                m.add_component(con_name, Constraint(expr=eq_inputX))

            for j in m.J:

                def eq_inputY(m):
                    return m.Y[j] == m.my_block.egb.inputs["Y" + str(j)]

                con_name = "con_Y_" + str(j)
                m.add_component(con_name, Constraint(expr=eq_inputY))

            # add objective
            m.objective = Objective(expr=m.my_block.egb.outputs['z'], sense=minimize)

        """Bound definitions"""
        # x (continuous) upper bounds
        x_ubs = {1: 4, 2: 4}
        for i, x_ub in x_ubs.items():
            X[i].setub(x_ub)

        m.optimal_value = 3.5
        m.optimal_solution = ComponentMap()
        m.optimal_solution[m.X[1]] = 1.0
        m.optimal_solution[m.X[2]] = 1.0
        m.optimal_solution[m.Y[1]] = 0.0
        m.optimal_solution[m.Y[2]] = 1.0
        m.optimal_solution[m.Y[3]] = 0.0
