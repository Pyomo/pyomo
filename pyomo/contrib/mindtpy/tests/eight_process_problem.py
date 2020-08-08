"""Re-implementation of eight-process problem.

Re-implementation of Duran example 3 superstructure synthesis problem in Pyomo
Author: Qi Chen <https://github.com/qtothec>.

Ref:
    SELECT OPTIMAL PROCESS FROM WITHIN GIVEN SUPERSTRUCTURE.
    MARCO DURAN , PH.D. THESIS (EX3) , 1984.
    CARNEGIE-MELLON UNIVERSITY , PITTSBURGH , PA.

The expected optimal solution value is 68.0.

(original problem, my implementation may vary)
    Problem type:    convex MINLP
            size:    8  binary variables
                    26  continuous variables
                    32  constraints

Pictoral representation can be found on
page 969 of Turkay & Grossmann, 1996.
http://dx.doi.org/10.1016/0098-1354(95)00219-7

"""
from __future__ import division

from six import iteritems

from pyomo.environ import (Binary, ConcreteModel, Constraint, NonNegativeReals,
                           Objective, Param, RangeSet, Var, exp, minimize)


class EightProcessFlowsheet(ConcreteModel):
    """Flowsheet for the 8 process problem."""

    def __init__(self, convex=True, *args, **kwargs):
        """Create the flowsheet."""
        kwargs.setdefault('name', 'DuranEx3')
        super(EightProcessFlowsheet, self).__init__(*args, **kwargs)
        m = self

        """Set declarations"""
        I = m.I = RangeSet(2, 25, doc="process streams")
        J = m.J = RangeSet(1, 8, doc="process units")
        m.PI = RangeSet(1, 4, doc="integer constraints")
        m.DS = RangeSet(1, 4, doc="design specifications")
        """
        1: Unit 8
        2: Unit 8
        3: Unit 4
        4: Unit 4
        """
        m.MB = RangeSet(1, 7, doc="mass balances")
        """Material balances:
        1: 4-6-7
        2: 3-5-8
        3: 4-5
        4: 1-2
        5: 1-2-3
        6: 6-7-4
        7: 6-7
        """

        """Parameter and initial point declarations"""
        # FIXED COST INVESTMENT COEFF FOR PROCESS UNITS
        # Format: process #: cost
        fixed_cost = {1: 5, 2: 8, 3: 6, 4: 10, 5: 6, 6: 7, 7: 4, 8: 5}
        CF = m.CF = Param(J, initialize=fixed_cost)

        # VARIABLE COST COEFF FOR PROCESS UNITS - STREAMS
        # Format: stream #: cost
        variable_cost = {3: -10, 5: -15, 9: -40, 19: 25, 21: 35, 25: -35,
                         17: 80, 14: 15, 10: 15, 2: 1, 4: 1, 18: -65, 20: -60,
                         22: -80}
        CV = m.CV = Param(I, initialize=variable_cost, default=0)

        # initial point information for equipment selection (for each NLP
        # subproblem)
        initY = {
            'sub1': {1: 1, 2: 0, 3: 1, 4: 1, 5: 0, 6: 0, 7: 1, 8: 1},
            'sub2': {1: 0, 2: 1, 3: 1, 4: 1, 5: 0, 6: 1, 7: 0, 8: 1},
            'sub3': {1: 1, 2: 0, 3: 1, 4: 0, 5: 1, 6: 0, 7: 0, 8: 1}
        }
        # initial point information for stream flows
        initX = {1: 0, 2: 2, 3: 1.5, 4: 0, 5: 0, 6: 0.75, 7: 0.5, 8: 0.5,
                 9: 0.75, 10: 0, 11: 1.5, 12: 1.34, 13: 2, 14: 2.5, 15: 0,
                 16: 0, 17: 2, 18: 0.75, 19: 2, 20: 1.5, 21: 0, 22: 0,
                 23: 1.7, 24: 1.5, 25: 0.5}

        """Variable declarations"""
        # BINARY VARIABLE DENOTING EXISTENCE-NONEXISTENCE
        Y = m.Y = Var(J, domain=Binary, initialize=initY['sub1'])
        # FLOWRATES OF PROCESS STREAMS
        X = m.X = Var(I, domain=NonNegativeReals, initialize=initX)
        # OBJECTIVE FUNCTION CONSTANT TERM
        CONSTANT = m.constant = Param(initialize=122.0)

        """Constraint definitions"""
        # INPUT-OUTPUT RELATIONS FOR process units 1 through 8
        m.inout3 = Constraint(expr=1.5 * m.X[9] + m.X[10] == m.X[8])
        m.inout4 = Constraint(expr=1.25 * (m.X[12] + m.X[14]) == m.X[13])
        m.inout5 = Constraint(expr=m.X[15] == 2 * m.X[16])
        if convex:
            m.inout1 = Constraint(expr=exp(m.X[3]) - 1 <= m.X[2])
            m.inout2 = Constraint(expr=exp(m.X[5] / 1.2) - 1 <= m.X[4])
            m.inout7 = Constraint(expr=exp(m.X[22]) - 1 <= m.X[21])
            m.inout8 = Constraint(expr=exp(m.X[18]) - 1 <= m.X[10] + m.X[17])
            m.inout6 = Constraint(expr=exp(m.X[20] / 1.5) - 1 <= m.X[19])
        else:
            m.inout1 = Constraint(expr=exp(m.X[3]) - 1 == m.X[2])
            m.inout2 = Constraint(expr=exp(m.X[5] / 1.2) - 1 == m.X[4])
            m.inout7 = Constraint(expr=exp(m.X[22]) - 1 == m.X[21])
            m.inout8 = Constraint(expr=exp(m.X[18]) - 1 == m.X[10] + m.X[17])
            m.inout6 = Constraint(expr=exp(m.X[20] / 1.5) - 1 == m.X[19])

        # Mass balance equations
        m.massbal1 = Constraint(expr=m.X[13] == m.X[19] + m.X[21])
        m.massbal2 = Constraint(expr=m.X[17] == m.X[9] + m.X[16] + m.X[25])
        m.massbal3 = Constraint(expr=m.X[11] == m.X[12] + m.X[15])
        m.massbal4 = Constraint(expr=m.X[3] + m.X[5] == m.X[6] + m.X[11])
        m.massbal5 = Constraint(expr=m.X[6] == m.X[7] + m.X[8])
        m.massbal6 = Constraint(expr=m.X[23] == m.X[20] + m.X[22])
        m.massbal7 = Constraint(expr=m.X[23] == m.X[14] + m.X[24])

        # process specifications
        m.specs1 = Constraint(expr=m.X[10] <= 0.8 * m.X[17])
        m.specs2 = Constraint(expr=m.X[10] >= 0.4 * m.X[17])
        m.specs3 = Constraint(expr=m.X[12] <= 5 * m.X[14])
        m.specs4 = Constraint(expr=m.X[12] >= 2 * m.X[14])

        # Logical constraints (big-M) for each process.
        # These allow for flow iff unit j exists
        m.logical1 = Constraint(expr=m.X[2] <= 10 * m.Y[1])
        m.logical2 = Constraint(expr=m.X[4] <= 10 * m.Y[2])
        m.logical3 = Constraint(expr=m.X[9] <= 10 * m.Y[3])
        m.logical4 = Constraint(expr=m.X[12] + m.X[14] <= 10 * m.Y[4])
        m.logical5 = Constraint(expr=m.X[15] <= 10 * m.Y[5])
        m.logical6 = Constraint(expr=m.X[19] <= 10 * m.Y[6])
        m.logical7 = Constraint(expr=m.X[21] <= 10 * m.Y[7])
        m.logical8 = Constraint(expr=m.X[10] + m.X[17] <= 10 * m.Y[8])

        # pure integer constraints
        m.pureint1 = Constraint(expr=m.Y[1] + m.Y[2] == 1)
        m.pureint2 = Constraint(expr=m.Y[4] + m.Y[5] <= 1)
        m.pureint3 = Constraint(expr=m.Y[6] + m.Y[7] - m.Y[4] == 0)
        m.pureint4 = Constraint(expr=m.Y[3] - m.Y[8] <= 0)

        """Cost (objective) function definition"""
        m.cost = Objective(expr=sum(Y[j] * CF[j] for j in J) +
                           sum(X[i] * CV[i] for i in I) + CONSTANT,
                           sense=minimize)

        """Bound definitions"""
        # x (flow) upper bounds
        # x_ubs = {3: 2, 5: 2, 9: 2, 10: 1, 14: 1, 17: 2, 19: 2, 21: 2, 25: 3}
        x_ubs = {2: 10, 3: 2, 4: 10, 5: 2, 9: 2, 10: 1, 14: 1, 17: 2, 18: 10, 19: 2,
                 20: 10, 21: 2, 22: 10, 25: 3}  # add bounds for variables in nonlinear constraints
        for i, x_ub in iteritems(x_ubs):
            X[i].setub(x_ub)
