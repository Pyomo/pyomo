#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Disjunctive re-implementation of eight-process problem.

Re-implementation of Duran example 3 superstructure synthesis problem in Pyomo
with full leverage of GDP framework.

This is a convex MINLP problem that has been formulated as a GDP model. The
expected optimal solution value is 68.0.

Ref:
    SELECT OPTIMAL PROCESS FROM WITHIN GIVEN SUPERSTRUCTURE.
    MARCO DURAN , PH.D. THESIS (EX3) , 1984.
    CARNEGIE-MELLON UNIVERSITY , PITTSBURGH , PA.

(original problem, my implementation may vary)
    Problem type:    convex MINLP
            size:    8  binary variables
                    26  continuous variables
                    32  constraints

Pictoral representation can be found on
page 969 of Turkay & Grossmann, 1996.
http://dx.doi.org/10.1016/0098-1354(95)00219-7

"""

from pyomo.environ import (
    ConcreteModel,
    Constraint,
    NonNegativeReals,
    Objective,
    Param,
    RangeSet,
    Var,
    exp,
    minimize,
)
from pyomo.gdp import Disjunction


def build_eight_process_flowsheet():
    """Build flowsheet for the 8 process problem."""
    m = ConcreteModel(name='DuranEx3 Disjunctive')

    """Set declarations"""
    m.streams = RangeSet(2, 25, doc="process streams")
    m.units = RangeSet(1, 8, doc="process units")

    """Parameter and initial point declarations"""
    # FIXED COST INVESTMENT COEFF FOR PROCESS UNITS
    # Format: process #: cost
    fixed_cost = {1: 5, 2: 8, 3: 6, 4: 10, 5: 6, 6: 7, 7: 4, 8: 5}
    m.CF = Param(m.units, initialize=fixed_cost)

    def fixed_cost_bounds(m, unit):
        return (0, m.CF[unit])

    m.yCF = Var(m.units, initialize=0, bounds=fixed_cost_bounds)

    # VARIABLE COST COEFF FOR PROCESS UNITS - STREAMS
    # Format: stream #: cost
    variable_cost = {
        3: -10,
        5: -15,
        9: -40,
        19: 25,
        21: 35,
        25: -35,
        17: 80,
        14: 15,
        10: 15,
        2: 1,
        4: 1,
        18: -65,
        20: -60,
        22: -80,
    }
    CV = m.CV = Param(m.streams, initialize=variable_cost, default=0)

    # initial point information for stream flows
    initX = {
        2: 2,
        3: 1.5,
        6: 0.75,
        7: 0.5,
        8: 0.5,
        9: 0.75,
        11: 1.5,
        12: 1.34,
        13: 2,
        14: 2.5,
        17: 2,
        18: 0.75,
        19: 2,
        20: 1.5,
        23: 1.7,
        24: 1.5,
        25: 0.5,
    }

    """Variable declarations"""
    # FLOWRATES OF PROCESS STREAMS
    m.flow = Var(m.streams, domain=NonNegativeReals, initialize=initX, bounds=(0, 10))
    # OBJECTIVE FUNCTION CONSTANT TERM
    CONSTANT = m.constant = Param(initialize=122.0)

    """Constraint definitions"""
    # INPUT-OUTPUT RELATIONS FOR process units 1 through 8
    m.use_unit_1or2 = Disjunction(
        expr=[
            # use unit 1 disjunct
            [
                m.yCF[1] == m.CF[1],
                exp(m.flow[3]) - 1 == m.flow[2],
                m.flow[4] == 0,
                m.flow[5] == 0,
            ],
            # use unit 2 disjunct
            [
                m.yCF[2] == m.CF[2],
                exp(m.flow[5] / 1.2) - 1 == m.flow[4],
                m.flow[2] == 0,
                m.flow[3] == 0,
            ],
        ]
    )
    m.use_unit_3ornot = Disjunction(
        expr=[
            # Use unit 3 disjunct
            [m.yCF[3] == m.CF[3], 1.5 * m.flow[9] + m.flow[10] == m.flow[8]],
            # No unit 3 disjunct
            [m.flow[9] == 0, m.flow[10] == m.flow[8]],
        ]
    )
    m.use_unit_4or5ornot = Disjunction(
        expr=[
            # Use unit 4 disjunct
            [
                m.yCF[4] == m.CF[4],
                1.25 * (m.flow[12] + m.flow[14]) == m.flow[13],
                m.flow[15] == 0,
            ],
            # Use unit 5 disjunct
            [
                m.yCF[5] == m.CF[5],
                m.flow[15] == 2 * m.flow[16],
                m.flow[12] == 0,
                m.flow[14] == 0,
            ],
            # No unit 4 or 5 disjunct
            [m.flow[15] == 0, m.flow[12] == 0, m.flow[14] == 0],
        ]
    )
    m.use_unit_6or7ornot = Disjunction(
        expr=[
            # use unit 6 disjunct
            [
                m.yCF[6] == m.CF[6],
                exp(m.flow[20] / 1.5) - 1 == m.flow[19],
                m.flow[21] == 0,
                m.flow[22] == 0,
            ],
            # use unit 7 disjunct
            [
                m.yCF[7] == m.CF[7],
                exp(m.flow[22]) - 1 == m.flow[21],
                m.flow[19] == 0,
                m.flow[20] == 0,
            ],
            # No unit 6 or 7 disjunct
            [m.flow[21] == 0, m.flow[22] == 0, m.flow[19] == 0, m.flow[20] == 0],
        ]
    )
    m.use_unit_8ornot = Disjunction(
        expr=[
            # use unit 8 disjunct
            [m.yCF[8] == m.CF[8], exp(m.flow[18]) - 1 == m.flow[10] + m.flow[17]],
            # no unit 8 disjunct
            [m.flow[10] == 0, m.flow[17] == 0, m.flow[18] == 0],
        ]
    )

    # Mass balance equations
    m.massbal1 = Constraint(expr=m.flow[13] == m.flow[19] + m.flow[21])
    m.massbal2 = Constraint(expr=m.flow[17] == m.flow[9] + m.flow[16] + m.flow[25])
    m.massbal3 = Constraint(expr=m.flow[11] == m.flow[12] + m.flow[15])
    m.massbal4 = Constraint(expr=m.flow[3] + m.flow[5] == m.flow[6] + m.flow[11])
    m.massbal5 = Constraint(expr=m.flow[6] == m.flow[7] + m.flow[8])
    m.massbal6 = Constraint(expr=m.flow[23] == m.flow[20] + m.flow[22])
    m.massbal7 = Constraint(expr=m.flow[23] == m.flow[14] + m.flow[24])

    # process specifications
    m.specs1 = Constraint(expr=m.flow[10] <= 0.8 * m.flow[17])
    m.specs2 = Constraint(expr=m.flow[10] >= 0.4 * m.flow[17])
    m.specs3 = Constraint(expr=m.flow[12] <= 5 * m.flow[14])
    m.specs4 = Constraint(expr=m.flow[12] >= 2 * m.flow[14])

    # pure integer constraints
    m.use4implies6or7 = Constraint(
        expr=m.use_unit_6or7ornot.disjuncts[0].binary_indicator_var
        + m.use_unit_6or7ornot.disjuncts[1].binary_indicator_var
        - m.use_unit_4or5ornot.disjuncts[0].binary_indicator_var
        == 0
    )
    m.use3implies8 = Constraint(
        expr=m.use_unit_3ornot.disjuncts[0].binary_indicator_var
        - m.use_unit_8ornot.disjuncts[0].binary_indicator_var
        <= 0
    )

    """Profit (objective) function definition"""
    m.profit = Objective(
        expr=sum(m.yCF[unit] for unit in m.units)
        + sum(m.flow[stream] * CV[stream] for stream in m.streams)
        + CONSTANT,
        sense=minimize,
    )

    """Bound definitions"""
    # x (flow) upper bounds
    x_ubs = {3: 2, 5: 2, 9: 2, 10: 1, 14: 1, 17: 2, 19: 2, 21: 2, 25: 3}
    for i, x_ub in x_ubs.items():
        m.flow[i].setub(x_ub)

    # Optimal solution uses units 2, 4, 6, 8 with objective value 68.

    return m
