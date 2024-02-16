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

This is the more verbose formulation of the same problem given in
eight_proc_model.py.

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
from pyomo.gdp import Disjunct, Disjunction


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
    CF = m.CF = Param(m.units, initialize=fixed_cost)

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
    m.use_unit1 = Disjunct()
    m.use_unit1.inout1 = Constraint(expr=exp(m.flow[3]) - 1 == m.flow[2])
    m.use_unit1.no_unit2_flow1 = Constraint(expr=m.flow[4] == 0)
    m.use_unit1.no_unit2_flow2 = Constraint(expr=m.flow[5] == 0)
    m.use_unit2 = Disjunct()
    m.use_unit2.inout2 = Constraint(expr=exp(m.flow[5] / 1.2) - 1 == m.flow[4])
    m.use_unit2.no_unit1_flow1 = Constraint(expr=m.flow[2] == 0)
    m.use_unit2.no_unit1_flow2 = Constraint(expr=m.flow[3] == 0)

    m.use_unit3 = Disjunct()
    m.use_unit3.inout3 = Constraint(expr=1.5 * m.flow[9] + m.flow[10] == m.flow[8])
    m.no_unit3 = Disjunct()
    m.no_unit3.no_unit3_flow1 = Constraint(expr=m.flow[9] == 0)
    m.no_unit3.flow_pass_through = Constraint(expr=m.flow[10] == m.flow[8])

    m.use_unit4 = Disjunct()
    m.use_unit4.inout4 = Constraint(expr=1.25 * (m.flow[12] + m.flow[14]) == m.flow[13])
    m.use_unit4.no_unit5_flow = Constraint(expr=m.flow[15] == 0)
    m.use_unit5 = Disjunct()
    m.use_unit5.inout5 = Constraint(expr=m.flow[15] == 2 * m.flow[16])
    m.use_unit5.no_unit4_flow1 = Constraint(expr=m.flow[12] == 0)
    m.use_unit5.no_unit4_flow2 = Constraint(expr=m.flow[14] == 0)
    m.no_unit4or5 = Disjunct()
    m.no_unit4or5.no_unit5_flow = Constraint(expr=m.flow[15] == 0)
    m.no_unit4or5.no_unit4_flow1 = Constraint(expr=m.flow[12] == 0)
    m.no_unit4or5.no_unit4_flow2 = Constraint(expr=m.flow[14] == 0)

    m.use_unit6 = Disjunct()
    m.use_unit6.inout6 = Constraint(expr=exp(m.flow[20] / 1.5) - 1 == m.flow[19])
    m.use_unit6.no_unit7_flow1 = Constraint(expr=m.flow[21] == 0)
    m.use_unit6.no_unit7_flow2 = Constraint(expr=m.flow[22] == 0)
    m.use_unit7 = Disjunct()
    m.use_unit7.inout7 = Constraint(expr=exp(m.flow[22]) - 1 == m.flow[21])
    m.use_unit7.no_unit6_flow1 = Constraint(expr=m.flow[19] == 0)
    m.use_unit7.no_unit6_flow2 = Constraint(expr=m.flow[20] == 0)
    m.no_unit6or7 = Disjunct()
    m.no_unit6or7.no_unit7_flow1 = Constraint(expr=m.flow[21] == 0)
    m.no_unit6or7.no_unit7_flow2 = Constraint(expr=m.flow[22] == 0)
    m.no_unit6or7.no_unit6_flow = Constraint(expr=m.flow[19] == 0)
    m.no_unit6or7.no_unit6_flow2 = Constraint(expr=m.flow[20] == 0)

    m.use_unit8 = Disjunct()
    m.use_unit8.inout8 = Constraint(expr=exp(m.flow[18]) - 1 == m.flow[10] + m.flow[17])
    m.no_unit8 = Disjunct()
    m.no_unit8.no_unit8_flow1 = Constraint(expr=m.flow[10] == 0)
    m.no_unit8.no_unit8_flow2 = Constraint(expr=m.flow[17] == 0)
    m.no_unit8.no_unit8_flow3 = Constraint(expr=m.flow[18] == 0)

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
    m.use1or2 = Disjunction(expr=[m.use_unit1, m.use_unit2])
    m.use4or5maybe = Disjunction(expr=[m.use_unit4, m.use_unit5, m.no_unit4or5])
    m.use4or5 = Constraint(
        expr=m.use_unit4.indicator_var + m.use_unit5.indicator_var <= 1
    )
    m.use6or7maybe = Disjunction(expr=[m.use_unit6, m.use_unit7, m.no_unit6or7])
    m.use4implies6or7 = Constraint(
        expr=m.use_unit6.indicator_var
        + m.use_unit7.indicator_var
        - m.use_unit4.indicator_var
        == 0
    )
    m.use3maybe = Disjunction(expr=[m.use_unit3, m.no_unit3])
    m.either3ornot = Constraint(
        expr=m.use_unit3.indicator_var + m.no_unit3.indicator_var == 1
    )
    m.use8maybe = Disjunction(expr=[m.use_unit8, m.no_unit8])
    m.use3implies8 = Constraint(
        expr=m.use_unit3.indicator_var - m.use_unit8.indicator_var <= 0
    )

    """Profit (objective) function definition"""
    m.profit = Objective(
        expr=sum(
            getattr(m, 'use_unit%s' % (unit,)).indicator_var * CF[unit]
            for unit in m.units
        )
        + sum(m.flow[stream] * CV[stream] for stream in m.streams)
        + CONSTANT,
        sense=minimize,
    )

    """Bound definitions"""
    # x (flow) upper bounds
    x_ubs = {3: 2, 5: 2, 9: 2, 10: 1, 14: 1, 17: 2, 19: 2, 21: 2, 25: 3}
    for i, x_ub in x_ubs.items():
        m.flow[i].setub(x_ub)

    # # optimal solution
    # m.use_unit1.indicator_var = 0
    # m.use_unit2.indicator_var = 1
    # m.use_unit3.indicator_var = 0
    # m.no_unit3.indicator_var = 1
    # m.use_unit4.indicator_var = 1
    # m.use_unit5.indicator_var = 0
    # m.no_unit4or5.indicator_var = 0
    # m.use_unit6.indicator_var = 1
    # m.use_unit7.indicator_var = 0
    # m.no_unit6or7.indicator_var = 0
    # m.use_unit8.indicator_var = 1
    # m.no_unit8.indicator_var = 0

    return m
