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

from pyomo.core.expr.logical_expr import land, lor
from pyomo.core.plugins.transform.logical_to_linear import (
    update_boolean_vars_from_binary,
)
from pyomo.environ import (
    ConcreteModel,
    Constraint,
    ConstraintList,
    NonNegativeReals,
    Objective,
    Param,
    RangeSet,
    Reference,
    Var,
    exp,
    minimize,
    LogicalConstraint,
)
from pyomo.gdp import Disjunct, Disjunction
from pyomo.opt import SolverFactory


def build_eight_process_flowsheet():
    """Build flowsheet for the 8 process problem."""
    m = ConcreteModel(name='DuranEx3 Disjunctive')

    """Set declarations"""
    m.streams = RangeSet(2, 25, doc="process streams")
    m.units = RangeSet(1, 8, doc="process units")

    no_unit_zero_flows = {
        1: (2, 3),
        2: (4, 5),
        3: (9,),
        4: (12, 14),
        5: (15,),
        6: (19, 20),
        7: (21, 22),
        8: (10, 17, 18),
    }

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
    @m.Disjunct(m.units)
    def use_unit(disj, unit):
        disj.impose_fixed_cost = Constraint(expr=m.yCF[unit] == m.CF[unit])
        disj.io_relation = ConstraintList(doc="Input-Output relationship")

    @m.Disjunct(m.units)
    def no_unit(disj, unit):
        disj.no_flow = ConstraintList()
        for stream in no_unit_zero_flows[unit]:
            disj.no_flow.add(expr=m.flow[stream] == 0)

    @m.Disjunction(m.units)
    def use_unit_or_not(m, unit):
        return [m.use_unit[unit], m.no_unit[unit]]

    # Note: this could be done in an automated manner by indexed construction.
    # Below is just for illustration
    m.use_unit[1].io_relation.add(expr=exp(m.flow[3]) - 1 == m.flow[2])
    m.use_unit[2].io_relation.add(expr=exp(m.flow[5] / 1.2) - 1 == m.flow[4])
    m.use_unit[3].io_relation.add(expr=1.5 * m.flow[9] + m.flow[10] == m.flow[8])
    m.use_unit[4].io_relation.add(expr=1.25 * (m.flow[12] + m.flow[14]) == m.flow[13])
    m.use_unit[5].io_relation.add(expr=m.flow[15] == 2 * m.flow[16])
    m.use_unit[6].io_relation.add(expr=exp(m.flow[20] / 1.5) - 1 == m.flow[19])
    m.use_unit[7].io_relation.add(expr=exp(m.flow[22]) - 1 == m.flow[21])
    m.use_unit[8].io_relation.add(expr=exp(m.flow[18]) - 1 == m.flow[10] + m.flow[17])

    m.no_unit[3].bypass = Constraint(expr=m.flow[10] == m.flow[8])

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

    m.Y = Reference(m.use_unit[:].indicator_var)

    # logical propositions
    m.use1or2 = LogicalConstraint(expr=m.Y[1].xor(m.Y[2]))
    m.use1or2implies345 = LogicalConstraint(
        expr=lor(m.Y[1], m.Y[2]).implies(lor(m.Y[3], m.Y[4], m.Y[5]))
    )
    m.use4implies6or7 = LogicalConstraint(expr=m.Y[4].implies(lor(m.Y[6], m.Y[7])))
    m.use3implies8 = LogicalConstraint(expr=m.Y[3].implies(m.Y[8]))
    m.use6or7implies4 = LogicalConstraint(expr=lor(m.Y[6], m.Y[7]).implies(m.Y[4]))
    m.use6or7 = LogicalConstraint(expr=m.Y[6].xor(m.Y[7]))

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


if __name__ == "__main__":
    m = build_eight_process_flowsheet()
    from pyomo.environ import TransformationFactory

    TransformationFactory('core.logical_to_linear').apply_to(m)
    SolverFactory('gdpopt.loa').solve(m, tee=True)
    update_boolean_vars_from_binary(m)
    m.Y.display()
    m.flow.display()
