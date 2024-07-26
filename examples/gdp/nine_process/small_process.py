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

"""Small process synthesis-inspired toy GDP example.

"""

from pyomo.core import ConcreteModel, RangeSet, Var, Constraint, Objective
from pyomo.core.expr.current import exp, log, sqrt
from pyomo.gdp import Disjunction


def build_model():
    """
    Base Model

    Optimal solution:
    Select units 1, 3, 8
    Objective value -36.62
    """
    m = ConcreteModel()
    m.streams = RangeSet(25)
    m.x = Var(m.streams, bounds=(0, 50), initialize=5)

    m.stage1_split = Constraint(expr=m.x[1] == m.x[2] + m.x[4])
    m.first_stage = Disjunction(
        expr=[
            [
                # Unit 1
                m.x[2] == exp(m.x[3]) - 1,
                m.x[4] == 0,
                m.x[5] == 0,
            ],
            [
                # Unit 2
                m.x[5] == log(m.x[4] + 1),
                m.x[2] == 0,
                m.x[3] == 0,
            ],
        ]
    )
    m.stage1_mix = Constraint(expr=m.x[3] + m.x[5] == m.x[6])
    m.stage2_split = Constraint(expr=m.x[6] == sum(m.x[i] for i in (7, 9, 11, 13)))
    m.second_stage = Disjunction(
        expr=[
            [
                # Unit 3
                m.x[8] == 2 * log(m.x[7]) + 3,
                m.x[7] >= 0.2,
            ]
            + [m.x[i] == 0 for i in (9, 10, 11, 12, 14, 15)],
            [
                # Unit 4
                m.x[10]
                == 1.8 * log(m.x[9] + 4)
            ]
            + [m.x[i] == 0 for i in (7, 8, 11, 12, 14, 15)],
            [
                # Unit 5
                m.x[12] == 1.2 * log(m.x[11]) + 2,
                m.x[11] >= 0.001,
            ]
            + [m.x[i] == 0 for i in (7, 8, 9, 10, 14, 15)],
            [
                # Unit 6
                m.x[15] == sqrt(m.x[14] - 3) * m.x[23] + 1,
                m.x[14] >= 5,
                m.x[14] <= 20,
            ]
            + [m.x[i] == 0 for i in (7, 8, 9, 10, 11, 12)],
        ]
    )
    m.stage2_special_mix = Constraint(expr=m.x[14] == m.x[13] + m.x[23])
    m.stage2_mix = Constraint(expr=sum(m.x[i] for i in (8, 10, 12, 15)) == m.x[16])
    m.stage3_split = Constraint(expr=m.x[16] == sum(m.x[i] for i in (17, 19, 21)))
    m.third_stage = Disjunction(
        expr=[
            [
                # Unit 7
                m.x[18]
                == m.x[17] * 0.9
            ]
            + [m.x[i] == 0 for i in (19, 20, 21, 22)],
            [
                # Unit 8
                m.x[20] == log(m.x[19] ** 1.5) + 2,
                m.x[19] >= 1,
            ]
            + [m.x[i] == 0 for i in (17, 18, 21, 22)],
            [
                # Unit 9
                m.x[22] == log(m.x[21] + sqrt(m.x[21])) + 1,
                m.x[21] >= 4,
            ]
            + [m.x[i] == 0 for i in (17, 18, 19, 20)],
        ]
    )
    m.stage3_special_split = Constraint(expr=m.x[22] == m.x[23] + m.x[24])
    m.stage3_mix = Constraint(expr=m.x[25] == sum(m.x[i] for i in (18, 20, 24)))

    m.obj = Objective(expr=-10 * m.x[25] + m.x[1])

    return m


def build_nonexclusive_model():
    m = ConcreteModel()
    m.streams = RangeSet(25)
    m.x = Var(m.streams, bounds=(0, 50), initialize=5)

    m.stage1_split = Constraint(expr=m.x[1] == m.x[2] + m.x[4])
    m.unit1 = Disjunction(
        expr=[
            [
                # Unit 1
                m.x[2]
                == exp(m.x[3]) - 1
            ],
            [
                # No Unit 1
                m.x[2] == 0,
                m.x[3] == 0,
            ],
        ]
    )
    m.unit2 = Disjunction(
        expr=[
            [
                # Unit 2
                m.x[5]
                == log(m.x[4] + 1)
            ],
            [
                # No Unit 2
                m.x[4] == 0,
                m.x[5] == 0,
            ],
        ]
    )
    m.stage1_mix = Constraint(expr=m.x[3] + m.x[5] == m.x[6])
    m.stage2_split = Constraint(expr=m.x[6] == sum(m.x[i] for i in (7, 9, 11, 13)))
    m.unit3 = Disjunction(
        expr=[
            [
                # Unit 3
                m.x[8] == 2 * log(m.x[7]) + 3,
                m.x[7] >= 0.2,
            ],
            [
                # No Unit 3
                m.x[7] == 0,
                m.x[8] == 0,
            ],
        ]
    )
    m.unit4 = Disjunction(
        expr=[
            [
                # Unit 4
                m.x[10]
                == 1.8 * log(m.x[9] + 4)
            ],
            [
                # No Unit 4
                m.x[9] == 0,
                m.x[10] == 0,
            ],
        ]
    )
    m.unit5 = Disjunction(
        expr=[
            [
                # Unit 5
                m.x[12] == 1.2 * log(m.x[11]) + 2,
                m.x[11] >= 0.001,
            ],
            [
                # No Unit 5
                m.x[11] == 0,
                m.x[12] == 0,
            ],
        ]
    )
    m.unit6 = Disjunction(
        expr=[
            [
                # Unit 6
                m.x[15] == sqrt(m.x[14] - 3) * m.x[23] + 1,
                m.x[14] >= 5,
                m.x[14] <= 20,
            ],
            [
                # No Unit 6
                m.x[14] == 0,
                m.x[15] == 0,
            ],
        ]
    )
    m.stage2_special_mix = Constraint(expr=m.x[14] == m.x[13] + m.x[23])
    m.stage2_mix = Constraint(expr=sum(m.x[i] for i in (8, 10, 12, 15)) == m.x[16])
    m.stage3_split = Constraint(expr=m.x[16] == sum(m.x[i] for i in (17, 19, 21)))
    m.unit7 = Disjunction(
        expr=[
            [
                # Unit 7
                m.x[18]
                == m.x[17] * 0.9
            ],
            [
                # No Unit 7
                m.x[17] == 0,
                m.x[18] == 0,
            ],
        ]
    )
    m.unit8 = Disjunction(
        expr=[
            [
                # Unit 8
                m.x[20] == log(m.x[19] ** 1.5) + 2,
                m.x[19] >= 1,
            ],
            [
                # No Unit 8
                m.x[19] == 0,
                m.x[20] == 0,
            ],
        ]
    )
    m.unit9 = Disjunction(
        expr=[
            [
                # Unit 9
                m.x[22] == log(m.x[21] + sqrt(m.x[21])) + 1,
                m.x[21] >= 4,
            ],
            [
                # No Unit 9
                m.x[21] == 0,
                m.x[22] == 0,
            ],
        ]
    )
    m.stage3_special_split = Constraint(expr=m.x[22] == m.x[23] + m.x[24])
    m.stage3_mix = Constraint(expr=m.x[25] == sum(m.x[i] for i in (18, 20, 24)))

    m.obj = Objective(expr=-10 * m.x[25] + m.x[1])

    return m


if __name__ == '__main__':
    from pyomo.environ import SolverFactory

    m = build_model()
    result = SolverFactory('gdpopt.gloa').solve(
        m,
        tee=True,
        mip_solver='gams',
        nlp_solver='gams',
        nlp_solver_args=dict(add_options=['option optcr=0.01;']),
        calc_disjunctive_bounds=True,
        obbt_disjunctive_bounds=False,
        iterlim=50,
    )
    print(result)
    m.x.display()

    m = build_nonexclusive_model()
    result = SolverFactory('gdpopt.gloa').solve(
        m,
        tee=True,
        mip_solver='gams',
        nlp_solver='gams',
        nlp_solver_args=dict(add_options=['option optcr=0.01;']),
        calc_disjunctive_bounds=True,
        obbt_disjunctive_bounds=False,
        iterlim=75,
    )
    print(result)
    m.x.display()
