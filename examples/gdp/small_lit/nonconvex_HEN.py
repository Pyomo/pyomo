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

""" Example from 'Systematic Modeling of Discrete-Continuous Optimization \
      Models through Generalized Disjunctive Programming'
    Ignacio E. Grossmann and Francisco Trespalacios, 2013

    Optimal solution ca. 114,385

    Pyomo model implementation by @RomeoV
"""

from pyomo.environ import (
    ConcreteModel,
    Constraint,
    NonNegativeReals,
    Objective,
    RangeSet,
    SolverFactory,
    TransformationFactory,
    Var,
)


def build_gdp_model():
    # PARAMETERS
    T1_lo, T1_up = 350.0, 400.0
    T2_lo, T2_up = 450.0, 500.0

    U = {'1': 1.5, '2': 0.5, '3': 1}
    FCP = {'hot': 10.0, 'cold': 7.5}
    T_in = {'hot': 500.0, 'cold': 350.0, 'cooling': 300.0, 'steam': 600.0}
    T_out = {'hot': 340.0, 'cold': 560.0, 'cooling': 320.0, 'steam': 600.0}
    Cost = {'cooling': 20.0, 'steam': 80.0}

    # VARIABLES
    m = ConcreteModel()
    m.T1 = Var(domain=NonNegativeReals, bounds=(T1_lo, T1_up))
    m.T2 = Var(domain=NonNegativeReals, bounds=(T2_lo, T2_up))

    m.exchangers = RangeSet(1, 3)
    m.A = Var(m.exchangers, domain=NonNegativeReals, bounds=(1e-4, 50))
    m.CP = Var(
        m.exchangers, domain=NonNegativeReals, bounds=(0, 600 * (50**0.6) + 2 * 46500)
    )
    # Note that A_lo=0 leads to an exception in MC++ if using gdpopt with strategy 'GLOA'
    # The exception occurs when constructing McCormick relaxations

    # OBJECTIVE
    m.objective = Objective(
        expr=(
            sum(m.CP[i] for i in m.exchangers)
            + FCP['hot'] * (m.T1 - T_out['hot']) * Cost['cooling']
            + FCP['cold'] * (T_out['cold'] - m.T2) * Cost['steam']
        )
    )

    # GLOBAL CONSTRAINTS
    m.constr1 = Constraint(
        expr=FCP['hot'] * (T_in['hot'] - m.T1)
        == m.A[1] * U['1'] * ((T_in['hot'] - m.T2) + (m.T1 - T_in['cold'])) / 2.0
    )
    m.constr2 = Constraint(  # Note the error in the paper in constraint 2
        expr=FCP['hot'] * (m.T1 - T_out['hot'])
        == m.A[2]
        * U['2']
        * ((T_out['hot'] - T_in['cooling']) + (m.T1 - T_out['cooling']))
        / 2.0
    )
    m.constr3 = Constraint(
        expr=FCP['cold'] * (T_out['cold'] - m.T2)
        == m.A[3]
        * U['3']
        * ((T_out['steam'] - m.T2) + (T_in['steam'] - T_out['cold']))
        / 2.0
    )
    m.constr4 = Constraint(
        expr=FCP['hot'] * (T_in['hot'] - m.T1) == FCP['cold'] * (m.T2 - T_in['cold'])
    )

    # DISJUNCTIONS
    @m.Disjunction(m.exchangers)
    def exchanger_disjunction(m, disjctn):
        return [
            [
                m.CP[disjctn] == 2750 * (m.A[disjctn] ** 0.6) + 3000,
                0.0 <= m.A[disjctn],
                m.A[disjctn] <= 10.0,
            ],
            [
                m.CP[disjctn] == 1500 * (m.A[disjctn] ** 0.6) + 15000,
                10.0 <= m.A[disjctn],
                m.A[disjctn] <= 25.0,
            ],
            [
                m.CP[disjctn] == 600 * (m.A[disjctn] ** 0.6) + 46500,
                25.0 <= m.A[disjctn],
                m.A[disjctn] <= 50.0,
            ],
        ]

    return m


if __name__ == "__main__":
    # Decide whether to reformulate as MINLP and what method to use
    reformulation = True
    reformulation_method = 'hull'

    model = build_gdp_model()
    model.pprint()

    if reformulation:
        if reformulation_method == 'bigm':
            TransformationFactory('gdp.bigm').apply_to(
                model, bigM=600 * (50**0.6) + 2 * 46500
            )
        elif reformulation_method == 'hull':
            TransformationFactory('gdp.hull').apply_to(model)
        res = SolverFactory('gams').solve(
            model,
            tee=True,
            solver='baron',
            add_options=['option optcr = 0;'],
            keepfiles=True,
        )
    else:
        # Note: MC++ needs to be properly installed to use strategy GLOA
        res = SolverFactory('gdpopt.gloa').solve(model, tee=True)

    # model.display()
    print(res)
