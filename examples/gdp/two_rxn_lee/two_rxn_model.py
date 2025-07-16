#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Two reactor model from literature. See README.md."""

import pyomo.environ as pyo
from pyomo.gdp import Disjunction


def build_model(use_mccormick=False):
    """Build the GDP model."""
    m = pyo.ConcreteModel()
    m.F = pyo.Var(bounds=(0, 8), doc="Flow into reactor")
    m.X = pyo.Var(bounds=(0, 1), doc="Reactor conversion")
    m.d = pyo.Param(initialize=2, doc="Max product demand")
    m.c = pyo.Param(
        [1, 2, 'I', 'II'],
        doc="Costs",
        initialize={
            1: 2,  # Value of product
            2: 0.2,  # Cost of raw material
            'I': 2.5,  # Cost of reactor I
            'II': 1.5,  # Cost of reactor II
        },
    )
    m.alpha = pyo.Param(
        ['I', 'II'], doc="Reactor coefficient", initialize={'I': -8, 'II': -10}
    )
    m.beta = pyo.Param(
        ['I', 'II'], doc="Reactor coefficient", initialize={'I': 9, 'II': 15}
    )
    m.X_LB = pyo.Param(
        ['I', 'II'],
        doc="Reactor conversion lower bound",
        initialize={'I': 0.2, 'II': 0.7},
    )
    m.X_UB = pyo.Param(
        ['I', 'II'],
        doc="Reactor conversion upper bound",
        initialize={'I': 0.95, 'II': 0.99},
    )
    m.C_rxn = pyo.Var(bounds=(1.5, 2.5), doc="Cost of reactor")
    m.reactor_choice = Disjunction(
        expr=[
            # Disjunct 1: Choose reactor I
            [
                m.F == m.alpha['I'] * m.X + m.beta['I'],
                m.X_LB['I'] <= m.X,
                m.X <= m.X_UB['I'],
                m.C_rxn == m.c['I'],
            ],
            # Disjunct 2: Choose reactor II
            [
                m.F == m.alpha['II'] * m.X + m.beta['II'],
                m.X_LB['II'] <= m.X,
                m.X <= m.X_UB['II'],
                m.C_rxn == m.c['II'],
            ],
        ],
        xor=True,
    )
    if use_mccormick:
        m.P = pyo.Var(bounds=(0, 8), doc="McCormick approximation of F*X")
        m.mccormick_1 = pyo.Constraint(
            expr=m.P <= m.F.lb * m.X + m.F * m.X.ub - m.F.lb * m.X.ub,
            doc="McCormick overestimator",
        )
        m.mccormick_2 = pyo.Constraint(
            expr=m.P <= m.F.ub * m.X + m.F * m.X.lb - m.F.ub * m.X.lb,
            doc="McCormick underestimator",
        )
        m.max_demand = pyo.Constraint(expr=m.P <= m.d, doc="product demand")
        m.profit = pyo.Objective(
            expr=m.c[1] * m.P - m.c[2] * m.F - m.C_rxn, sense=pyo.maximize
        )
    else:
        m.max_demand = pyo.Constraint(expr=m.F * m.X <= m.d, doc="product demand")
        m.profit = pyo.Objective(
            expr=m.c[1] * m.F * m.X - m.c[2] * m.F - m.C_rxn, sense=pyo.maximize
        )

    return m
