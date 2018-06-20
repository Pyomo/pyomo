"""Two reactor model from literature. See README.md."""
from __future__ import division

from pyomo.core import (ConcreteModel, Constraint, Objective, Param, Var,
                        maximize)
# from pyomo.environ import *  # NOQA
from pyomo.gdp import Disjunction


def build_model():
    """Build the GDP model."""
    m = ConcreteModel()
    m.F = Var(bounds=(0, 8), doc="Flow into reactor")
    m.X = Var(bounds=(0, 1), doc="Reactor conversion")
    m.d = Param(initialize=2, doc="Max product demand")
    m.c = Param([1, 2, 'I', 'II'], doc="Costs", initialize={
        1: 2,  # Value of product
        2: 0.2,  # Cost of raw material
        'I': 2.5,  # Cost of reactor I
        'II': 1.5  # Cost of reactor II
    })
    m.alpha = Param(['I', 'II'], doc="Reactor coefficient",
                    initialize={'I': -8, 'II': -10})
    m.beta = Param(['I', 'II'], doc="Reactor coefficient",
                   initialize={'I': 9, 'II': 15})
    m.X_LB = Param(['I', 'II'], doc="Reactor conversion lower bound",
                   initialize={'I': 0.2, 'II': 0.7})
    m.X_UB = Param(['I', 'II'], doc="Reactor conversion upper bound",
                   initialize={'I': 0.95, 'II': 0.99})
    m.C_rxn = Var(bounds=(1.5, 2.5), doc="Cost of reactor")
    m.max_demand = Constraint(expr=m.F * m.X <= m.d, doc="product demand")
    m.reactor_choice = Disjunction(expr=[
        # Disjunct 1: Choose reactor I
        [m.F == m.alpha['I'] * m.X + m.beta['I'],
         m.X_LB['I'] <= m.X,
         m.X <= m.X_UB['I'],
         m.C_rxn == m.c['I']],
        # Disjunct 2: Choose reactor II
        [m.F == m.alpha['II'] * m.X + m.beta['II'],
         m.X_LB['II'] <= m.X,
         m.X <= m.X_UB['II'],
         m.C_rxn == m.c['II']]
    ], xor=True)
    m.profit = Objective(
        expr=m.c[1] * m.F * m.X - m.c[2] * m.F - m.C_rxn, sense=maximize)

    return m
