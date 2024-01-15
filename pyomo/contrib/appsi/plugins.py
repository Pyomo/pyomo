from pyomo.common.extensions import ExtensionBuilderFactory
from .base import SolverFactory
from .solvers import Gurobi, Ipopt, Cbc, Cplex, Highs
from .build import AppsiBuilder


def load():
    ExtensionBuilderFactory.register('appsi')(AppsiBuilder)
    SolverFactory.register(
        name='appsi_gurobi', doc='Automated persistent interface to Gurobi'
    )(Gurobi)
    SolverFactory.register(
        name='appsi_cplex', doc='Automated persistent interface to Cplex'
    )(Cplex)
    SolverFactory.register(
        name='appsi_ipopt', doc='Automated persistent interface to Ipopt'
    )(Ipopt)
    SolverFactory.register(
        name='appsi_cbc', doc='Automated persistent interface to Cbc'
    )(Cbc)
    SolverFactory.register(
        name='appsi_highs', doc='Automated persistent interface to Highs'
    )(Highs)
