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

from pyomo.common.extensions import ExtensionBuilderFactory
from .base import SolverFactory
from .solvers import Gurobi, Ipopt, Cbc, Cplex, Highs, MAiNGO
from .build import AppsiBuilder


def load():
    ExtensionBuilderFactory.register('appsi')(AppsiBuilder)
    SolverFactory.register(
        name='gurobi', doc='Automated persistent interface to Gurobi'
    )(Gurobi)
    SolverFactory.register(name='cplex', doc='Automated persistent interface to Cplex')(
        Cplex
    )
    SolverFactory.register(name='ipopt', doc='Automated persistent interface to Ipopt')(
        Ipopt
    )
    SolverFactory.register(name='cbc', doc='Automated persistent interface to Cbc')(Cbc)
    SolverFactory.register(name='highs', doc='Automated persistent interface to Highs')(
        Highs
    )
    SolverFactory.register(
        name='maingo', doc='Automated persistent interface to MAiNGO'
    )(MAiNGO)
