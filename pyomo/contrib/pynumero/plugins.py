#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.extensions import ExtensionBuilderFactory
from pyomo.opt import SolverFactory
from .build import PyNumeroBuilder
from .algorithms.solvers.cyipopt_solver import PyomoCyIpoptSolver

def load():
    ExtensionBuilderFactory.register('pynumero')(PyNumeroBuilder)
    SolverFactory.register(
        'cyipopt',
        doc='Cyipopt: direct python bindings to the Ipopt NLP solver'
    )(PyomoCyIpoptSolver)
