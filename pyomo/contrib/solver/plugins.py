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


from .common.factory import SolverFactory
from .solvers.ipopt import Ipopt, LegacyIpoptSolver
from .solvers.gurobi.gurobi_direct import GurobiDirect
from .solvers.gurobi.gurobi_persistent import GurobiDirectQuadratic, GurobiPersistent
from .solvers.highs import Highs


def load():
    SolverFactory.register(
        name='ipopt', legacy_name='ipopt_v2', doc='The IPOPT NLP solver'
    )(Ipopt, LegacyIpoptSolver)
    SolverFactory.register(
        name='gurobi_persistent',
        legacy_name='gurobi_persistent_v2',
        doc='Persistent interface to Gurobi',
    )(GurobiPersistent)
    SolverFactory.register(
        name='gurobi_direct',
        legacy_name='gurobi_direct_v2',
        doc='Direct (scipy-based) interface to Gurobi',
    )(GurobiDirect)
    SolverFactory.register(
        name='gurobi_direct_quadratic',
        legacy_name='gurobi_direct_quadratic_v2',
        doc='Direct interface to Gurobi',
    )(GurobiDirectQuadratic)
    SolverFactory.register(
        name='highs', legacy_name='highs_v2', doc='Persistent interface to HiGHS'
    )(Highs)
