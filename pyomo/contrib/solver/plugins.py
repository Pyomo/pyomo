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


from .factory import SolverFactory
from .ipopt import Ipopt
from .gurobi import Gurobi


def load():
    SolverFactory.register(
        name='ipopt', legacy_name='ipopt_v2', doc='The IPOPT NLP solver (new interface)'
    )(Ipopt)
    SolverFactory.register(
        name='gurobi', legacy_name='gurobi_v2', doc='New interface to Gurobi'
    )(Gurobi)
