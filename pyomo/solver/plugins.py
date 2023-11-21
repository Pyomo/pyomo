#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


from .factory import SolverFactory
from .IPOPT import IPOPT


def load():
    SolverFactory.register(name='ipopt_v2', doc='The IPOPT NLP solver (new interface)')(
        IPOPT
    )
