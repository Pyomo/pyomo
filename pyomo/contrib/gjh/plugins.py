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

from pyomo.common.download import DownloadFactory
from pyomo.contrib.gjh.getGJH import get_gjh
from pyomo.contrib.gjh.GJH import GJHSolver
from pyomo.opt.base import SolverFactory


def load():
    DownloadFactory.register('gjh')(get_gjh)
    SolverFactory.register('contrib.gjh', doc='Interface to the AMPL GJH "solver"')(
        GJHSolver
    )
