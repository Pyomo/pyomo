#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import logging

from pyomo.common.download import DownloadFactory
from pyomo.opt.base import SolverFactory
from pyomo.contrib.trustregion.getGJH import get_gjh
from pyomo.contrib.trustregion.TRF import TrustRegionSolver

logger = logging.getLogger('pyomo.contrib.trustregion')

def load():
    DownloadFactory.register('gjh')(get_gjh)
    SolverFactory.register(
        name='trustregion',
        doc='Trust region filter method for black box/glass box optimization.'
    )(TrustRegionSolver)

