#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import logging
from pyutilib.misc import LogHandler

from os.path import abspath, dirname
pyomo_base = dirname(dirname(dirname(dirname(dirname(abspath(__file__))))))

logger = logging.getLogger('pyomo.opt')
logger.addHandler( LogHandler(pyomo_base) )
logger.setLevel( logging.WARNING )

class NullHandler(logging.Handler):
    def emit(self, record):         #pragma:nocover
        """Do not generate logging record"""

# TODO: move this into a pyomo.common project
logger = logging.getLogger('pyomo')
logger.addHandler( NullHandler() )
logger.setLevel( logging.WARNING )

logger = logging.getLogger('pyomo.solvers')
logger.addHandler( LogHandler(pyomo_base) )
logger.setLevel( logging.WARNING )
