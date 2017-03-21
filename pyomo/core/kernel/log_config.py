#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import logging
from os.path import abspath, dirname, join, normpath
pyomo_base = normpath(join(dirname(abspath(__file__)), '..', '..', '..'))

from pyutilib.misc import LogHandler

logger = logging.getLogger('pyomo.core')
logger.setLevel( logging.WARNING )
logger.addHandler( LogHandler(pyomo_base, verbosity=lambda: logger.isEnabledFor(logging.DEBUG)) )
