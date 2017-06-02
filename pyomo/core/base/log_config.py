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
from os.path import abspath, dirname, join, normpath
pyomo_base = normpath(join(dirname(abspath(__file__)), '..', '..', '..'))

from pyutilib.misc import LogHandler

logger = logging.getLogger('pyomo.core')
logger.setLevel( logging.WARNING )
logger.addHandler( LogHandler(pyomo_base, verbosity=lambda: logger.isEnabledFor(logging.DEBUG)) )
