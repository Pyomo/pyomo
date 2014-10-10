#  _________________________________________________________________________
#
#  Coopr: A COmmon Optimization Python Repository
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the Coopr README.txt file.
#  _________________________________________________________________________

import sys
import logging
from pyutilib.misc import LogHandler

from os.path import abspath, dirname, join, normpath
coopr_base = normpath(join(dirname(abspath(__file__)), '..', '..', '..'))

logger = logging.getLogger('coopr.pyomo')
logger.setLevel( logging.WARNING )
logger.addHandler( LogHandler(coopr_base, verbosity=lambda: logger.isEnabledFor(logging.DEBUG)) )
