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
from pyutilib.misc import LogHandler

from os.path import abspath, dirname
pyomo_base = dirname(dirname(dirname(dirname(abspath(__file__)))))

logger = logging.getLogger('pyomo.pysp')
logger.addHandler( LogHandler(pyomo_base) )
logger.setLevel( logging.WARNING )
