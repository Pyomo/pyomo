#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ['DeveloperError']


import logging
from os.path import abspath, dirname, join, normpath
pyomo_base = normpath(join(dirname(abspath(__file__)), '..', '..', '..'))

from pyutilib.misc import LogHandler

logger = logging.getLogger('pyomo.util')
logger.setLevel( logging.WARNING )
logger.addHandler( LogHandler(pyomo_base, verbosity=lambda: logger.isEnabledFor(logging.DEBUG)) )


class DeveloperError(NotImplementedError):
    """
    Exception class used to throw errors that result from Pyomo
    programming errors, rather than user modeling errors (e.g., a
    component not declaring a 'ctype').
    """

    def __init__(self, val):
        self.parameter = val

    def __str__(self):
        return ( "Internal Pyomo implementation error:\n\t%s\n"
                 "\tPlease report this to the Pyomo Developers."
                 % ( repr(self.parameter), ) )


