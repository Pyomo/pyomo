#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the Pyomo README.txt file.
#  _________________________________________________________________________


import os
import re
import re
import xml.dom.minidom
import time
import logging

import pyutilib.services
import pyutilib.common
import pyutilib.misc

import pyomo.util.plugin
from pyomo.opt.base import *
from pyomo.opt.base.solvers import _extract_version
from pyomo.opt.results import *
from pyomo.opt.solver import *
from pyomo.core.base.blockutil import has_discrete_variables
from pyomo.solvers.mockmip import MockMIP

import logging
logger = logging.getLogger('pyomo.solvers')

from six import iteritems
from six.moves import xrange

try:
    unicode
except:
    basestring = unicode = str


class pywrapper(OptSolver):
    """Direct python solver interface
    """

    pyomo.util.plugin.alias('py', doc='Direct python solver interfaces')

    def __new__(cls, *args, **kwds):
        mode = kwds.get('solver_io', 'python')
        if mode is None:
            mode = 'python'
        if mode != 'python':
            logging.getLogger('pyomo.solvers').error("Cannot specify IO mode '%s' for direct python solver interface" % mode)
            return None
        #
        if not 'solver' in kwds:
            logging.getLogger('pyomo.solvers').warning("No solver specified for direct python solver interface")
            return None
        kwds['solver_io'] = 'python'
        solver = kwds['solver']
        del kwds['solver']
        return SolverFactory(solver, **kwds)

