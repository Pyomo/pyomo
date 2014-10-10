#  _________________________________________________________________________
#
#  Coopr: A COmmon Optimization Python Repository
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the Coopr README.txt file.
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

import coopr.core.plugin
from coopr.opt.base import *
from coopr.opt.base.solvers import _extract_version
from coopr.opt.results import *
from coopr.opt.solver import *
from coopr.pyomo.base.blockutil import has_discrete_variables
from coopr.solvers.mockmip import MockMIP

import logging
logger = logging.getLogger('coopr.solvers')

from six import iteritems
from six.moves import xrange

try:
    unicode
except:
    basestring = unicode = str


class pywrapper(OptSolver):
    """Direct python solver interface
    """

    coopr.core.plugin.alias('py', doc='Direct python solver interfaces')

    def __new__(cls, *args, **kwds):
        mode = kwds.get('solver_io', 'python')
        if mode is None:
            mode = 'python'
        if mode != 'python':
            logging.getLogger('coopr.solvers').error("Cannot specify IO mode '%s' for direct python solver interface" % mode)
            return None
        #
        if not 'solver' in kwds:
            logging.getLogger('coopr.solvers').warning("No solver specified for direct python solver interface")
            return None
        kwds['solver_io'] = 'python'
        solver = kwds['solver']
        del kwds['solver']
        return SolverFactory(solver, **kwds)

