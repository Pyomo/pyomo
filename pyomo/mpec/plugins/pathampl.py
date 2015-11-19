#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import logging
import os
import six

import pyutilib.services
import pyutilib.misc

import pyomo.util.plugin
from pyomo.opt.base import *
from pyomo.opt.results import *
from pyomo.opt.solver import *
from pyomo.solvers.plugins.solvers.ASL import ASL

logger = logging.getLogger('pyomo.solvers')


class PATHAMPL(ASL):
    """An interface to the PATH MCP solver."""

    pyomo.util.plugin.alias('path', doc='Nonlinear MCP solver')

    def __init__(self, **kwds):
        #
        # Call base constructor
        #
        kwds["type"] = "path"
        ASL.__init__(self, **kwds)
        self._metasolver = False
        #
        # Define solver capabilities, which default to 'None'
        #
        self._capabilities = pyutilib.misc.Options()
        self._capabilities.linear = True

    def _default_executable(self):
        executable = pyutilib.services.registered_executable("pathampl")
        if executable is None:                      #pragma:nocover
            logger.warning("Could not locate the 'pathampl' executable, which is required for solver %s" % self.name)
            self.enable = False
            return None
        return executable.get_path()

    def create_command_line(self, executable, problem_files):
        self.options.solver = 'pathampl'
        return ASL.create_command_line(self, executable, problem_files)


pyutilib.services.register_executable(name="pathampl")
