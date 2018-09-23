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
import os
import six

import pyutilib.services
import pyutilib.misc

from pyomo.opt.base.solvers import SolverFactory
from pyomo.opt.base import *
from pyomo.opt.results import *
from pyomo.opt.solver import *
from pyomo.solvers.plugins.solvers.ASL import ASL

logger = logging.getLogger('pyomo.solvers')


@SolverFactory.register('path', doc='Nonlinear MCP solver')
class PATHAMPL(ASL):
    """An interface to the PATH MCP solver."""

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
        executable = pyomo.common.registered_executable("pathampl")
        if executable is None:                      #pragma:nocover
            logger.warning("Could not locate the 'pathampl' executable, which is required for solver %s" % self.name)
            self.enable = False
            return None
        return executable.get_path()

    def create_command_line(self, executable, problem_files):
        self.options.solver = 'pathampl'
        return ASL.create_command_line(self, executable, problem_files)


pyomo.common.register_executable(name="pathampl")
