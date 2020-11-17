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

from pyomo.opt.base.solvers import SolverFactory
from pyomo.common import Executable
from pyomo.common.collections import Options
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
        self._capabilities = Options()
        self._capabilities.linear = True

    def _default_executable(self):
        executable = Executable("pathampl")
        if not executable:                      #pragma:nocover
            logger.warning("Could not locate the 'pathampl' executable, "
                           "which is required for solver %s" % self.name)
            self.enable = False
            return None
        return executable.path()

    def create_command_line(self, executable, problem_files):
        self.options.solver = 'pathampl'
        return ASL.create_command_line(self, executable, problem_files)
