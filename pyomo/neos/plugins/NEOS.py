#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import os
import re

import pyutilib.misc
import pyutilib.services
import pyomo.util.plugin

from pyomo.opt.base import *
from pyomo.opt.results import *
from pyomo.opt.solver import *


class NEOSRemoteSolver(SystemCallSolver):
    """A wrapper class for NEOS Remote Solvers"""

    pyomo.util.plugin.alias('_neos', 'Interface for solvers hosted on NEOS')

    def __init__(self, **kwds):
        kwds["type"] = "neos"
        SystemCallSolver.__init__(self, **kwds)
        self._valid_problem_formats=[ProblemFormat.nl]
        self._valid_result_formats = {}
        self._valid_result_formats[ProblemFormat.nl] = [ResultsFormat.sol]
        self._problem_format = ProblemFormat.nl
        self._results_format = ResultsFormat.sol
        self.solver_options=""
        self.tmp_opt=SolverFactory('asl')

    def create_command_line(self,executable,problem_files):
        """
        Create the local *.sol and *.log files, which will be
        populated by NEOS.
        """
        if self.log_file is None:
           self.log_file = pyutilib.services.TempfileManager.create_tempfile(suffix=".neos.log")
        if self.soln_file is None:
           self.soln_file = pyutilib.services.TempfileManager.create_tempfile(suffix=".neos.sol")
           self.results_file = self.soln_file           
        return pyutilib.misc.Bunch(cmd="", log_file=self.log_file, env="")

    def process_logfile(self):
        """
        This function creates and returns a SolverResults object that
        is populated with data that is parsed from the solver output
        provided by NEOS.
        """
        self.tmp_opt.soln_file = self.soln_file
        self.tmp_opt.log_file = self.log_file
        self.tmp_opt.results_file = self.results_file
        return self.tmp_opt.process_logfile()

    #def process_other_data(self,results):
    #    """
    #    Process the *.sol file.
    #    """
    #    self.tmp_opt.process_other_data(results)

    def executable(self):
        return True

