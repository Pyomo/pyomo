#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyutilib.misc
import pyutilib.services

from pyomo.opt.base import *
from pyomo.opt.results import *
from pyomo.opt.solver import *

@SolverFactory.register('_neos', 'Interface for solvers hosted on NEOS')
class NEOSRemoteSolver(SystemCallSolver):
    """A wrapper class for NEOS Remote Solvers"""

    def __init__(self, **kwds):
        kwds["type"] = "neos"
        SystemCallSolver.__init__(self, **kwds)
        self._valid_problem_formats = [ProblemFormat.nl]
        self._valid_result_formats = {}
        self._valid_result_formats[ProblemFormat.nl] = [ResultsFormat.sol]
        self._problem_format = ProblemFormat.nl
        self._results_format = ResultsFormat.sol

    def create_command_line(self, executable, problem_files):
        """
        Create the local *.sol and *.log files, which will be
        populated by NEOS.
        """
        if self._log_file is None:
           self._log_file = pyutilib.services.TempfileManager.\
                            create_tempfile(suffix=".neos.log")
        if self._soln_file is None:
           self._soln_file = pyutilib.services.TempfileManager.\
                             create_tempfile(suffix=".neos.sol")
           self._results_file = self._soln_file

        # display the log/solver file names prior to execution. this is useful
        # in case something crashes unexpectedly, which is not without precedent.
        if self._keepfiles:
            if self._log_file is not None:
                print("Solver log file: '%s'" % self._log_file)
            if self._soln_file is not None:
                print("Solver solution file: '%s'" % self._soln_file)
            if self._problem_files is not []:
                print("Solver problem files: %s" % str(self._problem_files))

        return pyutilib.misc.Bunch(cmd="", log_file=self._log_file, env="")

    def _default_executable(self):
        return True
