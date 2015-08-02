#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import logging

from pyomo.util.plugin import alias
from pyutilib.misc import Bunch
from pyutilib.services import register_executable, registered_executable
from pyutilib.services import TempfileManager

from pyomo.opt.base import ProblemFormat, ResultsFormat
from pyomo.opt.solver import SystemCallSolver

logger = logging.getLogger('pyomo.os')

create_tempfile = TempfileManager.create_tempfile

class OSSolver(SystemCallSolver):
    """The Optimization Systems solver."""

    alias('os', doc='Interface to an OS solver service (alpha)')

    def __init__ (self, **kwargs):
        #
        # Call base constructor
        #
        kwargs['type'] = 'os'
        SystemCallSolver.__init__( self, **kwargs )
        self._metasolver = True
        #
        # Valid problem formats, and valid results for each format
        #
        self._valid_problem_formats = [ProblemFormat.osil]
        self._valid_result_formats  = {ProblemFormat.osil: [ResultsFormat.osrl]}
        self.set_problem_format(ProblemFormat.osil)

    def _default_results_format(self, prob_format):
        return ResultsFormat.osrl

    def available(self, flag=True):
        # TODO: change this when we start working on this solver
        # interface again...
        return False

    def executable(self):
        executable = registered_executable('OSSolverService')
        if executable is None:
            logger.error("Could not locate the OSSolverService "
                         "executable, which is required for solver %s"
                         % self.name)
            self.enable = False
            return None
        return executable.get_path()

    def create_command_line(self, executable, problem_files):
        #
        # Define log file
        #
        if self._log_file is None:
            self._log_file = TempfileManager.create_tempfile(suffix="_os.log")


        if self._soln_file is not None:
            # the solution file can not be redefined
            # (perhaps it can, but I don't know this interface
            #  the code currently ignores it so warn somebody)
            logger.warn("The 'soln_file' keyword will be ignored "
                        "for solver="+self.type)

        fname = problem_files[0]
        self._results_file = self._soln_file = problem_files[0]+'.osrl'
        #
        options = []
        for key in self.options:
            if key == 'solver':
                continue
            elif isinstance(self.options[key],basestring) and ' ' in self.options[key]:
                options.append('-'+key+" \""+str(self.options[key])+"\"")
            elif key == 'subsolver':
                options.append("-solver "+str(self.options[key]))
            else:
                options.append('-'+key+" "+str(self.options[key]))
        #
        options = ' '.join( options )
        proc = (self._timer + " " + executable +
                " -osil " + problem_files[0] + " -osrl " +
                self.results_file + ' ' + options)
        return Bunch(cmd=proc, log_file=self._log_file, env=None)


register_executable(name='OSSolverService')
