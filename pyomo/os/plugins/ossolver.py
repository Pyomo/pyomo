#  _________________________________________________________________________
#
#  Coopr: A COmmon Optimization Python Repository
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the Coopr README.txt file.
#  _________________________________________________________________________

from coopr.core.plugin import alias
from pyutilib.misc import Bunch
from pyutilib.services import register_executable, registered_executable
from pyutilib.services import TempfileManager

from coopr.opt.base import ProblemFormat, ResultsFormat
from coopr.opt.solver import SystemCallSolver

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
        #
        # Valid problem formats, and valid results for each format
        #
        self._valid_problem_formats = [ ProblemFormat.osil ]
        self._valid_result_formats  = { ProblemFormat.osil : [ResultsFormat.osrl] }
        self.set_problem_format(ProblemFormat.osil)

    def _default_results_format(self, prob_format):
        return ResultsFormat.osrl

    def available(self, flag=True):
        # TODO: change this when we start working on this solver interface again...
        return False

    def executable(self):
        executable = registered_executable('OSSolverService')
        if executable is None:
            log.error("Could not locate the OSSolverService executable, which is required for solver %s" % self.name)
            self.enable = False
            return None
        return executable.get_path()

    def create_command_line(self, executable, problem_files):
        #
        # Define log file
        #
        if self.log_file is None:
            self.log_file = TempfileManager.create_tempfile(suffix="_os.log")
        fname = problem_files[0]
        self.results_file = problem_files[0]+'.osrl'
        #
        options = []
        for key in self.options:
            if key == 'solver':
                continue
            elif isinstance(self.options[key],basestring) and ' ' in self.options[key]:
                opt.append('-'+key+" \""+str(self.options[key])+"\"")
            elif key == 'subsolver':
                opt.append("-solver "+str(self.options[key]))
            else:
                opt.append('-'+key+" "+str(self.options[key]))
        #
        options = ' '.join( options )
        proc = self._timer + " " + executable + " -osil " + problem_files[0] + " -osrl " + self.results_file + ' ' + options
        return Bunch(cmd=proc, log_file=None, env=None)


register_executable(name='OSSolverService')
