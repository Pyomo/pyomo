#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = ['SystemCallSolver']

import os
import sys
import time
import logging

import pyutilib.misc
from pyutilib.common import ApplicationError, WindowsError
from pyutilib.misc import Bunch
from pyutilib.services import registered_executable, TempfileManager
from pyutilib.subprocess import run

from pyomo.opt.base import *
from pyomo.opt.base.solvers import *
from pyomo.opt.results import SolverStatus, SolverResults

logger = logging.getLogger('pyomo.opt')


class SystemCallSolver(OptSolver):
    """ A generic command line solver """

    def __init__(self, **kwargs):
        """ Constructor """

        executable = kwargs.pop('executable', None)
        validate = kwargs.pop('validate', True)

        OptSolver.__init__(self, **kwargs)
        self._keepfiles  = False
        self._results_file = None
        self._timer      = ''
        self._user_executable = None
        # broadly useful for reporting, and in cases where
        # a solver plugin may not report execution time.
        self._last_solve_time = None

        if executable is not None:
            self.set_executable(name=executable, validate=validate)

    def set_executable(self, name=None, validate=True):
        """
        Set the executable for this solver.

        The 'name' keyword can be assigned a relative,
        absolute, or base filename. If it is unset (None),
        the executable will be reset to the default value
        associated with the solver interface.

        When 'validate' is True (default) extra checks take
        place that ensure an executable file with that name
        exists, and then 'name' is converted to an absolute
        path. On Windows platforms, a '.exe' extension will
        be appended if necessary when validating 'name'. If
        a file named 'name' does not appear to be a relative
        or absolute path, the search will be performed
        within the directories assigned to the PATH
        environment variable.
        """
        if name is None:
            self._user_executable = None
            if validate:
                if self._default_executable() is None:
                    raise ValueError(
                        "Failed to set executable for solver %s to "
                        "its default value. No available solver "
                        "executable was found." % (self.name))
            return

        if not validate:
            self._user_executable = name
        else:
            name = os.path.expanduser(name)
            if os.path.isabs(name):
                exe = pyutilib.misc.search_file(name,
                                                executable=True,
                                                search_path=[''])
            elif os.path.basename(name) != name:
                exe = pyutilib.misc.search_file(os.path.relpath(name),
                                                executable=True,
                                                search_path=[os.path.curdir])
            else:
                # Only search directories in the PATH if
                # name is not in the form of an absolute or
                # relative path.  E.g., it would be
                # confusing if someone called
                # set_executable('./foo') and forgot to copy
                # 'foo' into the local directory, but this
                # function picked up another 'foo' in the
                # users PATH that they did not want to use.
                exe = pyutilib.misc.search_file(name,
                                                executable=True)
            if exe is None:
                raise ValueError(
                    "Failed to set executable for solver %s. File "
                    "with name=%s either does not exist or it is "
                    "not executable. To skip this validation, "
                    "call set_executable with validate=False."
                    % (self.name, name))
            self._user_executable = exe

    def available(self, exception_flag=False):
        """ True if the solver is available """
        if self._assert_available:
            return True
        if not OptSolver.available(self,exception_flag):
            return False
        try:
            ans = self.executable()
        except NotImplementedError:
            ans = None
        if ans is None:
            if exception_flag:
                msg = "No executable found for solver '%s'"
                raise ApplicationError(msg % self.name)
            return False
        return True

    def create_command_line(self,executable,problem_files):
        """
        Create the command line that is executed.
        """
        raise NotImplementedError       #pragma:nocover

    def process_logfile(self):
        """
        Process the logfile for information about the optimization process.
        """
        return SolverResults()

    def process_soln_file(self,results):
        """
        Process auxilliary data files generated by the optimizer (e.g. solution
        files)
        """
        return results

    #
    # NOTE: As JDS has suggested, there could be some value
    #       to allowing the user to change the search path
    #       used to find the default solver executable name
    #       provided by the derived class implementation
    #       (unlike set_executable() which requires the
    #       executable name). This would allow users to
    #       avoid having to know that, for examples,
    #       gurobi.sh is the executable name for the LP
    #       interface and gurobi_ampl is the executable for
    #       the NL interface, while still being able to
    #       switch to a non-default location.
    #
    #       It seems possible that this functionality could
    #       be implemented here on the base class by simply
    #       adding an optional search_path keyword to the
    #       _default_executable method implemented by
    #       derived classes. How to propagate that through
    #       the pyomo.common.registered_executable
    #       framework once it gets there is another question
    #       (that I won't be dealing with today).
    #
    #      # E.g.,
    #      def set_search_path(self, path):
    #         self._search_path = path
    #
    #      where executable would call
    #      self._default_executable(self._search_path)
    #

    def executable(self):
        """
        Returns the executable used by this solver.
        """
        return self._user_executable if (self._user_executable is not None) else \
            self._default_executable()

    def _default_executable(self):
        """
        Returns the default executable used by this solver.
        """
        raise NotImplementedError

    def _presolve(self, *args, **kwds):
        """
        Peform presolves.
        """
        TempfileManager.push()

        self._keepfiles = kwds.pop("keepfiles", False)

        OptSolver._presolve(self, *args, **kwds)

        #
        # Verify that the input problems exists
        #
        for filename in self._problem_files:
            if not os.path.exists(filename):
                msg = 'Solver failed to locate input problem file: %s'
                raise ValueError(msg % filename)
        #
        # Create command line
        #
        self._command = self.create_command_line(
            self.executable(), self._problem_files)

        self._log_file=self._command.log_file
        #
        # The pre-cleanup is probably unncessary, but also not harmful.
        #
        if (self._log_file is not None) and \
           os.path.exists(self._log_file):
            os.remove(self._log_file)
        if (self._soln_file is not None) and \
           os.path.exists(self._soln_file):
            os.remove(self._soln_file)

    def _apply_solver(self):
        if registered_executable('timer'):
            self._timer = registered_executable('timer').get_path()
        #
        # Execute the command
        #
        if __debug__ and logger.isEnabledFor(logging.DEBUG):
            logger.debug("Running %s", self._command.cmd)

        # display the log/solver file names prior to execution. this is useful
        # in case something crashes unexpectedly, which is not without precedent.
        if self._keepfiles:
            if self._log_file is not None:
                print("Solver log file: '%s'" % self._log_file)
            if self._soln_file is not None:
                print("Solver solution file: '%s'" % self._soln_file)
            if self._problem_files is not []:
                print("Solver problem files: %s" % str(self._problem_files))

        sys.stdout.flush()
        self._rc, self._log = self._execute_command(self._command)
        sys.stdout.flush()
        return Bunch(rc=self._rc, log=self._log)

    def _postsolve(self):

        if self._log_file is not None:
            OUTPUT=open(self._log_file,"w")
            OUTPUT.write("Solver command line: "+str(self._command.cmd)+'\n')
            OUTPUT.write("\n")
            OUTPUT.write(self._log+'\n')
            OUTPUT.close()

        # JPW: The cleanup of the problem file probably shouldn't be here, but
        #   rather in the base OptSolver class. That would require movement of
        #   the keepfiles attribute and associated cleanup logic to the base
        #   class, which I didn't feel like doing at this present time. the
        #   base class remove_files method should clean up the problem file.

        if (self._log_file is not None) and \
           (not os.path.exists(self._log_file)):
            msg = "File '%s' not generated while executing %s"
            raise IOError(msg % (self._log_file, self.path))
        results = None

        if self._results_format is not None:
            results = self.process_output(self._rc)
            #
            # If keepfiles is true, then we pop the
            # TempfileManager context while telling it to
            # _not_ remove the files.
            #
            if not self._keepfiles:
                # in some cases, the solution filename is
                # not generated via the temp-file mechanism,
                # instead being automatically derived from
                # the input lp/nl filename. so, we may have
                # to clean it up manually.
                if (not self._soln_file is None) and \
                   os.path.exists(self._soln_file):
                    os.remove(self._soln_file)

        TempfileManager.pop(remove=not self._keepfiles)

        return results

    def _execute_command(self,command):
        """
        Execute the command
        """

        start_time = time.time()

        try:
            if 'script' in command:
                _input = command.script
            else:
                _input = None
            [rc, log] = run(
                command.cmd,
                stdin = _input,
                timelimit = self._timelimit if self._timelimit is None else self._timelimit + max(1, 0.01*self._timelimit),
                env   = command.env,
                tee   = self._tee
             )
        except WindowsError:
            err = sys.exc_info()[1]
            msg = 'Could not execute the command: %s\tError message: %s'
            raise ApplicationError(msg % (command.cmd, err))
        sys.stdout.flush()

        self._last_solve_time = time.time() - start_time

        return [rc,log]

    def process_output(self, rc):
        """
        Process the output files.
        """
        start_time = time.time()
        if self._results_format is None:
            raise ValueError("Results format is None")
        results = self.process_logfile()
        log_file_completion_time = time.time()
        if self._report_timing is True:
            print("      %6.2f seconds required to read logfile " % (log_file_completion_time - start_time))
        if self._results_reader is None:
            self.process_soln_file(results)
            soln_file_completion_time = time.time()
            if self._report_timing is True:
                print("      %6.2f seconds required to read solution file " % (soln_file_completion_time - log_file_completion_time))
        else:
            # There is some ambiguity here as to where the solution data
            # It's natural to expect that the log file contains solution
            # information, but perhaps also in a results file.
            # For now, if there is a single solution, then we assume that
            # the results file is going to add more data to it.
            if len(results.solution) == 1:
                results = self._results_reader(self._results_file,
                                               res=results,
                                               soln=results.solution(0),
                                               suffixes=self._suffixes)
            else:
                results = self._results_reader(self._results_file,
                                               res=results,
                                               suffixes=self._suffixes)
            results_reader_completion_time = time.time()
            if self._report_timing is True:
                print("      %6.2f seconds required to read solution file" % (results_reader_completion_time - log_file_completion_time))
        if rc != None:
            results.solver.error_rc=rc
            if rc != 0:
                results.solver.status=SolverStatus.error

        if self._last_solve_time != None:
            results.solver.time=self._last_solve_time

        return results

    def _default_results_format(self, prob_format):
        """ Returns the default results format for different problem
            formats.
        """
        return ResultsFormat.soln
