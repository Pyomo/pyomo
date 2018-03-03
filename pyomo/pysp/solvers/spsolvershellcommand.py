#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ("SPSolverShellCommand",)

import os
import logging
import pprint

import pyutilib.misc

from pyomo.pysp.solvers.spsolver import SPSolver

logger = logging.getLogger('pyomo.pysp')

class SPSolverShellCommand(SPSolver):

    def __init__(self, *args, **kwds):
        super(SPSolverShellCommand, self).__init__(*args, **kwds)
        self._executable = None
        self._files = {}

    def _create_tempdir(self, label, *args, **kwds):
        dirname = pyutilib.services.\
                  TempfileManager.create_tempdir(*args, **kwds)
        self._files[label] = dirname
        return dirname

    def _create_tempfile(self, label, *args, **kwds):
        filename = pyutilib.services.\
                   TempfileManager.create_tempfile(*args, **kwds)
        self._files[label] = filename
        return filename

    def _add_tempfile(self, label, filename):
        pyutilib.services.\
            TempfileManager.add_tempfile(filename,
                                         exists=False)
        self._files[label] = filename

    @property
    def executable(self):
        """The executable used by this solver."""
        return self._executable

    @property
    def files(self):
        """A dictionary maintaining the location of various
        solvers files generated during the most recent
        solve. All files will be removed before a solve
        completes unless the keep_solver_files keyword was
        set to True."""
        return self._files

    def set_executable(self, name, validate=True):
        """
        Set the executable for this solver.

        Args:
            name (str): A relative, absolute, or base
                executable name.
            validate (bool): When set to True (default)
                extra checks take place that ensure an
                executable file with that name exists, and
                then 'name' is converted to an absolute
                path. On Windows platforms, a '.exe'
                extension will be appended if necessary when
                validating 'name'. If a file named 'name'
                does not appear to be a relative or absolute
                path, the search will be performed within
                the directories assigned to the PATH
                environment variable.
        """
        if not validate:
            self._executable = name
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
            self._executable = exe

    def available(self):
        """Returns whether this solver is available by checking
        if the currently assigned executable exists."""
        exe = self._executable
        try:
            self.set_executable(exe, validate=True)
        except ValueError:
            return False
        else:
            return True
        finally:
            self._executable = exe

    def solve(self, sp, *args, **kwds):
        """
        Solve a stochastic program.

        See the 'solve' method on the base class for
        additional keyword documentation.

        Args:
            sp: The stochastic program to solve.
            keep_solver_files (bool): Retain temporary solver
                input and output files after the solve completes.
            *args: Passed to the derived solver class
                (see the _solve_impl method).
            **kwds: Passed to the derived solver class
                (see the _solve_impl method).

        Returns: A results object with information about the solution.
        """

        self._files.clear()
        assert self.executable is not None

        keep_solver_files = kwds.pop("keep_solver_files", False)
        pyutilib.services.TempfileManager.push()
        try:
            return super(SPSolverShellCommand, self).solve(sp, *args, **kwds)
        finally:
            # cleanup
            pyutilib.services.TempfileManager.pop(
                remove=not keep_solver_files)
            if keep_solver_files:
                logger.info("Retaining the following solver files:\n"
                            +pprint.pformat(self.files))
