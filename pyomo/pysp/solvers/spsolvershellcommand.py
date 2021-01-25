#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ("SPSolverShellCommand",)

import logging
import pprint

import pyomo.common
from pyomo.common.tempfiles import TempfileManager
from pyomo.pysp.solvers.spsolver import SPSolver

logger = logging.getLogger('pyomo.pysp')

class SPSolverShellCommand(SPSolver):

    def __init__(self, *args, **kwds):
        super(SPSolverShellCommand, self).__init__(*args, **kwds)
        self._executable = None
        self._files = {}

    def _create_tempdir(self, label, *args, **kwds):
        dirname = TempfileManager.create_tempdir(*args, **kwds)
        self._files[label] = dirname
        return dirname

    def _create_tempfile(self, label, *args, **kwds):
        filename = TempfileManager.create_tempfile(*args, **kwds)
        self._files[label] = filename
        return filename

    def _add_tempfile(self, label, filename):
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
            exe = pyomo.common.Executable(name)
            # This is a bit awkward, but we want Executable to re-check
            # the PATH, so we will explicitly call rehash().  In the
            # future, we should move to have the solver directly use the
            # Executable() singleton to manage getting / setting /
            # overriding paths to various executables.  Setting the
            # executable through the Executable() singleton will
            # automatically re-check the PATH.
            exe.rehash()
            exe = exe.path()
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
        TempfileManager.push()
        try:
            return super(SPSolverShellCommand, self).solve(sp, *args, **kwds)
        finally:
            # cleanup
            TempfileManager.pop(remove=not keep_solver_files)
            if keep_solver_files:
                logger.info("Retaining the following solver files:\n"
                            +pprint.pformat(self.files))
