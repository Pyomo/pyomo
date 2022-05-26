#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import os
import os.path
import subprocess

from pyomo.common import Executable
from pyomo.common.collections import Bunch
from pyomo.common.tempfiles import TempfileManager

from pyomo.opt.base import ProblemFormat, ResultsFormat
from pyomo.opt.base.solvers import _extract_version, SolverFactory
from pyomo.opt.results import SolverStatus, TerminationCondition, SolutionStatus 
from pyomo.opt.solver import SystemCallSolver

import logging
logger = logging.getLogger('pyomo.solvers')


@SolverFactory.register('scip', doc='The SCIP LP/MIP solver')
class SCIPAMPL(SystemCallSolver):
    """A generic optimizer that uses the AMPL Solver Library to interface with applications.
    """

    # Cache default executable, so we do not need to repeatedly query the
    # versions every time.
    _known_versions = {}

    def __init__(self, **kwds):
        #
        # Call base constructor
        #
        kwds["type"] = "scip"
        SystemCallSolver.__init__(self, **kwds)
        #
        # Setup valid problem formats, and valid results for each problem format
        # Also set the default problem and results formats.
        #
        self._valid_problem_formats=[ProblemFormat.nl]
        self._valid_result_formats = {}
        self._valid_result_formats[ProblemFormat.nl] = [ResultsFormat.sol]
        self.set_problem_format(ProblemFormat.nl)

        # Note: Undefined capabilities default to 'None'
        self._capabilities = Bunch()
        self._capabilities.linear = True
        self._capabilities.integer = True
        self._capabilities.quadratic_objective = True
        self._capabilities.quadratic_constraint = True
        self._capabilities.sos1 = True
        self._capabilities.sos2 = True

    def _default_results_format(self, prob_format):
        return ResultsFormat.sol

    def _default_executable(self):

        executable = Executable("scip")

        if executable:
            executable_path = executable.path()
            if executable_path not in self._known_versions:
                self._known_versions[executable_path] = self._get_version(executable_path)
            _ver = self._known_versions[executable_path]
            if _ver and _ver >= (8,):
                return executable_path

        # revert to scipampl for older versions
        executable = Executable("scipampl")
        if not executable:
            logger.warning("Could not locate the 'scipampl' executable or"
                           " the 'scip' executable since 8.0.0, which is "
                           "required for solver %s" % self.name)
            self.enable = False
            return None
        return executable.path()

    def _get_version(self, solver_exec=None):
        """
        Returns a tuple describing the solver executable version.
        """
        if solver_exec is None:
            solver_exec = self.executable()
            if solver_exec is None:
                return _extract_version('')
        results = subprocess.run([solver_exec, "--version"], timeout=1,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT,
                                 universal_newlines=True)
        return _extract_version(results.stdout)

    def create_command_line(self, executable, problem_files):

        assert(self._problem_format == ProblemFormat.nl)
        assert(self._results_format == ResultsFormat.sol)

        #
        # Define log file
        #
        if self._log_file is None:
            self._log_file = TempfileManager.\
                             create_tempfile(suffix="_scipampl.log")

        fname = problem_files[0]
        if '.' in fname:
            tmp = fname.split('.')
            if len(tmp) > 2:
                fname = '.'.join(tmp[:-1])
            else:
                fname = tmp[0]
        self._soln_file = fname+".sol"

        #
        # Define results file (since an external parser is used)
        #
        self._results_file = self._soln_file

        #
        # Define command line
        #
        env=os.environ.copy()
        #
        # Merge the PYOMO_AMPLFUNC (externals defined within
        # Pyomo/Pyomo) with any user-specified external function
        # libraries
        #
        if 'PYOMO_AMPLFUNC' in env:
            if 'AMPLFUNC' in env:
                env['AMPLFUNC'] += "\n" + env['PYOMO_AMPLFUNC']
            else:
                env['AMPLFUNC'] = env['PYOMO_AMPLFUNC']

        # Since Version 8.0.0 .nl problem file paths should be provided without the .nl
        # extension
        if executable not in self._known_versions:
            self._known_versions[executable] = self._get_version(executable)
        _ver = self._known_versions[executable]
        if _ver and _ver >= (8, 0, 0):
            problem_file = os.path.splitext(problem_files[0])[0]
        else:
            problem_file = problem_files[0]

        cmd = [executable, problem_file, '-AMPL']
        if self._timer:
            cmd.insert(0, self._timer)

        # GAH: I am going to re-add the code by Zev that passed options through
        # to the command line. I'm not sure what solvers this method of passing options
        # through the envstr variable works for, but it does not seem to work for cplex
        # or gurobi
        env_opt=[]
        of_opt = []
        for key in self.options:
            if key == 'solver':
                continue
            if isinstance(self.options[key], str) and ' ' in self.options[key]:
                env_opt.append(key+"=\""+str(self.options[key])+"\"")
            else:
                env_opt.append(key+"="+str(self.options[key]))
            of_opt.append(str(key)+" = "+str(self.options[key]))

        if self._timelimit is not None and self._timelimit > 0.0 and 'limits/time' not in self.options:
            of_opt.append("limits/time = " + str(self._timelimit))

        envstr = "%s_options" % self.options.solver
        # Merge with any options coming in through the environment
        env[envstr] = " ".join(env_opt)

        if len(of_opt) > 0:
            # Now check if an 'scip.set' file exists in the
            # current working directory. If so, we need to
            # make it clear that this file will be ignored.
            default_of_name = os.path.join(os.getcwd(), 'scip.set')
            if os.path.exists(default_of_name):
                logger.warning("A file named '%s' exists in "
                               "the current working directory, but "
                               "SCIP options are being set using a "
                               "separate options file. The options "
                               "file '%s' will be ignored."
                               % (default_of_name, default_of_name))

            options_dir = TempfileManager.create_tempdir()
            # Now write the new options file
            with open(os.path.join(options_dir, 'scip.set'), 'w') as f:
                for line in of_opt:
                    f.write(line+"\n")
        else:
            options_dir = None

        return Bunch(cmd=cmd, log_file=self._log_file, env=env, cwd=options_dir)

    def _postsolve(self):
        results = super(SCIPAMPL, self)._postsolve()
        if results.solver.message == "unknown":
            results.solver.status = \
                SolverStatus.unknown
            results.solver.termination_condition = \
                TerminationCondition.unknown
            if len(results.solution) > 0:
                results.solution(0).status = \
                    SolutionStatus.unknown
        elif results.solver.message == "user interrupt":
            results.solver.status = \
                SolverStatus.aborted
            results.solver.termination_condition = \
                TerminationCondition.userInterrupt
            if len(results.solution) > 0:
                results.solution(0).status = \
                    SolutionStatus.unknown
        elif results.solver.message == "node limit reached":
            results.solver.status = \
                SolverStatus.aborted
            results.solver.termination_condition = \
                TerminationCondition.maxEvaluations
            if len(results.solution) > 0:
                results.solution(0).status = \
                    SolutionStatus.stoppedByLimit
        elif results.solver.message == "total node limit reached":
            results.solver.status = \
                SolverStatus.aborted
            results.solver.termination_condition = \
                TerminationCondition.maxEvaluations
            if len(results.solution) > 0:
                results.solution(0).status = \
                    SolutionStatus.stoppedByLimit
        elif results.solver.message == "stall node limit reached":
            results.solver.status = \
                SolverStatus.aborted
            results.solver.termination_condition = \
                TerminationCondition.maxEvaluations
            if len(results.solution) > 0:
                results.solution(0).status = \
                    SolutionStatus.stoppedByLimit
        elif results.solver.message == "time limit reached":
            results.solver.status = \
                SolverStatus.aborted
            results.solver.termination_condition = \
                TerminationCondition.maxTimeLimit
            if len(results.solution) > 0:
                results.solution(0).status = \
                    SolutionStatus.stoppedByLimit
        elif results.solver.message == "memory limit reached":
            results.solver.status = \
                SolverStatus.aborted
            results.solver.termination_condition = \
                TerminationCondition.other
            if len(results.solution) > 0:
                results.solution(0).status = \
                    SolutionStatus.stoppedByLimit
        elif results.solver.message == "gap limit reached":
            results.solver.status = \
                SolverStatus.aborted
            results.solver.termination_condition = \
                TerminationCondition.other
            if len(results.solution) > 0:
                results.solution(0).status = \
                    SolutionStatus.stoppedByLimit
        elif results.solver.message == "solution limit reached":
            results.solver.status = \
                SolverStatus.aborted
            results.solver.termination_condition = \
                TerminationCondition.other
            if len(results.solution) > 0:
                results.solution(0).status = \
                    SolutionStatus.stoppedByLimit
        elif results.solver.message == "solution improvement limit reached":
            results.solver.status = \
                SolverStatus.aborted
            results.solver.termination_condition = \
                TerminationCondition.other
            if len(results.solution) > 0:
                results.solution(0).status = \
                    SolutionStatus.stoppedByLimit
        elif results.solver.message == "optimal solution found":
            results.solver.status = \
                SolverStatus.ok
            results.solver.termination_condition = \
                TerminationCondition.optimal
            if len(results.solution) > 0:
                results.solution(0).status = \
                    SolutionStatus.optimal
        elif results.solver.message == "infeasible":
            results.solver.status = \
                SolverStatus.warning
            results.solver.termination_condition = \
                TerminationCondition.infeasible
            if len(results.solution) > 0:
                results.solution(0).status = \
                    SolutionStatus.infeasible
        elif results.solver.message == "unbounded":
            results.solver.status = \
                SolverStatus.warning
            results.solver.termination_condition = \
                TerminationCondition.unbounded
            if len(results.solution) > 0:
                results.solution(0).status = \
                    SolutionStatus.unbounded
        elif results.solver.message == "infeasible or unbounded":
            results.solver.status = \
                SolverStatus.warning
            results.solver.termination_condition = \
                TerminationCondition.infeasibleOrUnbounded
            if len(results.solution) > 0:
                results.solution(0).status = \
                    SolutionStatus.unsure
        else:
            logger.warning("Unexpected SCIP solver message: %s"
                           % (results.solver.message))
            results.solver.status = \
                SolverStatus.unknown
            results.solver.termination_condition = \
                TerminationCondition.unknown
            if len(results.solution) > 0:
                results.solution(0).status = \
                    SolutionStatus.unknown

        return results
