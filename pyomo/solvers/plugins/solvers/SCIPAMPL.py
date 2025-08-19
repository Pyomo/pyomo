#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import os

# import os.path
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
    """A generic optimizer that uses the AMPL Solver Library to interface with applications."""

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
        self._valid_problem_formats = [ProblemFormat.nl]
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
                self._known_versions[executable_path] = self._get_version(
                    executable_path
                )
            _ver = self._known_versions[executable_path]
            if _ver and _ver >= (8,):
                return executable_path

        # revert to scipampl for older versions
        executable = Executable("scipampl")
        if not executable:
            logger.warning(
                "Could not locate the 'scip' executable or"
                " the older 'scipampl' executable, which is "
                "required for solver %s" % self.name
            )
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
        results = subprocess.run(
            [solver_exec, "--version"],
            timeout=self._version_timeout,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        return _extract_version(results.stdout)

    def create_command_line(self, executable, problem_files):
        assert self._problem_format == ProblemFormat.nl
        assert self._results_format == ResultsFormat.sol

        #
        # Define log file
        #
        if self._log_file is None:
            self._log_file = TempfileManager.create_tempfile(suffix="_scip.log")

        fname = problem_files[0]
        if '.' in fname:
            tmp = fname.split('.')
            if len(tmp) > 2:
                fname = '.'.join(tmp[:-1])
            else:
                fname = tmp[0]
        self._soln_file = fname + ".sol"

        #
        # Define results file (since an external parser is used)
        #
        self._results_file = self._soln_file

        #
        # Define command line
        #
        env = os.environ.copy()
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
        env_opt = []
        of_opt = []
        for key in self.options:
            if key == 'solver':
                continue
            if isinstance(self.options[key], str) and ' ' in self.options[key]:
                env_opt.append(key + "=\"" + str(self.options[key]) + "\"")
            else:
                env_opt.append(key + "=" + str(self.options[key]))
            of_opt.append(str(key) + " = " + str(self.options[key]))

        if (
            self._timelimit is not None
            and self._timelimit > 0.0
            and 'limits/time' not in self.options
        ):
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
                logger.warning(
                    "A file named '%s' exists in "
                    "the current working directory, but "
                    "SCIP options are being set using a "
                    "separate options file. The options "
                    "file '%s' will be ignored." % (default_of_name, default_of_name)
                )

            options_dir = TempfileManager.create_tempdir()
            # Now write the new options file
            with open(os.path.join(options_dir, 'scip.set'), 'w') as f:
                for line in of_opt:
                    f.write(line + "\n")
        else:
            options_dir = None

        return Bunch(cmd=cmd, log_file=self._log_file, env=env, cwd=options_dir)

    def _postsolve(self):
        # find SCIP version (calling version() and _get_version() mess things)

        executable = self._command.cmd[0]

        version = self._known_versions[executable]

        if version < (8, 0, 0, 0):
            # it may be possible to get results from older version but this was
            # not tested, so the old way of doing things is here preserved

            results = super(SCIPAMPL, self)._postsolve()

        else:
            # repeat code from super(SCIPAMPL, self)._postsolve()
            # in order to access the log file and get the results from there

            if self._log_file is not None:
                OUTPUT = open(self._log_file, "w")
                OUTPUT.write("Solver command line: " + str(self._command.cmd) + '\n')
                OUTPUT.write("\n")
                OUTPUT.write(self._log + '\n')
                OUTPUT.close()

            # JPW: The cleanup of the problem file probably shouldn't be here, but
            #   rather in the base OptSolver class. That would require movement of
            #   the keepfiles attribute and associated cleanup logic to the base
            #   class, which I didn't feel like doing at this present time. the
            #   base class remove_files method should clean up the problem file.

            if (self._log_file is not None) and (not os.path.exists(self._log_file)):
                msg = "File '%s' not generated while executing %s"
                raise IOError(msg % (self._log_file, self.path))
            results = None

            if self._results_format is not None:
                results = self.process_output(self._rc)

                # read results from the log file

                log_dict = self.read_scip_log(self._log_file)

                if len(log_dict) != 0:
                    # if any were read, store them

                    results.solver.time = log_dict['solving_time']
                    results.solver.gap = log_dict['gap']
                    results.solver.node_count = log_dict['solving_nodes']
                    results.solver.primal_bound = log_dict['primal_bound']
                    results.solver.dual_bound = log_dict['dual_bound']

                # TODO: get scip to produce a statistics file and read it
                # Why? It has all the information one can possibly need.
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
                    if (not self._soln_file is None) and os.path.exists(
                        self._soln_file
                    ):
                        os.remove(self._soln_file)

            TempfileManager.pop(remove=not self._keepfiles)

        # **********************************************************************
        # **********************************************************************

        # UNKNOWN # unknown='unknown' # An uninitialized value

        if "unknown" in results.solver.message:
            results.solver.status = SolverStatus.unknown
            results.solver.termination_condition = TerminationCondition.unknown
            if len(results.solution) > 0:
                results.solution(0).status = SolutionStatus.unknown

        # ABORTED # userInterrupt='userInterrupt' # Interrupt signal generated by user

        elif "user interrupt" in results.solver.message:
            results.solver.status = SolverStatus.aborted
            results.solver.termination_condition = TerminationCondition.userInterrupt
            if len(results.solution) > 0:
                results.solution(0).status = SolutionStatus.unknown

        # OK # maxEvaluations='maxEvaluations' # Exceeded maximum number of problem evaluations

        elif "node limit reached" in results.solver.message:
            results.solver.status = SolverStatus.ok
            results.solver.termination_condition = TerminationCondition.maxEvaluations
            if len(results.solution) > 0:
                results.solution(0).status = SolutionStatus.stoppedByLimit

        # OK # maxEvaluations='maxEvaluations' # Exceeded maximum number of problem evaluations

        elif "total node limit reached" in results.solver.message:
            results.solver.status = SolverStatus.ok
            results.solver.termination_condition = TerminationCondition.maxEvaluations
            if len(results.solution) > 0:
                results.solution(0).status = SolutionStatus.stoppedByLimit

        # OK # maxEvaluations='maxEvaluations' # Exceeded maximum number of problem evaluations

        elif "stall node limit reached" in results.solver.message:
            results.solver.status = SolverStatus.ok
            results.solver.termination_condition = TerminationCondition.maxEvaluations
            if len(results.solution) > 0:
                results.solution(0).status = SolutionStatus.stoppedByLimit

        # OK # maxTimeLimit='maxTimeLimit' # Exceeded maximum time limited allowed by user but having return a feasible solution

        elif "time limit reached" in results.solver.message:
            results.solver.status = SolverStatus.ok
            results.solver.termination_condition = TerminationCondition.maxTimeLimit
            if len(results.solution) > 0:
                results.solution(0).status = SolutionStatus.stoppedByLimit

        # OK # other='other' # Other, uncategorized normal termination

        elif "memory limit reached" in results.solver.message:
            results.solver.status = SolverStatus.ok
            results.solver.termination_condition = TerminationCondition.other
            if len(results.solution) > 0:
                results.solution(0).status = SolutionStatus.stoppedByLimit

        # OK # other='other' # Other, uncategorized normal termination

        elif "gap limit reached" in results.solver.message:
            results.solver.status = SolverStatus.ok
            results.solver.termination_condition = TerminationCondition.other
            if len(results.solution) > 0:
                results.solution(0).status = SolutionStatus.stoppedByLimit

        # OK # other='other' # Other, uncategorized normal termination

        elif "solution limit reached" in results.solver.message:
            results.solver.status = SolverStatus.ok
            results.solver.termination_condition = TerminationCondition.other
            if len(results.solution) > 0:
                results.solution(0).status = SolutionStatus.stoppedByLimit

        # OK # other='other' # Other, uncategorized normal termination

        elif "solution improvement limit reached" in results.solver.message:
            results.solver.status = SolverStatus.ok
            results.solver.termination_condition = TerminationCondition.other
            if len(results.solution) > 0:
                results.solution(0).status = SolutionStatus.stoppedByLimit

        # OK # optimal='optimal' # Found an optimal solution

        elif "optimal solution" in results.solver.message:
            results.solver.status = SolverStatus.ok
            results.solver.termination_condition = TerminationCondition.optimal
            if len(results.solution) > 0:
                results.solution(0).status = SolutionStatus.optimal
            try:
                if results.solver.primal_bound < results.solver.dual_bound:
                    results.problem.lower_bound = results.solver.primal_bound
                    results.problem.upper_bound = results.solver.dual_bound
                else:
                    results.problem.lower_bound = results.solver.dual_bound
                    results.problem.upper_bound = results.solver.primal_bound
            except AttributeError:
                """
                This may occur if SCIP solves the problem during presolve. In that case,
                the log file may not get parsed correctly (self.read_scip_log), and
                results.solver.primal_bound will not be populated.
                """
                pass

        # WARNING # infeasible='infeasible' # Demonstrated that the problem is infeasible

        elif "infeasible" in results.solver.message:
            results.solver.status = SolverStatus.warning
            results.solver.termination_condition = TerminationCondition.infeasible
            if len(results.solution) > 0:
                results.solution(0).status = SolutionStatus.infeasible

        # WARNING # unbounded='unbounded' # Demonstrated that problem is unbounded

        elif "unbounded" in results.solver.message:
            results.solver.status = SolverStatus.warning
            results.solver.termination_condition = TerminationCondition.unbounded
            if len(results.solution) > 0:
                results.solution(0).status = SolutionStatus.unbounded

        # WARNING # infeasibleOrUnbounded='infeasibleOrUnbounded'   # Problem is either infeasible or unbounded

        elif "infeasible or unbounded" in results.solver.message:
            results.solver.status = SolverStatus.warning
            results.solver.termination_condition = (
                TerminationCondition.infeasibleOrUnbounded
            )
            if len(results.solution) > 0:
                results.solution(0).status = SolutionStatus.unsure

        # UNKNOWN # unknown='unknown' # An uninitialized value

        else:
            logger.warning(
                "Unexpected SCIP solver message: %s" % (results.solver.message)
            )
            results.solver.status = SolverStatus.unknown
            results.solver.termination_condition = TerminationCondition.unknown
            if len(results.solution) > 0:
                results.solution(0).status = SolutionStatus.unknown

        return results

    @staticmethod
    def read_scip_log(filename: str):
        # TODO: check file exists, ensure opt has finished, etc

        from collections import deque

        with open(filename) as f:
            scip_lines = list(deque(f, 7))
            scip_lines.pop()

        expected_labels = [
            'SCIP Status        :',
            'Solving Time (sec) :',
            'Solving Nodes      :',
            'Primal Bound       :',
            'Dual Bound         :',
            'Gap                :',
        ]

        colon_position = 19  # or scip_lines[0].index(':')

        for i, log_file_line in enumerate(scip_lines):
            if expected_labels[i] != log_file_line[0 : colon_position + 1]:
                return {}

        # get data

        solver_status = scip_lines[0][colon_position + 2 : scip_lines[0].index('\n')]

        solving_time = float(
            scip_lines[1][colon_position + 2 : scip_lines[1].index('\n')].split(' ')[0]
        )

        try:
            solving_nodes = int(
                scip_lines[2][colon_position + 2 : scip_lines[2].index('(')]
            )
        except ValueError:
            solving_nodes = int(
                scip_lines[2][colon_position + 2 : scip_lines[2].index('\n')]
            )

        primal_bound = float(
            scip_lines[3][colon_position + 2 : scip_lines[3].index('(')]
        )

        dual_bound = float(
            scip_lines[4][colon_position + 2 : scip_lines[4].index('\n')]
        )

        try:
            gap = float(scip_lines[5][colon_position + 2 : scip_lines[5].index('%')])
        except ValueError:
            gap = scip_lines[5][colon_position + 2 : scip_lines[5].index('\n')]

            if gap == 'infinite':
                gap = float('inf')

        out_dict = {
            'solver_status': solver_status,
            'solving_time': solving_time,
            'solving_nodes': solving_nodes,
            'primal_bound': primal_bound,
            'dual_bound': dual_bound,
            'gap': gap,
        }

        return out_dict
