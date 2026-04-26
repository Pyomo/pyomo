# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import os
import re
import logging
import subprocess

from pyomo.common import Executable
from pyomo.common.enums import minimize
from pyomo.common.collections import Bunch
from pyomo.common.tempfiles import TempfileManager

from pyomo.opt.base import ProblemFormat, ResultsFormat, OptSolver
from pyomo.opt.base.solvers import _extract_version, SolverFactory
from pyomo.opt.results import (
    SolverResults,
    SolverStatus,
    TerminationCondition,
    SolutionStatus,
    Solution,
)
from pyomo.opt.solver import SystemCallSolver

logger = logging.getLogger('pyomo.solvers')


@SolverFactory.register('highs', doc='The HiGHS LP/MIP solver')
class HIGHS(OptSolver):
    """The HiGHS LP/MIP solver"""

    def __new__(cls, *args, **kwds):
        mode = kwds.pop('solver_io', 'lp')
        if mode == 'lp' or mode == 'mps' or mode is None:
            opt = SolverFactory('_highs_shell', **kwds)
            if mode == 'mps':
                opt.set_problem_format(ProblemFormat.mps)
            else:
                opt.set_problem_format(ProblemFormat.cpxlp)
            return opt
        elif mode == 'direct' or mode == 'python':
            return SolverFactory('highs_persistent_v2', **kwds)
        else:
            logger.error('Unknown IO type: %s' % mode)
            return SolverFactory('_failsafe_unknown_solver')


@SolverFactory.register('_highs_shell', doc='Shell interface to the HiGHS solver')
class HIGHSSHELL(SystemCallSolver):
    """Shell interface to the HiGHS LP/MIP solver"""

    def __init__(self, **kwds):
        kwds['type'] = 'highs'
        super(HIGHSSHELL, self).__init__(**kwds)

        self._valid_problem_formats = [
            ProblemFormat.cpxlp,
            ProblemFormat.mps,
        ]
        self._valid_result_formats = {
            ProblemFormat.cpxlp: [ResultsFormat.soln],
            ProblemFormat.mps: [ResultsFormat.soln],
        }

        self._capabilities = Bunch()
        self._capabilities.linear = True
        self._capabilities.integer = True
        self._capabilities.quadratic_objective = True
        self._capabilities.quadratic_constraint = False
        self._capabilities.sos1 = False
        self._capabilities.sos2 = False

        self.set_problem_format(ProblemFormat.cpxlp)
        self._timelimit = None

    def _default_results_format(self, prob_format):
        return ResultsFormat.soln

    def _default_executable(self):
        executable = Executable("highs")
        if not executable:
            logger.warning(
                "Could not locate the 'highs' executable, which is "
                "required for solver %s" % self.name
            )
            self.enable = False
            return None
        return executable.path()

    def _get_version(self):
        result = subprocess.run(
            [self.executable(), "--version"],
            timeout=5,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        return _extract_version(result.stdout)

    def create_command_line(self, executable, problem_files):
        if self._log_file is None:
            self._log_file = TempfileManager.create_tempfile(suffix=".highs.log")

        problem_filename_prefix = problem_files[0]
        if '.' in problem_filename_prefix:
            tmp = problem_filename_prefix.split('.')
            if len(tmp) > 2:
                problem_filename_prefix = '.'.join(tmp[:-1])
            else:
                problem_filename_prefix = tmp[0]

        self._soln_file = problem_filename_prefix + ".sol"

        cmd = [executable, problem_files[0]]

        if self._timelimit is not None and self._timelimit > 0.0:
            cmd.extend(['--time_limit', str(self._timelimit)])

        cmd.extend(['--solution_file', self._soln_file])

        for key, val in self.options.items():
            if val is None or (isinstance(val, str) and val.strip() == ''):
                cmd.append(f"--{key}")
            else:
                cmd.extend([f"--{key}", str(val)])

        return Bunch(cmd=cmd, log_file=self._log_file, env=None)

    def process_logfile(self):
        results = SolverResults()
        soln = Solution()

        results.problem.name = None
        results.problem.sense = minimize
        optim_value = None

        try:
            with open(self._log_file, 'r') as f:
                output = f.read()
        except Exception:
            output = ""

        for line in output.split("\n"):
            line = line.strip()
            if line.startswith("Model status"):
                status_str = line.split(":")[-1].strip().lower()
                if status_str == "optimal":
                    results.solver.status = SolverStatus.ok
                    results.solver.termination_condition = TerminationCondition.optimal
                    soln.status = SolutionStatus.optimal
                elif status_str == "infeasible":
                    results.solver.status = SolverStatus.warning
                    results.solver.termination_condition = TerminationCondition.infeasible
                    soln.status = SolutionStatus.infeasible
                elif status_str == "unbounded":
                    results.solver.status = SolverStatus.warning
                    results.solver.termination_condition = TerminationCondition.unbounded
                    soln.status = SolutionStatus.unbounded
                elif status_str in ('infeasible or unbounded', 'infeasibleorunbounded'):
                    results.solver.status = SolverStatus.warning
                    results.solver.termination_condition = TerminationCondition.infeasibleOrUnbounded
                    soln.status = SolutionStatus.infeasible
                elif status_str in ('time limit reached', 'solution limit reached'):
                    results.solver.status = SolverStatus.aborted
                    results.solver.termination_condition = TerminationCondition.maxTimeLimit
                    soln.status = SolutionStatus.stoppedByLimit
                elif status_str == 'iteration limit reached':
                    results.solver.status = SolverStatus.aborted
                    results.solver.termination_condition = TerminationCondition.maxIterations
                    soln.status = SolutionStatus.stoppedByLimit
            elif line.startswith("Objective value"):
                try:
                    optim_value = float(line.split(":")[-1].strip())
                except ValueError:
                    pass

        if soln.status is SolutionStatus.optimal and optim_value is not None:
            soln.objective['__default_objective__'] = {'Value': optim_value}

        if soln.status in [SolutionStatus.optimal, SolutionStatus.stoppedByLimit]:
            results.solution.insert(soln)

        return results

    def process_soln_file(self, results):
        if len(results.solution) == 0:
            return

        soln = results.solution[0]

        extract_duals = False
        extract_reduced_costs = False
        for suffix in self._suffixes:
            if re.match(suffix, "dual"):
                extract_duals = True
            elif re.match(suffix, "rc"):
                extract_reduced_costs = True
            else:
                raise RuntimeError(
                    f"***HiGHS solver plugin cannot extract solution suffix={suffix}"
                )

        if not os.path.exists(self._soln_file):
            return

        with open(self._soln_file, 'r') as f:
            lines = f.readlines()

        section = None
        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("Columns"):
                section = "columns"
                continue
            elif line.startswith("Rows"):
                section = "rows"
                continue
            elif line.startswith("Model status"):
                section = None
                continue
            elif line.startswith("Index Status"):
                continue

            if section == "columns":
                parts = line.split()
                if len(parts) >= 7:
                    try:
                        primal = float(parts[4])
                        dual = float(parts[5])
                        name = parts[6]
                        if extract_reduced_costs:
                            soln.variable[name] = {"Value": primal, "Rc": dual}
                        else:
                            soln.variable[name] = {"Value": primal}
                    except ValueError:
                        pass

            elif section == "rows":
                parts = line.split()
                if len(parts) >= 7:
                    try:
                        dual = float(parts[5])
                        name = parts[6]
                        if extract_duals:
                            soln.constraint[name] = {"Dual": dual}
                    except ValueError:
                        pass
