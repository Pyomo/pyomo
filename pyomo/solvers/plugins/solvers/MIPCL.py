#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import logging

import pyutilib.services as services

from pyutilib.misc import Bunch, Options
from pyutilib.services import TempfileManager

from pyomo.util.plugin import alias
from pyomo.opt import SolverResults, TerminationCondition
from pyomo.opt.base import OptSolver, SolverFactory, ProblemFormat, ResultsFormat
from pyomo.opt.solver import SystemCallSolver
from pyomo.opt.base.solvers import _extract_version

from six import iteritems, string_types

logger = logging.getLogger('pyomo.solvers')

_mipcl_version = None

class MIPCL(OptSolver):
    """The MIPCL LP/MIP solver"""

    alias('mipcl', doc='The MIPCL LP/MIP solver')

    def __new__(cls, *args, **kwds):
        kwds['solver_options']['skip_objective_sense'] = True
        mode = kwds.pop('solver_io', 'mps')
        if mode == 'mps':
            opt = SolverFactory('_mipcl_shell', **kwds)
            opt.set_problem_format(ProblemFormat.mps)
            return opt
        else:
            logger.error('Unknown IO type: %s' % mode)
            return
        opt.set_options('solver=mps_mipcl')
        return opt

class MIPCLSHELL(SystemCallSolver):
    """Shell interface to the MIPCL LP/MIP solver"""

    alias('_mipcl_shell', doc='Shell interface to the MIPCL LP/MIP solver')

    def __init__(self, **kwds):
        kwds['type'] = 'mipcl'
        SystemCallSolver.__init__(self, **kwds)

        self._valid_problem_formats = [ProblemFormat.mps]
        self._valid_result_formats = {ProblemFormat.mps: ResultsFormat.soln}
        self.set_problem_format(ProblemFormat.mps)

        self._capabilities = Options()
        self._capabilities.linear = True
        self._capabilities.integer = True

    def _default_results_format(self, prob_format):
        return ResultsFormat.soln

    def _default_executable(self):
        executable = services.registered_executable('mps_mipcl')
        if executable is None:
            logger.warning("Could not locate the 'mps_mipcl' executable,"
                           " which is required for solver '%s'" % self.name)
            self.enable = False
            return None
        return executable.get_path()

    def _get_version(self):
        """Returns a tuple describing the solver executable version."""
        if _mipcl_version is None:
            return _extract_version('')
        return _mipcl_version

    def _set_version(self, ver):
        """Sets version while processing log file"""
        global _mipcl_version
        if _mipcl_version is None:
            _mipcl_version = ver

    def create_command_line(self, executable, problem_files):
        if self._log_file is None:
            self._log_file = TempfileManager.create_tempfile(suffix='.mipcl.log')

        problem_filename_prefix = problem_files[0]
        if '.' in problem_filename_prefix:
            tmp = problem_filename_prefix.split('.')
            if len(tmp) > 2:
                problem_filename_prefix = '.'.join(tmp[:-1])
            else:
                problem_filename_prefix = tmp[0]
        self._soln_file = problem_filename_prefix+".sol"

        cmd = [executable, problem_files[0], '-solfile', self._soln_file]
        if self._timer:
            cmd.insert(0, self._timer)
        for k, v in iteritems(self.options):
            if v is None or (isinstance(v, string_types) and v.strip() == ''):
                cmd.append("-%s" % k)
            else:
                cmd.extend(["-%s" % k, str(v)])

        if self._timelimit is not None and self._timelimit > 0.0:
            cmd.extend(['-time', str(self._timelimit)])

        return Bunch(cmd=cmd, log_file=self._log_file, env=None)

    def process_logfile(self):
        """Process logfile"""
        results = SolverResults()

        prob   = results.problem
        solv   = results.solver
        solv.termination_condition = TerminationCondition.unknown
        stats  = results.solver.statistics
        bbound = stats.branch_and_bound

        prob.upper_bound = float('inf')
        prob.lower_bound = float('-inf')
        bbound.number_of_created_subproblems = 0
        bbound.number_of_bounded_subproblems = 0

        with open(self._log_file, 'r') as f:
            lines = f.readlines()
            lines = [line.rstrip('\r\n') for line in lines]
        for i, line in enumerate(lines):
            if i in [0, 1, 2, 3, 4, 5] or line == '':
                continue
            toks = line.split()
            if 'NAME' in line and len(toks) == 2:
                prob.name = toks[-1]
            elif 'Start preprocessing:' in line:
                prob.number_of_constraints = eval(toks[4][:-1]) # Rows ?
                prob.number_of_nonzeros    = eval(toks[-1])
                prob.number_of_variables   = eval(toks[7]) # Cols ?
            elif 'After preprocessing:' in line:
                prob.number_of_constraints = eval(toks[4][:-1]) # Rows ?
                prob.number_of_nonzeros    = eval(toks[-1])
                prob.number_of_variables   = eval(toks[7]) # Cols ?
            elif 'Preprocessing Time:' in line:
                solv.system_time = eval(toks[-1])
            elif 'Solution time:' in line:
                solv.system_time = eval(toks[-1])
            elif 'MIPCL version' in line:
                self._set_version(line)
            elif 'Branch-and-Cut' in line:
                nodes = eval(toks[-1]) # number_of_created or number_of_bounded ?
            elif 'Objective value:' in line:
                value = eval(toks[2]) # assign ?
                if toks[-2] + toks[-1] == 'optimalityproven':
                    solv.termination_condition = TerminationCondition.optimal

        return results

    def process_soln_file(self, results):
        """Process solution file"""
        try:
            with open(self._soln_file) as f:
                lines = f.readlines()
            for i, line in enumerate(lines):
                if i == 0:
                    line = line.split()
                    if line[0] == '=infeas=':
                        results.Solver.Termination_condition = TerminationCondition.infeasible
                    elif line[0] == '=obj=':
                        results.Solution.Objective['obj'] = {"Value" : eval(line[1])}
                    else:
                        raise RuntimeError('objective status unknown')
                else:
                    line = line.split()
                    results.solution.variable[line[0]] = {"Value" : eval(line[1])} 
        except:
            raise RuntimeError('please debug me')



services.register_executable(name="mps_mipcl")
