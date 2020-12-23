#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import re
import os

from six import iteritems

from pyomo.common import Executable
from pyomo.common.errors import  ApplicationError
from pyomo.common.collections import Options, Bunch
from pyomo.common.tempfiles import TempfileManager
from pyutilib.subprocess import run

from pyomo.opt.base import ProblemFormat, ResultsFormat, OptSolver
from pyomo.opt.base.solvers import _extract_version, SolverFactory
from pyomo.opt.results import SolverResults, SolverStatus, TerminationCondition, SolutionStatus, ProblemSense, Solution
from pyomo.opt.solver import  SystemCallSolver
from pyomo.solvers.mockmip import MockMIP

import logging
logger = logging.getLogger('pyomo.solvers')


@SolverFactory.register('pico', doc='The PICO LP/MIP solver')
class PICO(OptSolver):
    """The PICO LP/MIP solver
    """

    def __new__(cls, *args, **kwds):
        try:
            mode = kwds['solver_io']
            if mode is None:
                mode = 'lp'
            del kwds['solver_io']
        except KeyError:
            mode = 'lp'
        #
        if mode  == 'lp':
            opt = SolverFactory('_pico_shell', **kwds)
            opt.set_problem_format(ProblemFormat.cpxlp)
            return opt
        # PICO's MPS parser seems too buggy to expose
        # this option
#        if mode  == 'mps':
#            opt = SolverFactory('_pico_shell', **kwds)
#            opt.set_problem_format(ProblemFormat.mps)
#            return opt
        #
        if mode == 'nl':
            # PICO does not accept all asl style command line options
            # (-s in particular, which is required for streaming output of
            # all asl solvers). Therefore we need to send it through the cbc_shell
            # instead of ASL
            opt = SolverFactory('_pico_shell',**kwds)
            opt.set_problem_format(ProblemFormat.nl)
        elif mode == 'os':
            opt = SolverFactory('_ossolver', **kwds)
        else:
            logger.error('Unknown IO type: %s' % mode)
            return
        opt.set_options('solver=PICO')
        return opt


@SolverFactory.register('_pico_shell', doc='Shell interface to the PICO MIP solver')
class PICOSHELL(SystemCallSolver):
    """Shell interface to the PICO LP/MIP solver
    """

    def __init__(self, **kwds):
        #
        # Call base constructor
        #
        kwds["type"] = "pico"
        SystemCallSolver.__init__(self, **kwds)
        #
        # Setup valid problem formats, and valid results for each problem format
        #
        self._valid_problem_formats=[ProblemFormat.cpxlp, ProblemFormat.nl, ProblemFormat.mps]
        self._valid_result_formats = {}
        self._valid_result_formats[ProblemFormat.cpxlp] = [ResultsFormat.soln]
        self._valid_result_formats[ProblemFormat.nl] = [ResultsFormat.sol]
        self._valid_result_formats[ProblemFormat.mps] = [ResultsFormat.soln]
        self.set_problem_format(ProblemFormat.cpxlp)

        # Note: Undefined capabilities default to 'None'
        self._capabilities = Options()
        self._capabilities.linear = True
        self._capabilities.integer = True
        #self._capabilities.sos1 = True
        #self._capabilities.sos2 = True

    #
    # Override the base class method so we can update
    #  _default_variable_value
    #
    def set_problem_format(self, format):
        #
        # The NL file interface for PICO does not return sparse
        # results
        #
        if format == ProblemFormat.nl:
            self._default_variable_value = None
        else:
            self._default_variable_value = 0.0
        super(PICOSHELL, self).set_problem_format(format)

    def _default_results_format(self, prob_format):
        if prob_format == ProblemFormat.nl:
            return ResultsFormat.sol
        return ResultsFormat.soln

    def _default_executable(self):
        executable = Executable("PICO_deprecated_not_supported")
        if not executable:
            logger.warning("Could not locate the 'PICO' executable, "
                           "which is required for solver %s" % self.name)
            self.enable = False
            return None
        return executable.path()

    def _get_version(self):
        """
        Returns a tuple describing the solver executable version.
        """
        solver_exec = self.executable()
        if solver_exec is None:
            return _extract_version('')
        results = run([solver_exec, "--version"], timelimit=1)
        # 'PICO --version' seems to print 'pebble <version>, PICO <version>
        # we don't wan't the pebble version being advertised so we split
        # the string at the comma before extracting a version number. It
        # also exits with a nonzero return code so don't bother checking it.
        return _extract_version(results[1].split(',')[1])

    # Nothing needs to be done here
    #def _presolve(self, *args, **kwds):
    #    # let the base class handle any remaining keywords/actions.
    #    SystemCallSolver._presolve(self, *args, **kwds)

    def create_command_line(self,executable,problem_files):
        #
        # Define log file
        #
        if self._log_file is None:
            self._log_file = TempfileManager.\
                            create_tempfile(suffix="PICO.log")

        problem_filename_prefix = problem_files[0]
        if '.' in problem_filename_prefix:
            tmp = problem_filename_prefix.split('.')
            if len(tmp) > 2:
                problem_filename_prefix = '.'.join(tmp[:-1])
            else:
                problem_filename_prefix = tmp[0]
        if self._results_format is ResultsFormat.sol:
            self._soln_file = problem_filename_prefix+".sol"
        else:
            self._soln_file = problem_filename_prefix+".soln"

        #
        # Define results file (if the sol external parser is used)
        #
        if self._results_format is ResultsFormat.sol:
            self._results_file = self._soln_file

        #
        # Eventually, these formats will be added to PICO...
        #
        #elif self._results_format == ResultsFormat.osrl:
            #self._results_file = self.tmpDir+os.sep+"PICO.osrl.xml"

        def _check_and_escape_options(options):
            for key, val in iteritems(self.options):
                tmp_k = str(key)
                _bad = ' ' in tmp_k

                tmp_v = str(val)
                if ' ' in tmp_v:
                    if '"' in tmp_v:
                        if "'" in tmp_v:
                            _bad = True
                        else:
                            tmp_v = "'" + tmp_v + "'"
                    else:
                        tmp_v = '"' + tmp_v + '"'

                if _bad:
                    raise ValueError("Unable to properly escape solver option:"
                                     "\n\t%s=%s" % (key, val) )
                yield (tmp_k, tmp_v)

        #
        # Define command line
        #
        cmd = [ executable ]
        if self._timer:
            cmd.insert(0, self._timer)

        if (self.options.mipgap is not None):
            raise ValueError("The mipgap parameter is currently not being "
                             "processed by PICO solver plugin")
        env=os.environ.copy()
        if self._problem_format is None \
                or self._problem_format == ProblemFormat.nl:
            cmd.append(problem_files[0])
            cmd.append('-AMPL')

            opts = ['allowInfiniteIntegerVarBounds=true']
            if "debug" not in self.options:
                opts.append("debug=2")
            for key, val in _check_and_escape_options(self.options):
                if key == 'solver':
                    continue
                opts.append("%s=%s" % ( key, tmp ))
            opts.append('output='+self._soln_file)
            env["PICO_options"] = " ".join(opts)

        elif self._problem_format == ProblemFormat.cpxlp \
                or self._problem_format == ProblemFormat.mps:
            cmd.append('--allowInfiniteIntegerVarBounds=true')
            # This option should appear in the next PICO release,
            # assuming that ever happens
            if self.version() > (1,3,1,0):
                cmd.append('--reportDenseSolution=true')
            if "debug" not in self.options:
                cmd.extend(["--debug", "2"])
            for key, val in _check_and_escape_options(self.options):
                if key == 'mipgap':
                    continue
                cmd.extend(['--'+key, val])
            cmd.extend(['--output', self._soln_file,
                        problem_files[0]])

        return Bunch(cmd=cmd, log_file=self._log_file, env=env)

    def process_logfile(self):
        """
        Process a logfile
        """
        results = SolverResults()

        #
        # Initial values
        #
        #results.solver.statistics.branch_and_bound.number_of_created_subproblems=0
        #results.solver.statistics.branch_and_bound.number_of_bounded_subproblems=0
        soln = Solution()
        soln.objective['__default_objective__'] = {'Value': None}
        #
        # Process logfile
        #
        OUTPUT = open(self._log_file)
        output = "".join(OUTPUT.readlines())
        OUTPUT.close()
        #
        # Parse logfile lines
        #
        for line in output.split("\n"):
            tokens = re.split('[ \t]+',line.strip())
            if len(tokens) > 3 and tokens[0] == "ABORTED:":
                results.solver.status=SolverStatus.aborted
            elif len(tokens) > 1 and tokens[0].startswith("ERROR"):
                results.solver.status=SolverStatus.error
            elif len(tokens) == 3 and tokens[0] == 'Problem' and tokens[2].startswith('infeasible'):
                results.solver.termination_condition = TerminationCondition.infeasible
            elif len(tokens) == 2 and tokens[0] == 'Integer' and tokens[1] == 'Infeasible':
                results.solver.termination_condition = TerminationCondition.infeasible
            elif len(tokens) == 5 and tokens[0] == "Final" and tokens[1] == "Solution:":
                soln.objective['__default_objective__']['Value'] = eval(tokens[4])
                soln.status = SolutionStatus.optimal
            elif len(tokens) == 3 and tokens[0] == "LP" and tokens[1] == "value=":
                soln.objective['__default_objective__']['Value'] = eval(tokens[2])
                soln.status=SolutionStatus.optimal
                if results.problem.sense == ProblemSense.minimize:
                    results.problem.lower_bound = eval(tokens[2])
                else:
                    results.problem.upper_bound = eval(tokens[2])
            elif len(tokens) == 2 and tokens[0] == "Bound:":
                if results.problem.sense == ProblemSense.minimize:
                    results.problem.lower_bound = eval(tokens[1])
                else:
                    results.problem.upper_bound = eval(tokens[1])
            elif len(tokens) == 3 and tokens[0] == "Created":
                results.solver.statistics.branch_and_bound.number_of_created_subproblems = eval(tokens[1])
            elif len(tokens) == 3 and tokens[0] == "Bounded":
                results.solver.statistics.branch_and_bound.number_of_bounded_subproblems = eval(tokens[1])
            elif len(tokens) == 2 and tokens[0] == "sys":
                results.solver.system_time=eval(tokens[1])
            elif len(tokens) == 2 and tokens[0] == "user":
                results.solver.user_time=eval(tokens[1])
            elif len(tokens) == 3 and tokens[0] == "Solving" and tokens[1] == "problem:":
                results.problem.name = tokens[2]
            elif len(tokens) == 4 and tokens[2] == "constraints:":
                results.problem.number_of_constraints = eval(tokens[3])
            elif len(tokens) == 4 and tokens[2] == "variables:":
                results.problem.number_of_variables = eval(tokens[3])
            elif len(tokens) == 4 and tokens[2] == "nonzeros:":
                results.problem.number_of_nonzeros = eval(tokens[3])
            elif len(tokens) == 3 and tokens[1] == "Sense:":
                if tokens[2] == "minimization":
                    results.problem.sense = ProblemSense.minimize
                else:
                    results.problem.sense = ProblemSense.maximize

        if results.solver.status is SolverStatus.aborted:
            soln.optimality=SolutionStatus.unsure
        if soln.status is SolutionStatus.optimal:
            soln.gap=0.0
            results.problem.lower_bound = soln.objective['__default_objective__']['Value']
            results.problem.upper_bound = soln.objective['__default_objective__']['Value']

        if soln.status == SolutionStatus.optimal:
            results.solver.termination_condition = TerminationCondition.optimal

        if not results.solver.status is SolverStatus.error and \
            results.solver.termination_condition in [TerminationCondition.unknown,
                        #TerminationCondition.maxIterations,
                        #TerminationCondition.minFunctionValue,
                        #TerminationCondition.minStepLength,
                        TerminationCondition.globallyOptimal,
                        TerminationCondition.locallyOptimal,
                        TerminationCondition.optimal,
                        #TerminationCondition.maxEvaluations,
                        TerminationCondition.other]:
                results.solution.insert(soln)
        return results


    def process_soln_file(self,results):

        if self._results_format is ResultsFormat.sol:
            return

        # the only suffixes that we extract from PICO are
        # constraint duals. scan through the solver suffix
        # list and throw an exception if the user has
        # specified any others.
        extract_duals = False
        for suffix in self._suffixes:
            if re.match(suffix,"dual"):
                extract_duals = True
            else:
                raise RuntimeError("***PICO solver plugin cannot extract solution suffix="+suffix)

        #if os.path.exists(self.sol_file):
            #results_reader = ReaderFactory(ResultsFormat.sol)
            #results = results_reader(self.sol_file, results, results.solution(0))
            #return

        if not os.path.exists(self._soln_file):
            return
        if len(results.solution) == 0:
            return
        soln = results.solution(0)
        results.problem.num_objectives=1
        tmp=[]
        flag=False
        INPUT = open(self._soln_file, "r")
        lp_flag=None
        var_flag=True
        for line in INPUT:
            tokens = re.split('[ \t]+',line.strip())
            if len(tokens) == 0 or (len(tokens) == 1 and tokens[0]==''):
                continue
            if tokens[0] == "Objective":
                continue
            if lp_flag is None:
                lp_flag = (tokens[0] == "LP")
                continue
            if tokens[0] == "Dual" and tokens[1] == "solution:":
                var_flag=False
                # It looks like we've just been processing primal
                # variables.
                for (var,val) in tmp:
                    if var == 'ONE_VAR_CONSTANT':
                        continue
                    soln.variable[var] = {"Value" : val}
                tmp=[]
                continue
            if len(tokens) < 3:
                print("ERROR", line,tokens)
            tmp.append( (tokens[0],eval(tokens[2])) )
        if var_flag:
            for (var,val) in tmp:
                if var == 'ONE_VAR_CONSTANT':
                    continue
                soln.variable[var] = {"Value" : val}
        else:
            range_duals = {}
            soln_constraints = soln.constraint
            if (lp_flag is True) and (extract_duals is True):
                for (var,val) in tmp:
                    if var.startswith('c_'):
                        soln_constraints[var] = {"Dual" : val}
                    elif var.startswith('r_l_'):
                        range_duals.setdefault(var[4:],[0,0])[0] = val
                    elif var.startswith('r_u_'):
                        range_duals.setdefault(var[4:],[0,0])[1] = val
            # For the range constraints, supply only the dual with the largest
            # magnitude (at least one should always be numerically zero)
            for key,(ld,ud) in iteritems(range_duals):
                if abs(ld) > abs(ud):
                    soln_constraints['r_l_'+key] = {"Dual" : ld}
                else:
                    soln_constraints['r_l_'+key] = {"Dual" : ud}        # Use the same key
        INPUT.close()


@SolverFactory.register('_mock_pico')
class MockPICO(PICOSHELL,MockMIP):
    """A Mock PICO solver used for testing
    """

    def __init__(self, **kwds):
        try:
            PICOSHELL.__init__(self,**kwds)
        except ApplicationError: #pragma:nocover
            pass                        #pragma:nocover
        MockMIP.__init__(self,"pico")

    def available(self, exception_flag=True):
        return PICOSHELL.available(self,exception_flag)

    def create_command_line(self,executable,problem_files):
        command = PICOSHELL.create_command_line(self,executable,problem_files)
        MockMIP.create_command_line(self,executable,problem_files)
        return command

    def _default_executable(self):
        return MockMIP.executable(self)

    def _execute_command(self,cmd):
        return MockMIP._execute_command(self,cmd)
