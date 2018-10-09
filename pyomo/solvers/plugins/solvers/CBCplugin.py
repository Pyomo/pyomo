#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = ['CBC', 'MockCBC']

import os
import re
import logging

from six import iteritems

import pyomo.common
import pyutilib.misc
import pyutilib.common
import pyutilib.subprocess

from pyomo.opt.base import *
from pyomo.opt.base.solvers import _extract_version
from pyomo.opt.results import *
from pyomo.opt.solver import *
from pyomo.solvers.mockmip import MockMIP

logger = logging.getLogger('pyomo.solvers')

def _version_to_string(version):
    if version is None:
        return "<unknown>"
    return ('.'.join(str(i) for i in version))

_cbc_compiled_with_asl = None
_cbc_version = None
_cbc_old_version = None
def configure_cbc():
    global _cbc_compiled_with_asl
    global _cbc_version
    global _cbc_old_version
    if _cbc_compiled_with_asl is not None:
        return
    # manually look for the cbc executable to prevent the
    # CBC.execute() from logging an error when CBC is missing
    if pyomo.common.registered_executable("cbc") is None:
        return
    cbc_exec = pyomo.common.registered_executable("cbc").get_path()
    results = pyutilib.subprocess.run( [cbc_exec,"-stop"], timelimit=1 )
    _cbc_version = _extract_version(results[1])
    results = pyutilib.subprocess.run(
        [cbc_exec,"dummy","-AMPL","-stop"], timelimit=1 )
    _cbc_compiled_with_asl = not ('No match for AMPL' in results[1])
    if _cbc_version is not None:
        _cbc_old_version = _cbc_version < (2,7,0,0)


@SolverFactory.register('cbc', doc='The CBC LP/MIP solver')
class CBC(OptSolver):
    """The CBC LP/MIP solver
    """

    def __new__(cls, *args, **kwds):
        configure_cbc()
        try:
            mode = kwds['solver_io']
            if mode is None:
                mode = 'lp'
            del kwds['solver_io']
        except KeyError:
            mode = 'lp'
        #
        if mode  == 'lp':
            opt = SolverFactory('_cbc_shell', **kwds)
            opt.set_problem_format(ProblemFormat.cpxlp)
            return opt
        # CBC's MPS parser seems too buggy to expose
        # this option
#        if mode == 'mps':
#            # *NOTE: CBC uses the COIN-OR MPS reader,
#            #        which ignores any objective sense
#            #        declared in the OBJSENSE section.
#            opt = SolverFactory('_cbc_shell', **kwds)
#            opt.set_problem_format(ProblemFormat.mps)
#            return opt
        #
        if mode == 'nl':
            # the _cbc_compiled_with_asl and
            # _cbc_old_version flags are tristate
            # (None, False, True), so don't
            # simplify the if statements from
            # checking "is"
            if _cbc_compiled_with_asl is not False:
                if _cbc_old_version is True:
                    logger.warning("found CBC version "
                                   +_version_to_string(_cbc_version)+
                                   " < 2.7; ASL support disabled.")
                    logger.warning("Upgrade CBC to activate ASL "
                                   "support in this plugin")
            else:
                logger.warning("CBC solver is not compiled with ASL "
                               "interface.")
            # CBC doesn't not accept all asl style command line
            # options (-s in particular, which is required for
            # streaming output of all asl solvers). Therefore we need
            # to send it through the cbc_shell instead of ASL
            opt = SolverFactory('_cbc_shell',**kwds)
            opt.set_problem_format(ProblemFormat.nl)
            return opt
        elif mode == 'os':
            opt = SolverFactory('_ossolver', **kwds)
        else:
            logger.error('Unknown IO type: %s' % mode)
            return
        opt.set_options('solver=cbc')
        return opt



@SolverFactory.register('_cbc_shell',  doc='Shell interface to the CBC LP/MIP solver')
class CBCSHELL(SystemCallSolver):
    """Shell interface to the CBC LP/MIP solver
    """

    def __init__(self, **kwds):
        #
        # Call base constructor
        #
        kwds['type'] = 'cbc'
        SystemCallSolver.__init__(self, **kwds)

        #
        # Set up valid problem formats and valid results for each problem format
        #
        self._valid_problem_formats=[ProblemFormat.cpxlp, ProblemFormat.mps]
        if (_cbc_compiled_with_asl is not False) and \
           (_cbc_old_version is not True):
            self._valid_problem_formats.append(ProblemFormat.nl)
        self._valid_result_formats={}
        self._valid_result_formats[ProblemFormat.cpxlp] = [ResultsFormat.soln]
        if (_cbc_compiled_with_asl is not False) and \
           (_cbc_old_version is not True):
            self._valid_result_formats[ProblemFormat.nl] = [ResultsFormat.sol]
        self._valid_result_formats[ProblemFormat.mps] = [ResultsFormat.soln]

        # Note: Undefined capabilities default to 'None'
        self._capabilities = pyutilib.misc.Options()
        self._capabilities.linear = True
        self._capabilities.integer = True
        # The quadratic capabilities may be true but there is
        # some weirdness in the solution file that this
        # plugin does not handle correctly  (extra variables
        # added that are not in the symbol map?)
        self._capabilities.quadratic_objective = False
        self._capabilities.quadratic_constraint = False
        # These flags are updated by the set_problem_format method
        # as cbc can handle SOS constraints with the NL file format but
        # currently not through the LP file format
        self._capabilities.sos1 = False
        self._capabilities.sos2 = False

        self.set_problem_format(ProblemFormat.cpxlp)

    def set_problem_format(self, format):
        super(CBCSHELL,self).set_problem_format(format)
        if self._problem_format == ProblemFormat.cpxlp:
            self._capabilities.sos1 = False
            self._capabilities.sos2 = False
        else:
            self._capabilities.sos1 = True
            self._capabilities.sos2 = True

    def _default_results_format(self, prob_format):
        if prob_format == ProblemFormat.nl:
            return ResultsFormat.sol
        return ResultsFormat.soln

    # Nothing needs to be done here
    #def _presolve(self, *args, **kwds):
    #    # let the base class handle any remaining keywords/actions.
    #    SystemCallSolver._presolve(self, *args, **kwds)

    def _default_executable(self):
        executable = pyomo.common.registered_executable("cbc")
        if executable is None:
            logger.warning("Could not locate the 'cbc' executable, which is required for solver %s" % self.name)
            self.enable = False
            return None
        return executable.get_path()

    def _get_version(self):
        """
        Returns a tuple describing the solver executable version.
        """
        if _cbc_version is None:
            return _extract_version('')
        return _cbc_version

    def create_command_line(self, executable, problem_files):
        #
        # Define the log file
        #
        if self._log_file is None:
            self._log_file = pyutilib.services.TempfileManager.create_tempfile(suffix=".cbc.log")

        #
        # Define the solution file
        #
        # the prefix of the problem filename is required because CBC has a specific
        # and automatic convention for generating the output solution filename.
        # the extracted prefix is the same name as the input filename, e.g., minus
        # the ".lp" extension.
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
        # Define the results file (if the sol external parser is used)
        #
        # results in CBC are split across the log file (solver statistics) and
        # the solution file (solutions!)
        if self._results_format is ResultsFormat.sol:
            self._results_file = self._soln_file

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
        if self._problem_format == ProblemFormat.nl:
            cmd.append(problem_files[0])
            cmd.append('-AMPL')

            if self._timelimit is not None and self._timelimit > 0.0:
                cmd.extend(['-sec', str(self._timelimit)])
                cmd.extend(['-timeMode', "elapsed"])
            if "debug" in self.options:
                cmd.extend(["-log","5"])
            for key, val in _check_and_escape_options(self.options):
                if key == 'solver':
                    continue
                cmd.append(key+"="+val)
            os.environ['cbc_options']="printingOptions=all"
            #cmd.extend(["-printingOptions=all",
                        #"-stat"])
        else:
            if self._timelimit is not None and self._timelimit > 0.0:
                cmd.extend(['-sec', str(self._timelimit)])
                cmd.extend(['-timeMode', "elapsed"])
            if "debug" in self.options:
                cmd.extend(["-log","5"])
            # these must go after options that take a value
            action_options = []
            for key, val in _check_and_escape_options(self.options):
                if val.strip() != '':
                    cmd.extend(['-'+key, val])
                else:
                    action_options.append('-'+key)
            cmd.extend(["-printingOptions", "all",
                        "-import", problem_files[0]])
            cmd.extend(action_options)
            cmd.extend(["-stat=1",
                        "-solve",
                        "-solu", self._soln_file])

        return pyutilib.misc.Bunch(cmd=cmd, log_file=self._log_file, env=None)

    def process_logfile(self):
        """
        Process logfile
        """
        results = SolverResults()

        # The logfile output for cbc when using nl files
        # provides no information worth parsing here
        if self._problem_format is ProblemFormat.nl:
            return results

        #
        # Initial values
        #
        soln = Solution()
        soln.objective['__default_objective__'] = {'Value': float('inf')}
        #
        # Process logfile
        #
        OUTPUT = open(self._log_file)
        output = "".join(OUTPUT.readlines())
        OUTPUT.close()
        #
        # Parse logfile lines
        #
        results.problem.sense = ProblemSense.minimize
        results.problem.name = None
        for line in output.split("\n"):
            tokens = re.split('[ \t]+',line.strip())
            if len(tokens) == 10 and tokens[0] == "Current" and tokens[1] == "default" and tokens[2] == "(if" and results.problem.name is None:
                results.problem.name = tokens[-1]
                if '.' in results.problem.name:
                    parts = results.problem.name.split('.')
                    if len(parts) > 2:
                        results.problem.name = '.'.join(parts[:-1])
                    else:
                        results.problem.name = results.problem.name.split('.')[0]
                if '/' in results.problem.name:
                    results.problem.name = results.problem.name.split('/')[-1]
                if '\\' in results.problem.name:
                    results.problem.name = results.problem.name.split('\\')[-1]
            if len(tokens) ==11 and tokens[0] == "Presolve" and tokens[3] == "rows,":
                results.problem.number_of_variables = eval(tokens[4])-eval(tokens[5][1:-1])
                results.problem.number_of_constraints = eval(tokens[1])-eval(tokens[2][1:-1])
                results.problem.number_of_nonzeros = eval(tokens[8])-eval(tokens[9][1:-1])
                results.problem.number_of_objectives = "1"
            if len(tokens) >=9 and tokens[0] == "Problem" and tokens[2] == "has":
                results.problem.number_of_variables = eval(tokens[5])
                results.problem.number_of_constraints = eval(tokens[3])
                results.problem.number_of_nonzeros = eval(tokens[8])
                results.problem.number_of_objectives = "1"
            if len(tokens) == 5 and tokens[3] == "NAME":
                results.problem.name = tokens[4]
            if " ".join(tokens) == '### WARNING: CoinLpIO::readLp(): Maximization problem reformulated as minimization':
                results.problem.sense = ProblemSense.maximize
            if len(tokens) > 6 and tokens[0] == "Presolve" and tokens[6] == "infeasible":
                soln.status = SolutionStatus.infeasible
                soln.objective['__default_objective__']['Value'] = None
            if len(tokens) > 3 and tokens[0] == "Result" and tokens[2] == "Optimal":
                # parser for log file generetated with discrete variable
                soln.status = SolutionStatus.optimal
            if len(tokens) >= 3 and tokens[0] == "Objective" and tokens[1] == "value:":
                # parser for log file generetated with discrete variable
                soln.objective['__default_objective__']['Value'] = float(tokens[2])
            if len(tokens) > 4 and tokens[0] == "Optimal" and tokens[2] == "objective" and tokens[4] != "and":
                # parser for log file generetated without discrete variable
                # see pull request #339: last check avoids lines like "Optimal - objective gap and complementarity gap both smallish and small steps"
                soln.status = SolutionStatus.optimal
                soln.objective['__default_objective__']['Value'] = float(tokens[4])
            if len(tokens) > 6 and tokens[4] == "best" and tokens[5] == "objective":
                if tokens[6].endswith(','):
                    tokens[6] = tokens[6][:-1]
                soln.objective['__default_objective__']['Value'] = float(tokens[6])
            if len(tokens) > 9 and tokens[7] == "(best" and tokens[8] == "possible":
                results.problem.lower_bound=tokens[9]
                results.problem.lower_bound = eval(results.problem.lower_bound.split(")")[0])
            if len(tokens) > 12 and tokens[10] == "best" and tokens[11] == "possible":
                results.problem.lower_bound=eval(tokens[12])
            if len(tokens) > 3 and tokens[0] == "Result" and tokens[2] == "Finished":
                soln.status = SolutionStatus.optimal
                soln.objective['__default_objective__']['Value'] = float(tokens[4])
            if len(tokens) > 10 and tokens[4] == "time" and tokens[9] == "nodes":
                results.solver.statistics.branch_and_bound.number_of_created_subproblems=eval(tokens[8])
                results.solver.statistics.branch_and_bound.number_of_bounded_subproblems=eval(tokens[8])
                if eval(results.solver.statistics.branch_and_bound.number_of_bounded_subproblems) > 0:
                    soln.objective['__default_objective__']['Value'] = float(tokens[6])
            if len(tokens) == 5 and tokens[1] == "Exiting" and tokens[4] == "time":
                soln.status = SolutionStatus.stoppedByLimit
            if len(tokens) > 8 and tokens[7] == "nodes":
                results.solver.statistics.branch_and_bound.number_of_created_subproblems=eval(tokens[6])
                results.solver.statistics.branch_and_bound.number_of_bounded_subproblems=eval(tokens[6])
            if len(tokens) == 2 and tokens[0] == "sys":
                results.solver.system_time=float(tokens[1])
            if len(tokens) == 2 and tokens[0] == "user":
                results.solver.user_time=float(tokens[1])
            if len(tokens) == 10 and "Presolve" in tokens and  \
               "iterations" in tokens and tokens[0] == "Optimal" and "objective" == tokens[1]:
                soln.status = SolutionStatus.optimal
                soln.objective['__default_objective__']['Value'] = float(tokens[2])
            results.solver.user_time=-1.0

        if soln.objective['__default_objective__']['Value'] == "1e+50":
            if results.problem.sense == ProblemSense.minimize:
                soln.objective['__default_objective__']['Value'] = float('inf')
            else:
                soln.objective['__default_objective__']['Value'] = float('-inf')
        elif results.problem.sense == ProblemSense.maximize and soln.status != SolutionStatus.infeasible:
            soln.objective['__default_objective__']['Value'] *= -1
        if soln.status is SolutionStatus.optimal:
            soln.gap=0.0
            results.problem.lower_bound = soln.objective['__default_objective__']['Value']
            results.problem.upper_bound = soln.objective['__default_objective__']['Value']

        if soln.status == SolutionStatus.optimal:
            results.solver.termination_condition = TerminationCondition.optimal
        elif soln.status == SolutionStatus.infeasible:
            results.solver.termination_condition = TerminationCondition.infeasible

        if results.problem.name is None:
            results.problem.name = 'unknown'

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

    def process_soln_file(self, results):
        # the only suffixes that we extract from CBC are
        # constraint duals and variable reduced-costs. scan
        # through the solver suffix list and throw an
        # exception if the user has specified any others.
        extract_duals = False
        extract_reduced_costs = False
        for suffix in self._suffixes:
            flag=False
            if re.match(suffix, "dual"):
                extract_duals = True
                flag=True
            if re.match(suffix, "rc"):
                extract_reduced_costs = True
                flag=True
            if not flag:
                raise RuntimeError("***CBC solver plugin cannot extract solution suffix="+suffix)

        # if dealing with SOL format files, we've already read
        # this via the base class reader functionality.
        if self._results_format is ResultsFormat.sol:
            return

        # otherwise, go with the native CBC solution format.
        if len(results.solution) > 0:
            solution = results.solution(0)
        if results.solver.termination_condition is TerminationCondition.infeasible:
            # NOTE: CBC _does_ print a solution file.  However, I'm not
            # sure how to interpret it yet.
            return
        results.problem.number_of_objectives=1

        processing_constraints = None # None means header, True means constraints, False means variables.
        header_processed = False
        optim_value = None

        try:
            INPUT = open(self._soln_file,"r")
        except IOError:
            INPUT = []

        for line in INPUT:
            tokens = re.split('[ \t]+',line.strip())

            #
            # These are the only header entries CBC will generate (identified via browsing CbcSolver.cpp)
            #
            if not header_processed:
                if tokens[0] == "Optimal":
                    solution.status = SolutionStatus.optimal
                    solution.gap = 0.0
                    optim_value = float(tokens[-1])

                elif tokens[0] == "Unbounded" or (len(tokens)>2 and tokens[0] == "Problem" and tokens[2] == 'unbounded') or (len(tokens)>1 and tokens[0] ==    'Dual' and tokens[1] == 'infeasible'):
                    results.solver.termination_condition = TerminationCondition.unbounded
                    solution.gap = None
                    results.solution.delete(0)
                    INPUT.close()
                    return

                elif tokens[0] == "Infeasible" or tokens[0] == 'PrimalInfeasible' or (len(tokens)>1 and tokens[0] == 'Integer' and tokens[1] == 'infeasible'):
                    results.solver.termination_condition = TerminationCondition.infeasible
                    solution.gap = None
                    results.solution.delete(0)
                    INPUT.close()
                    return

                elif len(tokens) > 2 and tokens[0:3] == ['Stopped','on','time']:
                    if tokens[4] == 'objective':
                        results.solver.termination_condition = TerminationCondition.maxTimeLimit
                        solution.gap = None
                        optim_value = float(tokens[-1])
                    elif len(tokens) > 8 and tokens[3:9] == ['(no', 'integer', 'solution', '-', 'continuous', 'used)']:
                        results.solver.termination_condition = TerminationCondition.intermediateNonInteger
                        solution.gap = None
                        optim_value = float(tokens[-1])
                    else:
                        print("***WARNING: CBC plugin currently not processing this solution status correctly. Full status line is: "+line.strip())

                elif tokens[0] in ("Optimal", "Infeasible", "Unbounded", "Stopped", "Integer", "Status"):
                    print("***WARNING: CBC plugin currently not processing solution status="+tokens[0]+" correctly. Full status line is: "+line.strip())

            # most of the first tokens should be integers
            # if it's not an integer, only then check the list of results
            try:
                row_number = int( tokens[0])
                if row_number == 0: # indicates section start.
                    if processing_constraints is None:
                        processing_constraints = True
                    elif processing_constraints is True:
                        processing_constraints = False
                    else:
                        raise RuntimeError("CBC plugin encountered unexpected line=("+line.strip()+") in solution file="+self._soln_file+"; constraint and variable sections already processed!")
            except ValueError:
                if tokens[0] in ("Optimal", "Infeasible", "Unbounded", "Stopped", "Integer", "Status"):
                    if optim_value:
                        solution.objective['__default_objective__']['Value'] = optim_value
                        if results.problem.sense == ProblemSense.maximize:
                            solution.objective['__default_objective__']['Value'] *= -1
                    header_processed = True

            if (processing_constraints is True) and (extract_duals is True):
                if len(tokens) == 4:
                    pass
                elif (len(tokens) == 5) and tokens[0] == "**":
                    tokens = tokens[1:]
                else:
                    raise RuntimeError("Unexpected line format encountered in CBC solution file - line="+line)

                constraint = tokens[1]
                constraint_ax = float(tokens[2]) # CBC reports the constraint row times the solution vector - not the slack.
                constraint_dual = float(tokens[3])
                if constraint[:2] == 'c_':
                    solution.constraint[constraint] = {"Dual" : constraint_dual}
                elif constraint[:2] == 'r_':
                    # For the range constraints, supply only the dual with the largest
                    # magnitude (at least one should always be numerically zero)
                    existing_constraint_dual_dict = solution.constraint.get( 'r_l_' + constraint[4:], None )
                    if existing_constraint_dual_dict:
                        # if a constraint dual is already saved, then update it if its
                        # magnitude is larger than existing; this avoids checking vs
                        # zero (avoiding problems with solver tolerances)
                        existing_constraint_dual = existing_constraint_dual_dict["Dual"]
                        if abs( constraint_dual) > abs(existing_constraint_dual):
                            solution.constraint[ 'r_l_' + constraint[4:] ] = {"Dual": constraint_dual}
                    else:
                        # if no constraint with that name yet, just save it in the solution constraint dictionary
                        solution.constraint[ 'r_l_' + constraint[4:] ] = {"Dual": constraint_dual}

            elif processing_constraints is False:
                if len(tokens) == 4:
                    pass
                elif (len(tokens) == 5) and tokens[0] == "**":
                    tokens = tokens[1:]
                else:
                    raise RuntimeError("Unexpected line format encountered "
                                       "in CBC solution file - line="+line)

                variable_name = tokens[1]
                variable_value = float(tokens[2])
                variable = solution.variable[variable_name] = {"Value" : variable_value}
                if extract_reduced_costs is True:
                    variable_reduced_cost = float(tokens[3]) # currently ignored.
                    variable["Rc"] = variable_reduced_cost

            elif header_processed is True:
                pass

            else:
                raise RuntimeError("CBC plugin encountered unexpected "
                                   "line=("+line.strip()+") in solution file="
                                   +self._soln_file+"; expecting header, but "
                                   "found data!")

        if not type(INPUT) is list:
            INPUT.close()


@SolverFactory.register('_mock_cbc')
class MockCBC(CBCSHELL,MockMIP):
    """A Mock CBC solver used for testing
    """

    def __init__(self, **kwds):
        try:
            CBCSHELL.__init__(self,**kwds)
        except pyutilib.common.ApplicationError: #pragma:nocover
            pass                        #pragma:nocover
        MockMIP.__init__(self,"cbc")

    def available(self, exception_flag=True):
        return CBCSHELL.available(self,exception_flag)

    def create_command_line(self,executable,problem_files):
        command = CBCSHELL.create_command_line(self,executable,problem_files)
        MockMIP.create_command_line(self,executable,problem_files)
        return command

    def executable(self):
        return MockMIP.executable(self)

    def _execute_command(self,cmd):
        return MockMIP._execute_command(self,cmd)

    def _convert_problem(self,args,pformat,valid_pformats):
        if pformat in [ProblemFormat.mps, ProblemFormat.cpxlp, ProblemFormat.nl]:
            return (args, pformat, None)
        else:
            return (args, ProblemFormat.mps, None)


pyomo.common.register_executable(name="cbc")
