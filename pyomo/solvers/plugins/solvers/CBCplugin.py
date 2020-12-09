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
import time
import logging

from six import iteritems, string_types

from pyomo.common import Executable
from pyomo.common.errors import ApplicationError
from pyomo.common.collections import Options, Bunch
from pyomo.common.tempfiles import TempfileManager
from pyutilib.subprocess import run

from pyomo.core.kernel.block import IBlock
from pyomo.core import Var
from pyomo.opt.base import ProblemFormat, ResultsFormat, OptSolver
from pyomo.opt.base.solvers import _extract_version, SolverFactory
from pyomo.opt.results import SolverResults, SolverStatus, TerminationCondition, SolutionStatus, ProblemSense, Solution
from pyomo.opt.solver import SystemCallSolver
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
    executable = Executable("cbc")
    if not executable:
        return
    cbc_exec = executable.path()
    results = run( [cbc_exec,"-stop"], timelimit=1 )
    _cbc_version = _extract_version(results[1])
    results = run(
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
        super(CBCSHELL, self).__init__(**kwds)

        # NOTE: eventually both of the following attributes should be migrated to a common base class.
        # is the current solve warm-started? a transient data member to communicate state information
        # across the _presolve, _apply_solver, and _postsolve methods.
        self._warm_start_solve = False
        # related to the above, the temporary name of the SOLN warm-start file (if any).
        self._warm_start_file_name = None

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
        self._capabilities = Options()
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

    def warm_start_capable(self):
        if self._problem_format == ProblemFormat.cpxlp \
                and _cbc_version >= (2,8,0,0):
            return True
        else:
            return False

    def _write_soln_file(self, instance, filename):

        # Maybe this could be a useful method for any instance.

        if isinstance(instance, IBlock):
            smap = getattr(instance, "._symbol_maps")[self._smap_id]
        else:
            smap = instance.solutions.symbol_map[self._smap_id]
        byObject = smap.byObject

        column_index = 0
        with open(filename, 'w') as solnfile:
            for var in instance.component_data_objects(Var):
                # Cbc only expects integer variables with non-zero values for mipstart.
                if var.value \
                        and (var.is_integer() or var.is_binary()) \
                        and (id(var) in byObject):
                    name = byObject[id(var)]
                    solnfile.write(
                        '{} {} {}\n'.format(
                            column_index, name, var.value
                        )
                    )
                    # Cbc ignores column indexes, so the value does not matter.
                    column_index += 1

    #
    # Write a warm-start file in the SOLN format.
    #
    def _warm_start(self, instance):

        self._write_soln_file(instance, self._warm_start_file_name)

    # over-ride presolve to extract the warm-start keyword, if specified.
    def _presolve(self, *args, **kwds):

        # create a context in the temporary file manager for
        # this plugin - is "pop"ed in the _postsolve method.
        TempfileManager.push()

        # if the first argument is a string (representing a filename),
        # then we don't have an instance => the solver is being applied
        # to a file.
        self._warm_start_solve = kwds.pop('warmstart', False)
        self._warm_start_file_name = kwds.pop('warmstart_file', None)
        user_warmstart = False
        if self._warm_start_file_name is not None:
            user_warmstart = True

        # the input argument can currently be one of two things: an instance or a filename.
        # if a filename is provided and a warm-start is indicated, we go ahead and
        # create the temporary file - assuming that the user has already, via some external
        # mechanism, invoked warm_start() with a instance to create the warm start file.
        if self._warm_start_solve and \
                isinstance(args[0], string_types):
            # we assume the user knows what they are doing...
            pass
        elif self._warm_start_solve and \
                (not isinstance(args[0], string_types)):
            # assign the name of the warm start file *before* calling the base class
            # presolve - the base class method ends up creating the command line,
            # and the warm start file-name is (obviously) needed there.
            if self._warm_start_file_name is None:
                assert not user_warmstart
                self._warm_start_file_name = TempfileManager.\
                                             create_tempfile(suffix = '.cbc.soln')

        # let the base class handle any remaining keywords/actions.
        # let the base class handle any remaining keywords/actions.
        super(CBCSHELL, self)._presolve(*args, **kwds)

        # NB: we must let the base class presolve run first so that the
        # symbol_map is actually constructed!

        if (len(args) > 0) and (not isinstance(args[0], string_types)):

            if len(args) != 1:
                raise ValueError(
                    "CBCplugin _presolve method can only handle a single "
                    "problem instance - %s were supplied" % (len(args),))

            # write the warm-start file - currently only supports MIPs.
            # we only know how to deal with a single problem instance.
            if self._warm_start_solve and (not user_warmstart):

                start_time = time.time()
                self._warm_start(args[0])
                end_time = time.time()
                if self._report_timing is True:
                    print("Warm start write time=%.2f seconds" % (end_time-start_time))


    def _default_executable(self):
        executable = Executable("cbc")
        if not executable:
            logger.warning(
                "Could not locate the 'cbc' executable, which is "
                "required for solver %s" % self.name)
            self.enable = False
            return None
        return executable.path()

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
            self._log_file = TempfileManager.create_tempfile(suffix=".cbc.log")

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
            if self._warm_start_solve:
                cmd.extend(["-mipstart",self._warm_start_file_name])
            cmd.extend(["-stat=1",
                        "-solve",
                        "-solu", self._soln_file])

        return Bunch(cmd=cmd, log_file=self._log_file, env=None)

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
        optim_value = float('inf')
        lower_bound = None
        upper_bound = None
        gap = None
        nodes = None
        # See https://www.coin-or.org/Cbc/cbcuserguide.html#messages
        for line in output.split("\n"):
            tokens = tuple(re.split('[ \t]+', line.strip()))
            n_tokens = len(tokens)
            if n_tokens > 1:
                # https://projects.coin-or.org/Cbc/browser/trunk/Cbc/src/CbcSolver.cpp?rev=2497#L3769
                if n_tokens > 4 and tokens[:4] == ('Continuous', 'objective', 'value', 'is'):
                    lower_bound = float(tokens[4])
                # Search completed - best objective %g, took %d iterations and %d nodes
                elif n_tokens > 12 and tokens[1:3] == ('Search', 'completed') \
                        and tokens[4:6] == ('best', 'objective') and tokens[9] == 'iterations' \
                        and tokens[12] == 'nodes':
                    optim_value = float(tokens[6][:-1])
                    results.solver.statistics.black_box.number_of_iterations = int(tokens[8])
                    nodes = int(tokens[11])
                elif tokens[1] == 'Exiting' and n_tokens > 4:
                    if tokens[2:4] == ('on', 'maximum'):
                        results.solver.termination_condition = {'nodes': TerminationCondition.maxEvaluations,
                                                                'time': TerminationCondition.maxTimeLimit,
                                                                'solutions': TerminationCondition.other,
                                                                'iterations': TerminationCondition.maxIterations
                                                                }.get(tokens[4], TerminationCondition.other)
                    # elif tokens[2:5] == ('as', 'integer', 'gap'):
                    #     # We might want to handle this case
                # Integer solution of %g found...
                elif n_tokens >= 4 and tokens[1:4] == ('Integer', 'solution', 'of'):
                    optim_value = float(tokens[4])
                    try:
                        results.solver.statistics.black_box.number_of_iterations = \
                            int(tokens[tokens.index('iterations') - 1])
                        nodes = int(tokens[tokens.index('nodes') - 1])
                    except ValueError:
                        pass
                # Partial search - best objective %g (best possible %g), took %d iterations and %d nodes
                elif n_tokens > 15 and tokens[1:3] == ('Partial', 'search') \
                        and tokens[4:6] == ('best', 'objective') and tokens[7:9] == ('(best', 'possible') \
                        and tokens[12] == 'iterations' and tokens[15] == 'nodes':
                    optim_value = float(tokens[6])
                    lower_bound = float(tokens[9][:-2])
                    results.solver.statistics.black_box.number_of_iterations = int(tokens[11])
                    nodes = int(tokens[14])
                elif n_tokens > 12 and tokens[1] == 'After' and tokens[3] == 'nodes,' \
                        and tokens[8:10] == ('best', 'solution,') and tokens[10:12] == ('best', 'possible'):
                    nodes = int(tokens[2])
                    optim_value = float(tokens[7])
                    lower_bound = float(tokens[12])
                elif tokens[0] == "Current" and n_tokens == 10 and tokens[1] == "default" and tokens[2] == "(if" \
                        and results.problem.name is None:
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
                # https://projects.coin-or.org/Cbc/browser/trunk/Cbc/src/CbcSolver.cpp?rev=2497#L10840
                elif tokens[0] == 'Presolve':
                    if n_tokens > 9 and tokens[3] == 'rows,' and tokens[6] == 'columns':
                        results.problem.number_of_variables = int(tokens[4]) - int(tokens[5][1:-1])
                        results.problem.number_of_constraints = int(tokens[1]) - int(tokens[2][1:-1])
                        results.problem.number_of_objectives = 1
                    elif n_tokens > 6 and tokens[6] == 'infeasible':
                        soln.status = SolutionStatus.infeasible
                # https://projects.coin-or.org/Cbc/browser/trunk/Cbc/src/CbcSolver.cpp?rev=2497#L11105
                elif n_tokens > 11 and tokens[:2] == ('Problem', 'has') and tokens[3] == 'rows,' and \
                        tokens[5] == 'columns' and tokens[7:9] == ('with', 'objective)'):
                    results.problem.number_of_variables = int(tokens[4])
                    results.problem.number_of_constraints = int(tokens[2])
                    results.problem.number_of_nonzeros = int(tokens[6][1:])
                    results.problem.number_of_objectives = 1
                # https://projects.coin-or.org/Cbc/browser/trunk/Cbc/src/CbcSolver.cpp?rev=2497#L10814
                elif n_tokens > 8 and tokens[:3] == ('Original', 'problem', 'has') and tokens[4] == 'integers' \
                        and tokens[6:9] == ('of', 'which', 'binary)'):
                    results.problem.number_of_integer_variables = int(tokens[3])
                    results.problem.number_of_binary_variables = int(tokens[5][1:])
                elif n_tokens == 5 and tokens[3] == "NAME":
                    results.problem.name = tokens[4]
                elif 'CoinLpIO::readLp(): Maximization problem reformulated as minimization' in ' '.join(tokens):
                    results.problem.sense = ProblemSense.maximize
                # https://projects.coin-or.org/Cbc/browser/trunk/Cbc/src/CbcSolver.cpp?rev=2497#L3047
                elif n_tokens > 3 and tokens[:2] == ('Result', '-'):
                    if tokens[2:4] in [('Run', 'abandoned'), ('User', 'ctrl-c')]:
                        results.solver.termination_condition = TerminationCondition.userInterrupt
                    if n_tokens > 4:
                        if tokens[2:5] == ('Optimal', 'solution', 'found'):
                            # parser for log file generetated with discrete variable
                            soln.status = SolutionStatus.optimal
                            # if n_tokens > 7 and tokens[5:8] == ('(within', 'gap', 'tolerance)'):
                            #     # We might want to handle this case
                        elif tokens[2:5] in [('Linear', 'relaxation', 'infeasible'),
                                             ('Problem', 'proven', 'infeasible')]:
                            soln.status = SolutionStatus.infeasible
                        elif tokens[2:5] == ('Linear', 'relaxation', 'unbounded'):
                            soln.status = SolutionStatus.unbounded
                        elif n_tokens > 5 and tokens[2:4] == ('Stopped', 'on') and tokens[5] == 'limit':
                            results.solver.termination_condition = {'node': TerminationCondition.maxEvaluations,
                                                                    'time': TerminationCondition.maxTimeLimit,
                                                                    'solution': TerminationCondition.other,
                                                                    'iterations': TerminationCondition.maxIterations
                                                                    }.get(tokens[4], TerminationCondition.other)
                    # perhaps from https://projects.coin-or.org/Cbc/browser/trunk/Cbc/src/CbcSolver.cpp?rev=2497#L12318
                    elif n_tokens > 3 and tokens[2] == "Finished":
                        soln.status = SolutionStatus.optimal
                        optim_value = float(tokens[4])
                # https://projects.coin-or.org/Cbc/browser/trunk/Cbc/src/CbcSolver.cpp?rev=2497#L7904
                elif n_tokens >= 3 and tokens[:2] == ('Objective', 'value:'):
                    # parser for log file generetated with discrete variable
                    optim_value = float(tokens[2])
                # https://projects.coin-or.org/Cbc/browser/trunk/Cbc/src/CbcSolver.cpp?rev=2497#L7904
                elif n_tokens >= 4 and tokens[:4] == ('No', 'feasible', 'solution', 'found'):
                    soln.status = SolutionStatus.infeasible
                elif n_tokens > 2 and tokens[:2] == ('Lower', 'bound:'):
                    if lower_bound is None:  # Only use if not already found since this is to less decimal places
                        results.problem.lower_bound = float(tokens[2])
                # https://projects.coin-or.org/Cbc/browser/trunk/Cbc/src/CbcSolver.cpp?rev=2497#L7918
                elif tokens[0] == 'Gap:':
                    # This is relative and only to 2 decimal places - could calculate explicitly using lower bound
                    gap = float(tokens[1])
                # https://projects.coin-or.org/Cbc/browser/trunk/Cbc/src/CbcSolver.cpp?rev=2497#L7923
                elif n_tokens > 2 and tokens[:2] == ('Enumerated', 'nodes:'):
                    nodes = int(tokens[2])
                # https://projects.coin-or.org/Cbc/browser/trunk/Cbc/src/CbcSolver.cpp?rev=2497#L7926
                elif n_tokens > 2 and tokens[:2] == ('Total', 'iterations:'):
                    results.solver.statistics.black_box.number_of_iterations = int(tokens[2])
                # https://projects.coin-or.org/Cbc/browser/trunk/Cbc/src/CbcSolver.cpp?rev=2497#L7930
                elif n_tokens > 3 and tokens[:3] == ('Time', '(CPU', 'seconds):'):
                    results.solver.system_time = float(tokens[3])
                # https://projects.coin-or.org/Cbc/browser/trunk/Cbc/src/CbcSolver.cpp?rev=2497#L7933
                elif n_tokens > 3 and tokens[:3] == ('Time', '(Wallclock', 'Seconds):'):
                    results.solver.wallclock_time = float(tokens[3])
                # https://projects.coin-or.org/Cbc/browser/trunk/Cbc/src/CbcSolver.cpp?rev=2497#L10477
                elif n_tokens > 4 and tokens[:4] == ('Total', 'time', '(CPU', 'seconds):'):
                    results.solver.system_time = float(tokens[4])
                    if n_tokens > 7 and tokens[5:7] == ('(Wallclock', 'seconds):'):
                        results.solver.wallclock_time = float(tokens[7])
                elif tokens[0] == "Optimal":
                    if n_tokens > 4 and tokens[2] == "objective" and tokens[4] != "and":
                        # parser for log file generetated without discrete variable
                        # see pull request #339: last check avoids lines like "Optimal - objective gap and
                        # complementarity gap both smallish and small steps"
                        soln.status = SolutionStatus.optimal
                        optim_value = float(tokens[4])
                    elif n_tokens > 5 and tokens[1] == 'objective' and tokens[5] == 'iterations':
                        soln.status = SolutionStatus.optimal
                        optim_value = float(tokens[2])
                        results.solver.statistics.black_box.number_of_iterations = int(tokens[4])
                elif tokens[0] == "sys" and n_tokens == 2:
                    results.solver.system_time = float(tokens[1])
                elif tokens[0] == "user" and n_tokens == 2:
                    results.solver.user_time = float(tokens[1])
                elif n_tokens == 10 and "Presolve" in tokens and \
                        "iterations" in tokens and tokens[0] == "Optimal" and "objective" == tokens[1]:
                    soln.status = SolutionStatus.optimal
                    optim_value = float(tokens[2])
                results.solver.user_time = -1.0  # Why is this set to -1?

        if results.problem.name is None:
            results.problem.name = 'unknown'

        if soln.status is SolutionStatus.optimal:
            results.solver.termination_message = "Model was solved to optimality (subject to tolerances), and an " \
                                                 "optimal solution is available."
            results.solver.termination_condition = TerminationCondition.optimal
            results.solver.status = SolverStatus.ok
            if gap is None:
                lower_bound = optim_value
                gap = 0.0
        elif soln.status == SolutionStatus.infeasible:
            results.solver.termination_message = "Model was proven to be infeasible."
            results.solver.termination_condition = TerminationCondition.infeasible
            results.solver.status = SolverStatus.warning
        elif soln.status == SolutionStatus.unbounded:
            results.solver.termination_message = "Model was proven to be unbounded."
            results.solver.termination_condition = TerminationCondition.unbounded
            results.solver.status = SolverStatus.warning
        elif results.solver.termination_condition in [TerminationCondition.maxTimeLimit,
                                                      TerminationCondition.maxEvaluations,
                                                      TerminationCondition.other,
                                                      TerminationCondition.maxIterations]:
            results.solver.status = SolverStatus.aborted
            soln.status = SolutionStatus.stoppedByLimit
            if results.solver.termination_condition == TerminationCondition.maxTimeLimit:
                results.solver.termination_message = "Optimization terminated because the time expended " \
                                                     "exceeded the value specified in the seconds " \
                                                     "parameter."
            elif results.solver.termination_condition == TerminationCondition.maxEvaluations:
                results.solver.termination_message = \
                    "Optimization terminated because the total number of branch-and-cut nodes explored " \
                    "exceeded the value specified in the maxNodes parameter"
            elif results.solver.termination_condition == TerminationCondition.other:
                results.solver.termination_message = "Optimization terminated because the number of " \
                                                     "solutions found reached the value specified in the " \
                                                     "maxSolutions parameter."
            elif results.solver.termination_condition == TerminationCondition.maxIterations:
                results.solver.termination_message = "Optimization terminated because the total number of simplex " \
                                                     "iterations performed exceeded the value specified in the " \
                                                     "maxIterations parameter."
        soln.gap = gap
        if results.problem.sense == ProblemSense.minimize:
            upper_bound = optim_value
        elif results.problem.sense == ProblemSense.maximize:
            optim_value *= -1
            upper_bound = None if lower_bound is None else -lower_bound
            lower_bound = optim_value
        soln.objective['__default_objective__'] = {'Value': optim_value}
        results.problem.lower_bound = lower_bound
        results.problem.upper_bound = upper_bound

        results.solver.statistics.branch_and_bound.number_of_bounded_subproblems = nodes
        results.solver.statistics.branch_and_bound.number_of_created_subproblems = nodes

        if soln.status in [SolutionStatus.optimal,
                           SolutionStatus.stoppedByLimit,
                           SolutionStatus.unknown,
                           SolutionStatus.other]:
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
        else:
            solution = Solution()

        results.problem.number_of_objectives = 1

        processing_constraints = None # None means header, True means constraints, False means variables.
        header_processed = False
        optim_value = None

        try:
            INPUT = open(self._soln_file,"r")
        except IOError:
            INPUT = []

        for line in INPUT:
            tokens = tuple(re.split('[ \t]+',line.strip()))
            n_tokens = len(tokens)
            #
            # These are the only header entries CBC will generate (identified via browsing CbcSolver.cpp)
            # See https://projects.coin-or.org/Cbc/browser/trunk/Cbc/src/CbcSolver.cpp
            # Search for (no integer solution - continuous used)      Currently line 9912 as of rev2497
            # Note that since this possibly also covers old CBC versions, we shall not be removing any functionality,
            # even if it is not seen in the current revision
            #
            if not header_processed:
                if tokens[0] == 'Optimal':
                    results.solver.termination_condition = TerminationCondition.optimal
                    results.solver.status = SolverStatus.ok
                    results.solver.termination_message = "Model was solved to optimality (subject to tolerances), " \
                                                         "and an optimal solution is available."
                    solution.status = SolutionStatus.optimal
                    optim_value = float(tokens[-1])
                elif tokens[0] in ('Infeasible', 'PrimalInfeasible') or (
                        n_tokens > 1 and tokens[0:2] == ('Integer', 'infeasible')):
                    results.solver.termination_message = "Model was proven to be infeasible."
                    results.solver.termination_condition = TerminationCondition.infeasible
                    results.solver.status = SolverStatus.warning
                    solution.status = SolutionStatus.infeasible
                    INPUT.close()
                    return
                elif tokens[0] == 'Unbounded' or (
                        n_tokens > 2 and tokens[0] == 'Problem' and tokens[2] == 'unbounded') or (
                        n_tokens > 1 and tokens[0:2] == ('Dual', 'infeasible')):
                    results.solver.termination_message = "Model was proven to be unbounded."
                    results.solver.termination_condition = TerminationCondition.unbounded
                    results.solver.status = SolverStatus.warning
                    solution.status = SolutionStatus.unbounded
                    INPUT.close()
                    return
                elif n_tokens > 2 and tokens[0:2] == ('Stopped', 'on'):
                    optim_value = float(tokens[-1])
                    solution.gap = None
                    results.solver.status = SolverStatus.aborted
                    solution.status = SolutionStatus.stoppedByLimit
                    if tokens[2] == 'time':
                        results.solver.termination_message = "Optimization terminated because the time expended " \
                                                             "exceeded the value specified in the seconds " \
                                                             "parameter."
                        results.solver.termination_condition = TerminationCondition.maxTimeLimit
                    elif tokens[2] == 'iterations':
                        # Only add extra info if not already obtained from logs (which give a better description)
                        if results.solver.termination_condition not in [TerminationCondition.maxEvaluations,
                                                                        TerminationCondition.other,
                                                                        TerminationCondition.maxIterations]:
                            results.solver.termination_message = "Optimization terminated because a limit was hit"
                            results.solver.termination_condition = TerminationCondition.maxIterations
                    elif tokens[2] == 'difficulties':
                        results.solver.termination_condition = TerminationCondition.solverFailure
                        results.solver.status = SolverStatus.error
                        solution.status = SolutionStatus.error
                    elif tokens[2] == 'ctrl-c':
                        results.solver.termination_message = "Optimization was terminated by the user."
                        results.solver.termination_condition = TerminationCondition.userInterrupt
                        solution.status = SolutionStatus.unknown
                    else:
                        results.solver.termination_condition = TerminationCondition.unknown
                        results.solver.status = SolverStatus.unknown
                        solution.status = SolutionStatus.unknown
                        results.solver.termination_message = ' '.join(tokens)
                        print('***WARNING: CBC plugin currently not processing solution status=Stopped correctly. Full '
                              'status line is: {}'.format(line.strip()))
                    if n_tokens > 8 and tokens[3:9] == ('(no', 'integer', 'solution', '-', 'continuous', 'used)'):
                        results.solver.termination_message = "Optimization terminated because a limit was hit, " \
                                                             "however it had not found an integer solution yet."
                        results.solver.termination_condition = TerminationCondition.intermediateNonInteger
                        solution.status = SolutionStatus.other
                else:
                    results.solver.termination_condition = TerminationCondition.unknown
                    results.solver.status = SolverStatus.unknown
                    solution.status = SolutionStatus.unknown
                    results.solver.termination_message = ' '.join(tokens)
                    print('***WARNING: CBC plugin currently not processing solution status={} correctly. Full status '
                          'line is: {}'.format(tokens[0], line.strip()))

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
                    if optim_value is not None:
                        if results.problem.sense == ProblemSense.maximize:
                            optim_value *= -1
                        solution.objective['__default_objective__'] = {'Value': optim_value}
                    header_processed = True

            if (processing_constraints is True) and (extract_duals is True):
                if n_tokens == 4:
                    pass
                elif (n_tokens == 5) and tokens[0] == "**":
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
                if n_tokens == 4:
                    pass
                elif (n_tokens == 5) and tokens[0] == "**":
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

        if len(results.solution) == 0 and solution.status in [SolutionStatus.optimal,
                                                              SolutionStatus.stoppedByLimit,
                                                              SolutionStatus.unknown,
                                                              SolutionStatus.other]:
            results.solution.insert(solution)


    def _postsolve(self):

        # let the base class deal with returning results.
        results = super(CBCSHELL, self)._postsolve()

        # finally, clean any temporary files registered with the temp file
        # manager, created populated *directly* by this plugin. does not
        # include, for example, the execution script. but does include
        # the warm-start file.
        TempfileManager.pop(remove=not self._keepfiles)

        return results


@SolverFactory.register('_mock_cbc')
class MockCBC(CBCSHELL,MockMIP):
    """A Mock CBC solver used for testing
    """

    def __init__(self, **kwds):
        try:
            CBCSHELL.__init__(self,**kwds)
        except ApplicationError: #pragma:nocover
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

