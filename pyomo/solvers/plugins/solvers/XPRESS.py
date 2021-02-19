#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


import os
import re
import logging

from pyomo.common import Executable
from pyomo.common.errors import ApplicationError
from pyomo.common.collections import Options, Bunch
from pyutilib.misc import yaml_fix
from pyomo.common.tempfiles import TempfileManager

from pyomo.opt.base import ProblemFormat, ResultsFormat, OptSolver
from pyomo.opt.base.solvers import _extract_version, SolverFactory
from pyomo.opt.results import SolverResults, SolverStatus, TerminationCondition, ProblemSense, Solution
from pyomo.opt.solver import ILMLicensedSystemCallSolver
from pyomo.solvers.mockmip import MockMIP

logger = logging.getLogger('pyomo.solvers')


@SolverFactory.register('xpress', doc='The XPRESS LP/MIP solver')
class XPRESS(OptSolver):
    """The XPRESS LP/MIP solver
    """

    def __new__(cls, *args, **kwds):
        try:
            mode = kwds['solver_io']
            if mode is None:
                mode = 'lp'
            del kwds['solver_io']
        except KeyError:
            mode = 'lp'

        if mode  == 'lp':
            return SolverFactory('_xpress_shell', **kwds)
        elif mode  == 'mps':
            opt = SolverFactory('_xpress_shell', **kwds)
            opt.set_problem_format(ProblemFormat.mps)
            return opt
        elif mode == 'nl':
            opt = SolverFactory('asl', **kwds)
        elif mode in ['python', 'direct']:
            opt = SolverFactory('xpress_direct', **kwds)
            if opt is None:
                logger.error('Python API for XPRESS is not installed')
                return
            return opt
        elif mode == 'persistent':
            opt = SolverFactory('xpress_persistent', **kwds)
            if opt is None:
                logger.error('Python API for XPRESS is not installed')
                return
            return opt
        else:
            logging.getLogger('pyomo.solvers').error(
                'Unknown IO type for solver xpress: %s'
                % (mode))
            return
        opt.set_options('solver=amplxpress')
        return opt


@SolverFactory.register('_xpress_shell', doc='Shell interface to the XPRESS LP/MIP solver')
class XPRESS_shell(ILMLicensedSystemCallSolver):
    """Shell interface to the XPRESS LP/MIP solver
    """

    def __init__(self, **kwds):
        #
        # Call base class constructor
        #
        kwds['type'] = 'xpress'
        ILMLicensedSystemCallSolver.__init__(self, **kwds)

        self.is_mip = kwds.pop('is_mip', False)

        #
        # Define valid problem formats and associated results formats
        #
        self._valid_problem_formats=[ProblemFormat.cpxlp, ProblemFormat.mps]
        self._valid_result_formats={}
        self._valid_result_formats[ProblemFormat.cpxlp] = [ResultsFormat.soln]
        self._valid_result_formats[ProblemFormat.mps] = [ResultsFormat.soln]
        self.set_problem_format(ProblemFormat.cpxlp)

        #
        # Cache the problem type - LP or MIP. Xpress needs to know this
        # on the command-line, and it matters when reading the solution file.
        #

        # Note: Undefined capabilities default to 'None'
        self._capabilities = Options()
        self._capabilities.linear = True
        self._capabilities.quadratic_objective = True
        self._capabilities.quadratic_constraint = True
        self._capabilities.integer = True
        self._capabilities.sos1 = True
        self._capabilities.sos2 = True

    def _default_results_format(self, prob_format):
        return ResultsFormat.soln

    #
    # we haven't reached this point quite yet.
    #
    def warm_start_capable(self):

        return False

    def _default_executable(self):
        executable = Executable("optimizer")
        if not executable:
            logger.warning("Could not locate the 'optimizer' executable, "
                           "which is required for solver %s" % self.name)
            self.enable = False
            return None
        return executable.path()

    # TODO: If anyone can get their hands on a working 'optimizer' executable
    #       we could add a custom version method
    def _get_version(self):
        """
        Returns a tuple describing the solver executable version.
        """
        return _extract_version('')

    def create_command_line(self, executable, problem_files):

        #
        # Define log file
        # The log file in XPRESS contains the solution trace, but the solver status can be found in the solution file.
        #
        if self._log_file is None:
            self._log_file = TempfileManager.\
                            create_tempfile(suffix = '.xpress.log')

        #
        # Define solution file
        # As indicated above, contains (in XML) both the solution and solver status.
        #
        self._soln_file = TempfileManager.\
                          create_tempfile(suffix = '.xpress.wrtsol')

        #
        # Write the XPRESS execution script
        #
        script = ""

        script = "setlogfile %s\n" % (self._log_file,)

        if self._timelimit is not None and self._timelimit > 0.0:
            script += "maxtime=%s\n" % (self._timelimit,)

        if (self.options.mipgap is not None) and (self.options.mipgap > 0.0):
            mipgap = self.options.pop('mipgap')
            script += "miprelstop=%s\n" % (mipgap,)

        for option_name in self.options:
            script += "%s=%s\n" % (option_name, self.options[option_name])

        script += "readprob %s\n" % (problem_files[0],)

        # doesn't seem to be a global solve command for mip versus lp
        # solves
        if self.is_mip:
            script += "mipoptimize\n"
        else:
            script += "lpoptimize\n"

        # a quick explanation of the various flags used below:
        # p: outputs in full precision
        # n: output the name
        # t: output the type
        # a: output the activity (value)
        # c: outputs the costs for variables, slacks for constraints.
        # d: outputs the reduced costs for columns, duals for constraints
        script += "writesol %s -pnatcd\n" % (self._soln_file,)

        script += "quit\n"

        # dump the script and warm-start file names for the
        # user if we're keeping files around.
        if self._keepfiles:
            script_fname = TempfileManager.create_tempfile(suffix = '.xpress.script')
            tmp = open(script_fname,'w')
            tmp.write(script)
            tmp.close()

            print("Solver script file=" + script_fname)

        #
        # Define command line
        #
        cmd = [executable]
        if self._timer:
            cmd.insert(0, self._timer)
        return Bunch(cmd=cmd, script=script,
                                   log_file=self._log_file, env=None)

    def process_logfile(self):

        results = SolverResults()
        results.problem.number_of_variables = None
        results.problem.number_of_nonzeros = None

        log_file = open(self._log_file)
        log_file_contents = "".join(log_file.readlines())
        log_file.close()

        for line in log_file_contents.split("\n"):
            tokens = re.split('[ \t]+',line.strip())

            if len(tokens) > 3 and tokens[0] == "XPRESS" and tokens[1] == "Error":
            # IMPT: See below - cplex can generate an error line and then terminate fine, e.g., in XPRESS 12.1.
            #       To handle these cases, we should be specifying some kind of termination criterion always
            #       in the course of parsing a log file (we aren't doing so currently - just in some conditions).
                results.solver.status=SolverStatus.error
                results.solver.error = " ".join(tokens)
            elif len(tokens) >= 3 and tokens[0] == "ILOG" and tokens[1] == "XPRESS":
                cplex_version = tokens[2].rstrip(',')
            elif len(tokens) >= 3 and tokens[0] == "Variables":
                if results.problem.number_of_variables is None: # XPRESS 11.2 and subsequent versions have two Variables sections in the log file output.
                    results.problem.number_of_variables = int(tokens[2])
            # In XPRESS 11 (and presumably before), there was only a single line output to
            # indicate the constriant count, e.g., "Linear constraints : 16 [Less: 7, Greater: 6, Equal: 3]".
            # In XPRESS 11.2 (or somewhere in between 11 and 11.2 - I haven't bothered to track it down
            # in that detail), there is another instance of this line prefix in the min/max problem statistics
            # block - which we don't care about. In this case, the line looks like: "Linear constraints :" and
            # that's all.
            elif len(tokens) >= 4 and tokens[0] == "Linear" and tokens[1] == "constraints":
                results.problem.number_of_constraints = int(tokens[3])
            elif len(tokens) >= 3 and tokens[0] == "Nonzeros":
                if results.problem.number_of_nonzeros is None: # XPRESS 11.2 and subsequent has two Nonzeros sections.
                    results.problem.number_of_nonzeros = int(tokens[2])
            elif len(tokens) >= 5 and tokens[4] == "MINIMIZE":
                results.problem.sense = ProblemSense.minimize
            elif len(tokens) >= 5 and tokens[4] == "MAXIMIZE":
                results.problem.sense = ProblemSense.maximize
            elif len(tokens) >= 4 and tokens[0] == "Solution" and tokens[1] == "time" and tokens[2] == "=":
                # technically, I'm not sure if this is XPRESS user time or user+system - XPRESS doesn't appear
                # to differentiate, and I'm not sure we can always provide a break-down.
                results.solver.user_time = float(tokens[3])
            elif len(tokens) >= 4 and tokens[0] == "Dual" and tokens[1] == "simplex" and tokens[3] == "Optimal:":
                results.solver.termination_condition = TerminationCondition.optimal
                results.solver.termination_message = ' '.join(tokens)
            elif len(tokens) >= 4 and tokens[0] == "Barrier" and tokens[2] == "Optimal:":
                results.solver.termination_condition = TerminationCondition.optimal
                results.solver.termination_message = ' '.join(tokens)
            elif len(tokens) >= 4 and tokens[0] == "Dual" and tokens[3] == "Infeasible:":
                results.solver.termination_condition = TerminationCondition.infeasible
                results.solver.termination_message = ' '.join(tokens)
            elif len(tokens) >= 4 and tokens[0] == "MIP" and tokens[2] == "Integer" and tokens[3] == "infeasible.":
                # if XPRESS has previously printed an error message, reduce it to a warning -
                # there is a strong indication it recovered, but we can't be sure.
                if results.solver.status == SolverStatus.error:
                    results.solver.status = SolverStatus.warning
                else:
                    results.solver.status = SolverStatus.ok
                results.solver.termination_condition = TerminationCondition.infeasible
                results.solver.termination_message = ' '.join(tokens)
            # for the case below, XPRESS sometimes reports "true" optimal (the first case)
            # and other times within-tolerance optimal (the second case).
            elif (len(tokens) >= 4 and tokens[0] == "MIP" and tokens[2] == "Integer" and tokens[3] == "optimal") or \
                 (len(tokens) >= 4 and tokens[0] == "MIP" and tokens[2] == "Integer" and tokens[3] == "optimal,"):
                # if XPRESS has previously printed an error message, reduce it to a warning -
                # there is a strong indication it recovered, but we can't be sure.
                if results.solver.status == SolverStatus.error:
                    results.solver.status = SolverStatus.warning
                else:
                    results.solver.status = SolverStatus.ok
                results.solver.termination_condition = TerminationCondition.optimal
                results.solver.termination_message = ' '.join(tokens)
            elif len(tokens) >= 3 and tokens[0] == "Presolve" and tokens[2] == "Infeasible.":
                # if XPRESS has previously printed an error message, reduce it to a warning -
                # there is a strong indication it recovered, but we can't be sure.
                if results.solver.status == SolverStatus.error:
                    results.solver.status = SolverStatus.warning
                else:
                    results.solver.status = SolverStatus.ok
                results.solver.termination_condition = TerminationCondition.infeasible
                results.solver.termination_message = ' '.join(tokens)
            elif (len(tokens) == 6 and tokens[2] == "Integer" and tokens[3] == "infeasible" and tokens[5] == "unbounded.") or (len(tokens) >= 5 and tokens[0] == "Presolve" and tokens[2] == "Unbounded" and tokens[4] == "infeasible."):
                # if XPRESS has previously printed an error message, reduce it to a warning -
                # there is a strong indication it recovered, but we can't be sure.
                if results.solver.status == SolverStatus.error:
                    results.solver.status = SolverStatus.warning
                else:
                    results.solver.status = SolverStatus.ok
                # It isn't clear whether we can determine if the problem is unbounded from
                # XPRESS's output.
                results.solver.termination_condition = TerminationCondition.unbounded
                results.solver.termination_message = ' '.join(tokens)

        try:
            results.solver.termination_message = yaml_fix(results.solver.termination_message)
        except:
            pass
        return results

    def process_soln_file(self, results):

        # the only suffixes that we extract from Xpress are
        # constraint duals, constraint slacks, and variable
        # reduced-costs. scan through the solver suffix list
        # and throw an exception if the user has specified
        # any others.
        extract_duals = False
        extract_slacks = False
        extract_reduced_costs = False
        extract_rc = False
        extract_lrc = False
        extract_urc = False
        for suffix in self._suffixes:
            flag=False
            if re.match(suffix,"dual"):
                extract_duals = True
                flag=True
            if re.match(suffix,"slack"):
                extract_slacks = True
                flag=True
            if re.match(suffix,"rc"):
                extract_reduced_costs = True
                extract_rc = True
                flag=True
            if re.match(suffix,"lrc"):
                extract_reduced_costs = True
                extract_lrc = True
                flag=True
            if re.match(suffix,"urc"):
                extract_reduced_costs = True
                extract_urc = True
                flag=True
            if not flag:
                raise RuntimeError("***The xpress solver plugin cannot extract solution suffix="+suffix)

        if not os.path.exists(self._soln_file):
            return

        soln = Solution()
        soln.objective['__default_objective__'] = {'Value': None} # TBD: NOT SURE HOW TO EXTRACT THE OBJECTIVE VALUE YET!
        soln_variable = soln.variable # caching for efficiency
        solution_file = open(self._soln_file, "r")
        results.problem.number_of_objectives=1

        for line in solution_file:

            line = line.strip()
            tokens=line.split(',')

            name = tokens[0].strip("\" ")
            type = tokens[1].strip("\" ")

            primary_value = float(tokens[2].strip("\" "))
            secondary_value = float(tokens[3].strip("\" "))
            tertiary_value = float(tokens[4].strip("\" "))

            if type == "C": # a 'C' type in Xpress is a variable (i.e., column) - everything else is a constraint.

                variable_name = name
                variable_value = primary_value
                variable_reduced_cost = None
                ### TODO: What is going on here with field_name, and shortly thereafter, with variable_status and whatnot? They've never been defined. 
                ### It seems like this is trying to mimic the CPLEX solver but has some issues
                if (extract_reduced_costs is True) and (field_name == "reducedCost"):
                    variable_reduced_cost = tertiary_value

                if variable_name != "ONE_VAR_CONSTANT":
                    variable = soln_variable[variable_name] = {"Value" : float(variable_value)}
                    if (variable_reduced_cost is not None) and (extract_reduced_costs is True):
                        try:
                            if extract_rc is True:
                                variable["Rc"] = float(variable_reduced_cost)
                            if variable_status is not None:
                                if extract_lrc is True:
                                    if variable_status == "LL":
                                        variable["Lrc"] = float(variable_reduced_cost)
                                    else:
                                        variable["Lrc"] = 0.0
                                if extract_urc is True:
                                    if variable_status == "UL":
                                        variable["Urc"] = float(variable_reduced_cost)
                                    else:
                                        variable["Urc"] = 0.0
                        except:
                            raise ValueError("Unexpected reduced-cost value="
                                             +str(variable_reduced_cost)+
                                             " encountered for variable="+variable_name)

            else:

                constraint = soln.constraint[name] = {}

                if (extract_duals is True) and (tertiary_value != 0.0):
                    constraint["Dual"] = tertiary_value
                if (extract_slacks is True) and (secondary_value != 0.0):
                    constraint["Slack"] = secondary_value

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
        solution_file.close()


@SolverFactory.register('_mock_xpress')
class MockXPRESS(XPRESS_shell,MockMIP):
    """A Mock XPRESS solver used for testing
    """

    def __init__(self, **kwds):
        try:
            XPRESS_shell.__init__(self, **kwds)
        except ApplicationError: #pragma:nocover
            pass                                 #pragma:nocover
        MockMIP.__init__(self,"cplex")

    def available(self, exception_flag=True):
        return XPRESS_shell.available(self,exception_flag)

    def create_command_line(self,executable,problem_files):
        command = XPRESS_shell.create_command_line(self,executable,problem_files)
        MockMIP.create_command_line(self,executable,problem_files)
        return command

    def executable(self):
        return MockMIP.executable(self)

    def _execute_command(self,cmd):
        return MockMIP._execute_command(self,cmd)

