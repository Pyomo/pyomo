#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________


import os
import re
import time
import logging

import pyutilib.services
import pyutilib.common
import pyutilib.misc

import pyomo.util.plugin
from pyomo.opt.base import *
from pyomo.opt.base.solvers import _extract_version
from pyomo.opt.results import *
from pyomo.opt.solver import *
from pyomo.solvers.mockmip import MockMIP

logger = logging.getLogger('pyomo.solvers')

from six import iteritems
from six.moves import xrange

try:
    unicode
except:
    basestring = unicode = str

class CPLEX(OptSolver):
    """The CPLEX LP/MIP solver
    """

    pyomo.util.plugin.alias('cplex', doc='The CPLEX LP/MIP solver')

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
            return SolverFactory('_cplex_shell', **kwds)
        if mode == 'mps':
            opt = SolverFactory('_cplex_shell', **kwds)
            opt.set_problem_format(ProblemFormat.mps)
            return opt
        if mode == 'python':
            opt = SolverFactory('_cplex_direct', **kwds)
            if opt is None:
                logging.getLogger('pyomo.solvers').error('Python API for CPLEX is not installed')
                return
            return opt
        #
        if mode == 'os':
            opt = SolverFactory('_ossolver', **kwds)
        elif mode == 'nl':
            opt = SolverFactory('asl', **kwds)
        else:
            logging.getLogger('pyomo.solvers').error('Unknown IO type: %s' % mode)
            return
        opt.set_options('solver=cplexamp')
        return opt


class CPLEXSHELL(ILMLicensedSystemCallSolver):
    """Shell interface to the CPLEX LP/MIP solver
    """

    pyomo.util.plugin.alias('_cplex_shell', doc='Shell interface to the CPLEX LP/MIP solver')

    def __init__(self, **kwds):
        #
        # Call base class constructor
        #
        kwds['type'] = 'cplex'
        ILMLicensedSystemCallSolver.__init__(self, **kwds)

        # NOTE: eventually both of the following attributes should be migrated to a common base class.
        # is the current solve warm-started? a transient data member to communicate state information
        # across the _presolve, _apply_solver, and _postsolve methods.
        self._warm_start_solve = False
        # related to the above, the temporary name of the MST warm-start file (if any).
        self._warm_start_file_name = None

        #
        # Define valid problem formats and associated results formats
        #
        self._valid_problem_formats=[ProblemFormat.cpxlp, ProblemFormat.mps]
        self._valid_result_formats={}
        self._valid_result_formats[ProblemFormat.cpxlp] = [ResultsFormat.soln]
        self._valid_result_formats[ProblemFormat.mps] = [ResultsFormat.soln]
        self.set_problem_format(ProblemFormat.cpxlp)

        # Note: Undefined capabilities default to 'None'
        self._capabilities = pyutilib.misc.Options()
        self._capabilities.linear = True
        self._capabilities.quadratic_objective = True
        self._capabilities.quadratic_constraint = True
        self._capabilities.integer = True
        self._capabilities.sos1 = True
        self._capabilities.sos2 = True

    def _default_results_format(self, prob_format):
        return ResultsFormat.soln

    #
    # CPLEX has a simple, easy-to-use warm-start capability.
    #
    def warm_start_capable(self):
        return True

    #
    # write a warm-start file in the CPLEX MST format.
    #
    def _warm_start(self, instance):

        from pyomo.core.base import Var

        # in principle, one could use a Python XML writer library like
        # xml.dom.minidom.  it works, but it is slow. hence, the
        # explicit direct-write of XML below.

        mst_file = open(self._warm_start_file_name, "w")

        mst_file.write("<?xml version=\"1.0\" ?>\n")
        mst_file.write("<CPLEXSolution version=\"1.0\">\n")
        mst_file.write("<header/>\n")
        mst_file.write("<quality/>\n")
        mst_file.write("<variables>\n")

        # for each variable in the symbol_map, add a child to the
        # variables element.  Both continuous and discrete are accepted
        # (and required, depending on other options), according to the
        # CPLEX manual.
        # **Note**: This assumes that the symbol_map is "clean", i.e.,
        # contains only references to the variables encountered in constraints
        output_index = 0
        smap = instance.solutions.symbol_map[self._smap_id]
        byObject = smap.byObject
        for var in instance.component_data_objects(Var, active=True):
            if (var.value is not None) and (id(var) in byObject):
                name = byObject[id(var)]
                mst_file.write("<variable index=\"%d\" name=\"%s\" value=\"%f\" />\n" % (output_index, name, var.value))
                output_index = output_index + 1

        mst_file.write("</variables>\n")
        mst_file.write("</CPLEXSolution>\n")
        mst_file.close()

    # over-ride presolve to extract the warm-start keyword, if specified.
    def _presolve(self, *args, **kwds):

        # create a context in the temporary file manager for
        # this plugin - is "pop"ed in the _postsolve method.
        pyutilib.services.TempfileManager.push()

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
           isinstance(args[0], basestring):
            # we assume the user knows what they are doing...
            pass
        elif self._warm_start_solve and \
             (not isinstance(args[0], basestring)):
            # assign the name of the warm start file *before* calling the base class
            # presolve - the base class method ends up creating the command line,
            # and the warm start file-name is (obviously) needed there.
            if self._warm_start_file_name is None:
                assert not user_warmstart
                self._warm_start_file_name = pyutilib.services.TempfileManager.\
                                             create_tempfile(suffix = '.cplex.mst')

        # let the base class handle any remaining keywords/actions.
        ILMLicensedSystemCallSolver._presolve(self, *args, **kwds)

        # NB: we must let the base class presolve run first so that the
        # symbol_map is actually constructed!

        if (len(args) > 0) and (not isinstance(args[0], basestring)):

            if len(args) != 1:
                raise ValueError(
                    "CPLEX _presolve method can only handle a "
                    "single problem instance - %s were supplied"
                    % (len(args),))

            # write the warm-start file - currently only supports MIPs.
            # we only know how to deal with a single problem instance.
            if self._warm_start_solve and (not user_warmstart):

                start_time = time.time()
                self._warm_start(args[0])
                end_time = time.time()
                if self._report_timing:
                    print("Warm start write time= %.2f seconds"
                          % (end_time-start_time))

    def _default_executable(self):
        executable = pyutilib.services.registered_executable("cplex")
        if executable is None:
            logger.warning("Could not locate the 'cplex' executable"
                           ", which is required for solver %s"
                           % self.name)
            self.enable = False
            return None
        return executable.get_path()

    def _get_version(self):
        """
        Returns a tuple describing the solver executable version.
        """
        solver_exec = self.executable()
        if solver_exec is None:
            return _extract_version('')
        results = pyutilib.subprocess.run( [solver_exec,'-c','quit'], timelimit=1 )
        return _extract_version(results[1])

    def create_command_line(self, executable, problem_files):

        #
        # Define log file
        # The log file in CPLEX contains the solution trace, but the solver status can be found in the solution file.
        #
        if self._log_file is None:
            self._log_file = pyutilib.services.TempfileManager.\
                            create_tempfile(suffix = '.cplex.log')

        #
        # Define solution file
        # As indicated above, contains (in XML) both the solution and solver status.
        #
        if self._soln_file is None:
            self._soln_file = pyutilib.services.TempfileManager.\
                              create_tempfile(suffix = '.cplex.sol')

        #
        # Write the CPLEX execution script
        #
        script = "set logfile %s\n" % (self._log_file,)
        if self._timelimit is not None and self._timelimit > 0.0:
            script += "set timelimit %s\n" % ( self._timelimit, )
        if (self.options.mipgap is not None) and \
           (self.options.mipgap > 0.0):
            script += ("set mip tolerances mipgap %s\n"
                       % (self.options.mipgap,))
        for key in self.options:
            if key == 'relax_integrality' or key == 'mipgap':
                continue
            elif isinstance(self.options[key], basestring) and \
                 (' ' in self.options[key]):
                opt = " ".join(key.split('_'))+" "+str(self.options[key])
            else:
                opt = " ".join(key.split('_'))+" "+str(self.options[key])
            script += "set %s\n" % ( opt, )
        script += "read %s\n" % ( problem_files[0], )

        # if we're dealing with an LP, the MST file will be empty.
        if self._warm_start_solve and \
           (self._warm_start_file_name is not None):
            script += "read %s\n" % (self._warm_start_file_name,)

        if 'relax_integrality' in self.options:
            script += "change problem lp\n"

        script += "display problem stats\n"
        script += "optimize\n"
        script += "write %s\n" % (self._soln_file,)
        script += "quit\n"

        # dump the script and warm-start file names for the
        # user if we're keeping files around.
        if self._keepfiles:
            script_fname = pyutilib.services.TempfileManager.\
                           create_tempfile(suffix = '.cplex.script')
            tmp = open(script_fname,'w')
            tmp.write(script)
            tmp.close()

            print("Solver script file=" + script_fname)
            if self._warm_start_solve and \
               (self._warm_start_file_name is not None):
                print("Solver warm-start file="
                      +self._warm_start_file_name)

        #
        # Define command line
        #
        cmd = [executable]
        if self._timer:
            cmd.insert(0, self._timer)
        return pyutilib.misc.Bunch(cmd=cmd, script=script,
                                   log_file=self._log_file, env=None)

    def process_logfile(self):
        """
        Process logfile
        """
        results = SolverResults()
        results.problem.number_of_variables = None
        results.problem.number_of_nonzeros = None
        #
        # Process logfile
        #
        OUTPUT = open(self._log_file)
        output = "".join(OUTPUT.readlines())
        OUTPUT.close()
        #
        # It is generally useful to know the CPLEX version number for logfile parsing.
        #
        cplex_version = None

        #
        # Parse logfile lines
        #

        # caching for subsequent use - we need to known the problem sense before using this information.
        # adding to plugin to cache across invocation of process_logfile and process_soln_file.
        self._best_bound = None
        self._gap = None

        for line in output.split("\n"):
            tokens = re.split('[ \t]+',line.strip())
            if len(tokens) > 3 and tokens[0] == "CPLEX" and tokens[1] == "Error":
            # IMPT: See below - cplex can generate an error line and then terminate fine, e.g., in CPLEX 12.1.
            #       To handle these cases, we should be specifying some kind of termination criterion always
            #       in the course of parsing a log file (we aren't doing so currently - just in some conditions).
                results.solver.status=SolverStatus.error
                results.solver.error = " ".join(tokens)
            elif len(tokens) >= 3 and tokens[0] == "ILOG" and tokens[1] == "CPLEX":
                cplex_version = tokens[2].rstrip(',')
            elif len(tokens) >= 3 and tokens[0] == "Variables":
                if results.problem.number_of_variables is None: # CPLEX 11.2 and subsequent versions have two Variables sections in the log file output.
                    results.problem.number_of_variables = int(tokens[2])
            # In CPLEX 11 (and presumably before), there was only a single line output to
            # indicate the constriant count, e.g., "Linear constraints : 16 [Less: 7, Greater: 6, Equal: 3]".
            # In CPLEX 11.2 (or somewhere in between 11 and 11.2 - I haven't bothered to track it down
            # in that detail), there is another instance of this line prefix in the min/max problem statistics
            # block - which we don't care about. In this case, the line looks like: "Linear constraints :" and
            # that's all.
            elif len(tokens) >= 4 and tokens[0] == "Linear" and tokens[1] == "constraints":
                results.problem.number_of_constraints = int(tokens[3])
            elif len(tokens) >= 3 and tokens[0] == "Nonzeros":
                if results.problem.number_of_nonzeros is None: # CPLEX 11.2 and subsequent has two Nonzeros sections.
                    results.problem.number_of_nonzeros = int(tokens[2])
            elif len(tokens) >= 5 and tokens[4] == "MINIMIZE":
                results.problem.sense = ProblemSense.minimize
            elif len(tokens) >= 5 and tokens[4] == "MAXIMIZE":
                results.problem.sense = ProblemSense.maximize
            elif len(tokens) >= 4 and tokens[0] == "Solution" and tokens[1] == "time" and tokens[2] == "=":
                # technically, I'm not sure if this is CPLEX user time or user+system - CPLEX doesn't appear
                # to differentiate, and I'm not sure we can always provide a break-down.
                results.solver.user_time = float(tokens[3])
            elif len(tokens) >= 4 and tokens[0] == "Primal" and tokens[1] == "simplex" and tokens[3] == "Optimal:":
                results.solver.termination_condition = TerminationCondition.optimal
                results.solver.termination_message = ' '.join(tokens)
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
                # if CPLEX has previously printed an error message, reduce it to a warning -
                # there is a strong indication it recovered, but we can't be sure.
                if results.solver.status == SolverStatus.error:
                    results.solver.status = SolverStatus.warning
                else:
                    results.solver.status = SolverStatus.ok
                results.solver.termination_condition = TerminationCondition.infeasible
                results.solver.termination_message = ' '.join(tokens)
            elif len(tokens) >= 10 and tokens[0] == "MIP" and tokens[2] == "Time" and tokens[3] == "limit" and tokens[6] == "feasible:":
                # handle processing when the time limit has been exceeded, and we have a feasible solution.
                results.solver.status = SolverStatus.ok
                results.solver.termination_condition = TerminationCondition.maxTimeLimit
                results.solver.termination_message = ' '.join(tokens)
            elif len(tokens) >= 10 and tokens[0] == "Current" and tokens[1] == "MIP" and tokens[2] == "best" and tokens[3] == "bound":
                self._best_bound = float(tokens[5])
                self._gap = float(tokens[8].rstrip(','))
            # for the case below, CPLEX sometimes reports "true" optimal (the first case)
            # and other times within-tolerance optimal (the second case).
            elif (len(tokens) >= 4 and tokens[0] == "MIP" and tokens[2] == "Integer" and tokens[3] == "optimal") or \
                 (len(tokens) >= 4 and tokens[0] == "MIP" and tokens[2] == "Integer" and tokens[3] == "optimal,"):
                # if CPLEX has previously printed an error message, reduce it to a warning -
                # there is a strong indication it recovered, but we can't be sure.
                if results.solver.status == SolverStatus.error:
                    results.solver.status = SolverStatus.warning
                else:
                    results.solver.status = SolverStatus.ok
                results.solver.termination_condition = TerminationCondition.optimal
                results.solver.termination_message = ' '.join(tokens)
            elif len(tokens) >= 3 and tokens[0] == "Presolve" and tokens[2] == "Infeasible.":
                # if CPLEX has previously printed an error message, reduce it to a warning -
                # there is a strong indication it recovered, but we can't be sure.
                if results.solver.status == SolverStatus.error:
                    results.solver.status = SolverStatus.warning
                else:
                    results.solver.status = SolverStatus.ok
                results.solver.termination_condition = TerminationCondition.infeasible
                results.solver.termination_message = ' '.join(tokens)
            elif (len(tokens) == 6 and tokens[2] == "Integer" and tokens[3] == "infeasible" and tokens[5] == "unbounded.") or (len(tokens) >= 5 and tokens[0] == "Presolve" and tokens[2] == "Unbounded" and tokens[4] == "infeasible."):
                # if CPLEX has previously printed an error message, reduce it to a warning -
                # there is a strong indication it recovered, but we can't be sure.
                if results.solver.status == SolverStatus.error:
                    results.solver.status = SolverStatus.warning
                else:
                    results.solver.status = SolverStatus.ok
                # It isn't clear whether we can determine if the problem is unbounded from
                # CPLEX's output.
                results.solver.termination_condition = TerminationCondition.unbounded
                results.solver.termination_message = ' '.join(tokens)

        try:
            results.solver.termination_message = pyutilib.misc.yaml_fix(results.solver.termination_message)
        except:
            pass
        return results

    def process_soln_file(self,results):

        # the only suffixes that we extract from CPLEX are
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
                raise RuntimeError("***The CPLEX solver plugin cannot extract solution suffix="+suffix)

        # check for existence of the solution file
        # not sure why we just return - would think that we
        # would want to indicate some sort of error
        if not os.path.exists(self._soln_file):
            return

        range_duals = {}
        range_slacks = {}
        soln = Solution()
        soln.objective['__default_objective__'] = {'Value':None}

        # caching for efficiency
        soln_variables = soln.variable
        soln_constraints = soln.constraint

        INPUT = open(self._soln_file, "r")
        results.problem.number_of_objectives=1
        time_limit_exceeded = False
        mip_problem=False
        for line in INPUT:
            line = line.strip()
            line = line.lstrip('<?/')
            line = line.rstrip('/>?')
            tokens=line.split(' ')

            if tokens[0] == "variable":
                variable_name = None
                variable_value = None
                variable_reduced_cost = None
                variable_status = None
                for i in xrange(1,len(tokens)):
                    field_name =  tokens[i].split('=')[0]
                    field_value = tokens[i].split('=')[1].lstrip("\"").rstrip("\"")
                    if field_name == "name":
                        variable_name = field_value
                    elif field_name == "value":
                        variable_value = field_value
                    elif (extract_reduced_costs is True) and (field_name == "reducedCost"):
                        variable_reduced_cost = field_value
                    elif (extract_reduced_costs is True) and (field_name == "status"):
                        variable_status = field_value

                # skip the "constant-one" variable, used to capture/retain objective offsets in the CPLEX LP format.
                if variable_name != "ONE_VAR_CONSTANT":
                    variable = soln_variables[variable_name] = {"Value" : float(variable_value)}
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
                            raise ValueError("Unexpected reduced-cost value="+str(variable_reduced_cost)+" encountered for variable="+variable_name)
            elif (tokens[0] == "constraint") and ((extract_duals is True) or (extract_slacks is True)):
                is_range = False
                rlabel = None
                rkey = None
                for i in xrange(1,len(tokens)):
                    field_name =  tokens[i].split('=')[0]
                    field_value = tokens[i].split('=')[1].lstrip("\"").rstrip("\"")
                    if field_name == "name":
                        if field_value.startswith('c_'):
                            constraint = soln_constraints[field_value] = {}
                        elif field_value.startswith('r_l_'):
                            is_range = True
                            rlabel = field_value[4:]
                            rkey = 0
                        elif field_value.startswith('r_u_'):
                            is_range = True
                            rlabel = field_value[4:]
                            rkey = 1
                    elif (extract_duals is True) and (field_name == "dual"): # for LPs
                        if is_range is False:
                            constraint["Dual"] = float(field_value)
                        else:
                            range_duals.setdefault(rlabel,[0,0])[rkey] = float(field_value)
                    elif (extract_slacks is True) and (field_name == "slack"): # for MIPs
                        if is_range is False:
                            constraint["Slack"] = float(field_value)
                        else:
                            range_slacks.setdefault(rlabel,[0,0])[rkey] = float(field_value)
            elif tokens[0].startswith("problemName"):
                filename = (tokens[0].split('=')[1].strip()).lstrip("\"").rstrip("\"")
                results.problem.name = os.path.basename(filename)
                if '.' in results.problem.name:
                    results.problem.name = results.problem.name.split('.')[0]
                tINPUT=open(filename,"r")
                for tline in tINPUT:
                    tline = tline.strip()
                    if tline == "":
                        continue
                    tokens = re.split('[\t ]+',tline)
                    if tokens[0][0] in ['\\', '*']:
                        continue
                    elif tokens[0] == "NAME":
                        results.problem.name = tokens[1]
                    else:
                        sense = tokens[0].lower()
                        if sense in ['max','maximize']:
                            results.problem.sense = ProblemSense.maximize
                        if sense in ['min','minimize']:
                            results.problem.sense = ProblemSense.minimize
                    break
                tINPUT.close()

            elif tokens[0].startswith("objectiveValue"):
                objective_value = (tokens[0].split('=')[1].strip()).lstrip("\"").rstrip("\"")
                soln.objective['__default_objective__']['Value'] = float(objective_value)
            elif tokens[0].startswith("solutionStatusValue"):
               pieces = tokens[0].split("=")
               solution_status = eval(pieces[1])
               # solution status = 1 => optimal
               # solution status = 3 => infeasible
               if soln.status == SolutionStatus.unknown:
                  if solution_status == 1:
                    soln.status = SolutionStatus.optimal
                  elif solution_status == 3:
                    soln.status = SolutionStatus.infeasible
                    soln.gap = None
                  else:
                      # we are flagging anything with a solution status >= 4 as an error, to possibly
                      # be over-ridden as we learn more about the status (e.g., due to time limit exceeded).
                      soln.status = SolutionStatus.error
                      soln.gap = None
            elif tokens[0].startswith("solutionStatusString"):
                solution_status = ((" ".join(tokens).split('=')[1]).strip()).lstrip("\"").rstrip("\"")
                if solution_status in ["optimal", "integer optimal solution", "integer optimal, tolerance"]:
                    soln.status = SolutionStatus.optimal
                    soln.gap = 0.0
                    results.problem.lower_bound = soln.objective['__default_objective__']['Value']
                    results.problem.upper_bound = soln.objective['__default_objective__']['Value']
                    if "integer" in solution_status:
                        mip_problem=True
                elif solution_status in ["infeasible"]:
                    soln.status = SolutionStatus.infeasible
                    soln.gap = None
                elif solution_status in ["time limit exceeded"]:
                    # we need to know if the solution is primal feasible, and if it is, set the solution status accordingly.
                    # for now, just set the flag so we can trigger the logic when we see the primalFeasible keyword.
                    time_limit_exceeded = True
            elif tokens[0].startswith("MIPNodes"):
                if mip_problem:
                    n = eval(eval((" ".join(tokens).split('=')[1]).strip()).lstrip("\"").rstrip("\""))
                    results.solver.statistics.branch_and_bound.number_of_created_subproblems=n
                    results.solver.statistics.branch_and_bound.number_of_bounded_subproblems=n
            elif tokens[0].startswith("primalFeasible") and (time_limit_exceeded is True):
                primal_feasible = int(((" ".join(tokens).split('=')[1]).strip()).lstrip("\"").rstrip("\""))
                if primal_feasible == 1:
                    soln.status = SolutionStatus.feasible
                    if (results.problem.sense == ProblemSense.minimize):
                        results.problem.upper_bound = soln.objective['__default_objective__']['Value']
                    else:
                        results.problem.lower_bound = soln.objective['__default_objective__']['Value']
                else:
                    soln.status = SolutionStatus.infeasible


        if self._best_bound is not None:
            if results.problem.sense == ProblemSense.minimize:
                results.problem.lower_bound = self._best_bound
            else:
                results.problem.upper_bound = self._best_bound
        if self._gap is not None:
            soln.gap = self._gap

        # For the range constraints, supply only the dual with the largest
        # magnitude (at least one should always be numerically zero)
        for key,(ld,ud) in iteritems(range_duals):
            if abs(ld) > abs(ud):
                soln_constraints['r_l_'+key] = {"Dual" : ld}
            else:
                soln_constraints['r_l_'+key] = {"Dual" : ud}                # Use the same key
        # slacks
        for key,(ls,us) in iteritems(range_slacks):
            if abs(ls) > abs(us):
                soln_constraints.setdefault('r_l_'+key,{})["Slack"] = ls
            else:
                soln_constraints.setdefault('r_l_'+key,{})["Slack"] = us    # Use the same key

        if not results.solver.status is SolverStatus.error:
            if results.solver.termination_condition in [TerminationCondition.unknown,
                                                        #TerminationCondition.maxIterations,
                                                        #TerminationCondition.minFunctionValue,
                                                        #TerminationCondition.minStepLength,
                                                        TerminationCondition.globallyOptimal,
                                                        TerminationCondition.locallyOptimal,
                                                        TerminationCondition.optimal,
                                                        #TerminationCondition.maxEvaluations,
                                                        TerminationCondition.other]:
                results.solution.insert(soln)
            elif (results.solver.termination_condition is \
                  TerminationCondition.maxTimeLimit) and \
                  (soln.status is not SolutionStatus.infeasible):
                results.solution.insert(soln)

        INPUT.close()

    def _postsolve(self):

        # take care of the annoying (and empty) CPLEX temporary files in the current directory.
        # this approach doesn't seem overly efficient, but python os module functions don't
        # accept regular expression directly.
        filename_list = os.listdir(".")
        for filename in filename_list:
            # CPLEX temporary files come in two flavors - cplex.log and clone*.log.
            # the latter is the case for multi-processor environments.
            # IMPT: trap the possible exception raised by the file not existing.
            #       this can occur in pyro environments where > 1 workers are
            #       running CPLEX, and were started from the same directory.
            #       these logs don't matter anyway (we redirect everything),
            #       and are largely an annoyance.
            try:
                if  re.match('cplex\.log', filename) != None:
                    os.remove(filename)
                elif re.match('clone\d+\.log', filename) != None:
                    os.remove(filename)
            except OSError:
                pass

        # let the base class deal with returning results.
        results = ILMLicensedSystemCallSolver._postsolve(self)

        # finally, clean any temporary files registered with the temp file
        # manager, created populated *directly* by this plugin. does not
        # include, for example, the execution script. but does include
        # the warm-start file.
        pyutilib.services.TempfileManager.pop(remove=not self._keepfiles)

        return results

class MockCPLEX(CPLEXSHELL,MockMIP):
    """A Mock CPLEX solver used for testing
    """

    pyomo.util.plugin.alias('_mock_cplex')

    def __init__(self, **kwds):
        try:
            CPLEXSHELL.__init__(self, **kwds)
        except pyutilib.common.ApplicationError: #pragma:nocover
            pass                        #pragma:nocover
        MockMIP.__init__(self,"cplex")

    def available(self, exception_flag=True):
        return CPLEXSHELL.available(self,exception_flag)

    def create_command_line(self, executable, problem_files):
        command = CPLEXSHELL.create_command_line(self,
                                                 executable,
                                                 problem_files)
        MockMIP.create_command_line(self,
                                    executable,
                                    problem_files)
        return command

    def _default_executable(self):
        return MockMIP.executable(self)

    def _execute_command(self, cmd):
        return MockMIP._execute_command(self, cmd)


pyutilib.services.register_executable(name="cplex")
pyutilib.services.register_executable(name="cplexamp")

