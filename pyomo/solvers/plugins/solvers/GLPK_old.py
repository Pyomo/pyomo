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
import sys

from pyutilib.common import ApplicationError
from pyomo.util.plugin import alias
from pyutilib.misc import Bunch, Options
from pyutilib.services import register_executable, registered_executable
from pyutilib.services import TempfileManager
import pyutilib.subprocess

from pyomo.opt.base import *
from pyomo.opt.base.solvers import _extract_version
from pyomo.opt.results import *
from pyomo.opt.solver import *
from pyomo.solvers.mockmip import MockMIP
from pyomo.solvers.plugins.solvers.GLPK import _glpk_version

from six import iteritems, string_types
import logging
logger = logging.getLogger('pyomo.solvers')


   # status of auxiliary / structural variables
GLP_BS = 1   # inactive constraint / basic variable
GLP_NL = 2   # active constraint or non-basic variable on lower bound
GLP_NU = 3   # active constraint or non-basic variable on upper bound
GLP_NF = 4   # active free row or non-basic free variable
GLP_NS = 5   # active equality constraint or non-basic fixed variable

   # solution status
GLP_UNDEF  = 1  # solution is undefined
GLP_FEAS   = 2  # solution is feasible
GLP_INFEAS = 3  # solution is infeasible
GLP_NOFEAS = 4  # no feasible solution exists
GLP_OPT    = 5  # solution is optimal
GLP_UNBND  = 6  # solution is unbounded


class GLPKSHELL_4_42(SystemCallSolver):
    """Shell interface to the GLPK LP/MIP solver"""

    pyomo.util.plugin.alias(
        '_glpk_shell_4_42',
        doc='Shell interface to the GNU Linear Programming Kit (4.42-4.59)')

    def __init__ (self, **kwargs):
        #
        # Call base constructor
        #
        kwargs['type'] = 'glpk'
        SystemCallSolver.__init__(self, **kwargs)

        self._rawfile = None

        #
        # Valid problem formats, and valid results for each format
        #
        self._valid_problem_formats = [ProblemFormat.cpxlp,
                                       ProblemFormat.mps,
                                       ProblemFormat.mod]
        self._valid_result_formats = {
          ProblemFormat.mod:   ResultsFormat.soln,
          ProblemFormat.cpxlp: ResultsFormat.soln,
          ProblemFormat.mps:   ResultsFormat.soln,
        }
        self.set_problem_format(ProblemFormat.cpxlp)

        # Note: Undefined capabilities default to 'None'
        self._capabilities = Options()
        self._capabilities.linear = True
        self._capabilities.integer = True

    def _default_results_format(self, prob_format):
        return ResultsFormat.soln

    def _default_executable(self):
        executable = registered_executable('glpsol')
        if executable is None:
            msg = ("Could not locate the 'glpsol' executable, which is "
                   "required for solver '%s'")
            logger.warning(msg % self.name)
            self.enable = False
            return None
        return executable.get_path()

    def _get_version(self):
        """
        Returns a tuple describing the solver executable version.
        """
        if _glpk_version is None:
            return _extract_version('')
        return _glpk_version

    def create_command_line(self, executable, problem_files):
        #
        # Define log file
        #
        if self._log_file is None:
            self._log_file = TempfileManager.create_tempfile(suffix='.glpk.log')

        #
        # Define solution file
        #
        self._glpfile = TempfileManager.create_tempfile(suffix='.glpk.glp')
        self._rawfile = TempfileManager.create_tempfile(suffix='.glpk.raw')
        self._soln_file = self._rawfile

        #
        # Define command line
        #
        cmd = [executable]
        if self._timer:
            cmd.insert(0, self._timer)
        for key in self.options:
            opt = self.options[key]
            if opt is None or (isinstance(opt, string_types) and opt.strip() == ''):
                # Handle the case for options that must be
                # specified without a value
                cmd.append("--%s" % key)
            else:
                cmd.extend(["--%s" % key, str(opt)])
            #if isinstance(opt, basestring) and ' ' in opt:
            #    cmd.append('--%s "%s"' % (key, str(opt)))
            #else:
            #    cmd.append('--%s %s' % (key, str(opt)))

        if self._timelimit is not None and self._timelimit > 0.0:
            cmd.extend(['--tmlim', str(self._timelimit)])

        cmd.extend(['--write', self._rawfile])
        cmd.extend(['--wglp', self._glpfile])

        if self._problem_format == ProblemFormat.cpxlp:
            cmd.extend(['--cpxlp', problem_files[0]])
        elif self._problem_format == ProblemFormat.mps:
            cmd.extend(['--freemps', problem_files[0]])
        elif self._problem_format == ProblemFormat.mod:
            cmd.extend(['--math', problem_files[0]])
            for fname in problem_files[1:]:
                cmd.extend(['--data', fname])

        return Bunch(cmd=cmd, log_file=self._log_file, env=None)

    def process_logfile(self):
        """
        Process logfile
        """
        results = SolverResults()

        # For the lazy programmer, handle long variable names
        prob   = results.problem
        solv   = results.solver
        solv.termination_condition = TerminationCondition.unknown
        stats  = results.solver.statistics
        bbound = stats.branch_and_bound

        prob.upper_bound = float('inf')
        prob.lower_bound = float('-inf')
        bbound.number_of_created_subproblems = 0
        bbound.number_of_bounded_subproblems = 0

        with open(self._log_file, 'r') as output:
            for line in output:
                toks = line.split()
                if 'tree is empty' in line:
                    bbound.number_of_created_subproblems = toks[-1][:-1]
                    bbound.number_of_bounded_subproblems = toks[-1][:-1]
                elif len(toks) == 2 and toks[0] == "sys":
                    solv.system_time = toks[1]
                elif len(toks) == 2 and toks[0] == "user":
                    solv.user_time = toks[1]
                elif len(toks) > 2 and (toks[0], toks[2]) == ("TIME", "EXCEEDED;"):
                    solv.termination_condition = TerminationCondition.maxTimeLimit
                elif len(toks) > 5 and (toks[:6] == ['PROBLEM', 'HAS', 'NO', 'DUAL', 'FEASIBLE', 'SOLUTION']):
                    solv.termination_condition = TerminationCondition.unbounded
                elif len(toks) > 5 and (toks[:6] == ['PROBLEM', 'HAS', 'NO', 'PRIMAL', 'FEASIBLE', 'SOLUTION']):
                    solv.termination_condition = TerminationCondition.infeasible
                elif len(toks) > 4 and (toks[:5] == ['PROBLEM', 'HAS', 'NO', 'FEASIBLE', 'SOLUTION']):
                    solv.termination_condition = TerminationCondition.infeasible
                elif len(toks) > 6 and (toks[:7] == ['LP', 'RELAXATION', 'HAS', 'NO', 'DUAL', 'FEASIBLE', 'SOLUTION']):
                    solv.termination_condition = TerminationCondition.unbounded

        return results

    def _glpk_get_solution_status(self, status):
        if   GLP_OPT    == status: return SolutionStatus.optimal
        elif GLP_FEAS   == status: return SolutionStatus.feasible
        elif GLP_INFEAS == status: return SolutionStatus.infeasible
        elif GLP_NOFEAS == status: return SolutionStatus.infeasible
        elif GLP_UNBND  == status: return SolutionStatus.unbounded
        elif GLP_UNDEF  == status: return SolutionStatus.other
        raise RuntimeError("Unknown solution status returned by GLPK solver")

    def process_soln_file(self, results):
        soln  = None
        pdata = self._glpfile
        psoln = self._rawfile

        prob = results.problem
        solv = results.solver

        prob.name = 'unknown'   # will ostensibly get updated

        # Step 1: Make use of the GLPK's machine parseable format (--wglp) to
        #    collect variable and constraint names.
        glp_line_count = ' -- File not yet opened'

        # The trick for getting the variable names correctly matched to their
        # values is the note that the --wglp option outputs them in the same
        # order as the --write output.
        # Note that documentation for these formats is available from the GLPK
        # documentation of 'glp_read_prob' and 'glp_write_sol'
        variable_names = dict()    # cols
        constraint_names = dict()  # rows
        obj_name = 'objective'

        try:
            f = open(pdata, 'r')

            glp_line_count = 1
            pprob, ptype, psense, prows, pcols, pnonz = f.readline().split()
            prows = int(prows)  # fails if not a number; intentional
            pcols = int(pcols)  # fails if not a number; intentional
            pnonz = int(pnonz)  # fails if not a number; intentional

            if pprob != 'p' or \
               ptype not in ('lp', 'mip') or \
               psense not in ('max', 'min') or \
               prows < 0 or pcols < 0 or pnonz < 0:
                raise ValueError

            self.is_integer = ('mip' == ptype and True or False)
            prob.sense = 'min' == psense and ProblemSense.minimize or ProblemSense.maximize
            prob.number_of_constraints = prows
            prob.number_of_nonzeros    = pnonz
            prob.number_of_variables   = pcols

            extract_duals = False
            extract_reduced_costs = False
            for suffix in self._suffixes:
                flag = False
                if re.match(suffix, "dual"):
                    if not self.is_integer:
                        flag = True
                        extract_duals = True
                if re.match(suffix, "rc"):
                    if not self.is_integer:
                        flag = True
                        extract_reduced_costs = True
                if not flag:
                    # TODO: log a warning
                    pass

            for line in f:
                glp_line_count += 1
                tokens = line.split()
                switch = tokens.pop(0)

                if switch in ('a', 'e', 'i', 'j'):
                    pass
                elif 'n' == switch:  # naming some attribute
                    ntype = tokens.pop(0)
                    name  = tokens.pop()
                    if 'i' == ntype:      # row
                        row = tokens.pop()
                        constraint_names[int(row)] = name
                        # --write order == --wglp order; store name w/ row no
                    elif 'j' == ntype:    # var
                        col = tokens.pop()
                        variable_names[int(col)] = name
                        # --write order == --wglp order; store name w/ col no
                    elif 'z' == ntype:    # objective
                        obj_name = name
                    elif 'p' == ntype:    # problem name
                        prob.name = name
                    else:                 # anything else is incorrect.
                        raise ValueError

                else:
                    raise ValueError
        except Exception:
            e = sys.exc_info()[1]
            msg = "Error parsing solution description file, line %s: %s"
            raise ValueError(msg % (glp_line_count, str(e)))
        finally:
            f.close()

        range_duals = {}
        # Step 2: Make use of the GLPK's machine parseable format (--write) to
        #    collect solution variable and constraint values.
        raw_line_count = ' -- File not yet opened'
        try:
            f = open(psoln, 'r')

            raw_line_count = 1
            prows, pcols = f.readline().split()
            prows = int(prows)  # fails if not a number; intentional
            pcols = int(pcols)  # fails if not a number; intentional

            raw_line_count = 2
            if self.is_integer:
                pstat, obj_val = f.readline().split()
            else:
                pstat, dstat, obj_val = f.readline().split()
                dstat = float(dstat) # dual status of basic solution.  Ignored.

            pstat = float(pstat)       # fails if not a number; intentional
            obj_val = float(obj_val)   # fails if not a number; intentional
            soln_status = self._glpk_get_solution_status(pstat)

            if soln_status is SolutionStatus.infeasible:
                solv.termination_condition = TerminationCondition.infeasible

            elif soln_status is SolutionStatus.unbounded:
                solv.termination_condition = TerminationCondition.unbounded

            elif soln_status is SolutionStatus.other:
                if solv.termination_condition == TerminationCondition.unknown:
                    solv.termination_condition = TerminationCondition.other

            elif soln_status in (SolutionStatus.optimal, SolutionStatus.feasible):
                soln   = results.solution.add()
                soln.status = soln_status

                prob.lower_bound = obj_val
                prob.upper_bound = obj_val

                # TODO: Does a 'feasible' status mean that we're optimal?
                soln.gap=0.0
                solv.termination_condition = TerminationCondition.optimal

                # I'd like to choose the correct answer rather than just doing
                # something like commenting the obj_name line.  The point is that
                # we ostensibly could or should make use of the user's choice in
                # objective name.  In that vein I'd like to set the objective value
                # to the objective name.  This would make parsing on the user end
                # less 'arbitrary', as in the yaml key 'f'.  Weird
                soln.objective[obj_name] = {'Value': obj_val}

                if (self.is_integer is True) or (extract_duals is False):
                    # we use nothing from this section so just read in the
                    # lines and throw them away
                    for mm in range(1, prows +1):
                        raw_line_count += 1
                        f.readline()
                else:
                    for mm in range(1, prows +1):
                        raw_line_count += 1

                        rstat, rprim, rdual = f.readline().split()
                        rstat = float(rstat)

                        cname = constraint_names[mm]
                        if 'ONE_VAR_CONSTANT' == cname[-16:]: continue

                        if cname.startswith('c_'):
                            soln.constraint[cname] = {"Dual":float(rdual)}
                        elif cname.startswith('r_l_'):
                            range_duals.setdefault(cname[4:],[0,0])[0] = float(rdual)
                        elif cname.startswith('r_u_'):
                            range_duals.setdefault(cname[4:],[0,0])[1] = float(rdual)

                for nn in range(1, pcols +1):
                    raw_line_count += 1
                    if self.is_integer:
                        cprim = f.readline()      # should be a single number
                    else:
                        cstat, cprim, cdual = f.readline().split()
                        cstat = float(cstat)  # fails if not a number; intentional

                    vname = variable_names[nn]
                    if 'ONE_VAR_CONSTANT' == vname: continue
                    cprim = float(cprim)
                    if extract_reduced_costs is False:
                        soln.variable[vname] = {"Value" : cprim}
                    else:
                        soln.variable[vname] = {"Value" : cprim,
                                                "Rc" : float(cdual)}

        except Exception:
            print(sys.exc_info()[1])
            msg = "Error parsing solution data file, line %d" % raw_line_count
            raise ValueError(msg)
        finally:
            f.close()

        if not soln is None:
            # For the range constraints, supply only the dual with the largest
            # magnitude (at least one should always be numerically zero)
            scon = soln.Constraint
            for key,(ld,ud) in iteritems(range_duals):
                if abs(ld) > abs(ud):
                    scon['r_l_'+key] = {"Dual":ld}
                else:
                    scon['r_l_'+key] = {"Dual":ud}      # Use the same key



class GLPKSHELL_old(SystemCallSolver):
    """Shell interface to the GLPK LP/MIP solver"""

    alias('_glpk_shell_old', doc='Shell interface to the GNU Linear Programming Kit (before 4.42)')

    def __init__(self, **kwds):
        #
        # Call base constructor
        #
        kwds['type'] = 'glpk'
        SystemCallSolver.__init__(self, **kwds)
        #
        # Valid problem formats, and valid results for each format
        #
        self._valid_problem_formats = [
               ProblemFormat.mod, ProblemFormat.cpxlp, ProblemFormat.mps]
        self._valid_result_formats = {
            ProblemFormat.mod: ResultsFormat.soln,
            ProblemFormat.cpxlp: ResultsFormat.soln,
            ProblemFormat.mps: ResultsFormat.soln,
        }
        self.set_problem_format(ProblemFormat.cpxlp)

        # Note: Undefined capabilities default to 'None'
        self._capabilities = Options()
        self._capabilities.linear = True
        self._capabilities.integer = True

    def _default_results_format(self, prob_format):
        return ResultsFormat.soln

    def _default_executable(self):
        executable = registered_executable('glpsol')
        if executable is None:
            msg = "Could not locate the 'glpsol' executable, which is " \
                  "required for solver '%s'"
            logger.warning(msg % self.name)
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
        errcode, results = pyutilib.subprocess.run(
        [solver_exec, "--version"], timelimit=1)
        if errcode == 0:
            return _extract_version(results)
        return _extract_version('')

    def create_command_line(self, executable, problem_files):
        #
        # Define log file
        #
        if self._log_file is None:
            self._log_file = TempfileManager.create_tempfile(suffix='.glpk.log')

        #
        # Define solution file
        #
        self._soln_file = TempfileManager.create_tempfile(suffix='.glpk.soln')

        #
        # Define command line
        #
        cmd = [executable]
        if self._timer:
            cmd.insert(0, self._timer)
        for key in self.options:
            opt = self.options[ key ]
            if (opt.__class__ is str) and (opt.strip() == ''):
                # Handle the case for options that must be
                # specified without a value
                cmd.append("--%s" % key)
            else:
                cmd.extend(["--%s" % key, str(opt)])
            #if isinstance(self.options[key], basestring) \
            #       and ' ' in self.options[key]:
            #    cmd.append('--%s "%s"' % (key, str(self.options[key])))
            #else:
            #    cmd.append('--%s %s' % (key, str(self.options[key])))

        if self._timelimit is not None and self._timelimit > 0.0:
            cmd.extend(['--tmlim', str(self._timelimit)])

        cmd.extend(['--output', self._soln_file])

        if self._problem_format == ProblemFormat.cpxlp:
            cmd.extend(['--cpxlp', problem_files[0]])
        elif self._problem_format == ProblemFormat.mps:
            cmd.extend(['--freemps', problem_files[0]])
        elif self._problem_format == ProblemFormat.mod:
            cmd.extend(['--math', problem_files[0]])
            for fname in problem_files[1:]:
                cmd.extend(['--data', fname])

        return Bunch(cmd=cmd, log_file=self._log_file, env=None)

    def process_logfile(self):
        """
        Process logfile
        """
        results = SolverResults()
        #
        # Initial values
        #
        # results.solver.statistics.branch_and_bound.
        #     number_of_created_subproblems=0
        # results.solver.statistics.branch_and_bound.
        #     number_of_bounded_subproblems=0
        soln = results.solution.add()
        #
        # Process logfile
        #
        OUTPUT = open(self._log_file)
        output = "".join(OUTPUT.readlines())
        OUTPUT.close()
        #
        # Parse logfile lines
        #

        # Handle long variable names
        stats = results.solver.statistics

        for line in output.split("\n"):
            tokens = re.split('[ \t]+', line.strip())
            if len(tokens) > 4 and tokens[0] == "+" and \
                     tokens[2] == "mip" and tokens[4] == "not":
                results.problem.lower_bound = tokens[8]
            elif len(tokens) > 4 and tokens[0] == "+" and \
                     tokens[1] == "mip" and tokens[4] == "not":
                results.problem.lower_bound = tokens[7]
            elif len(tokens) > 4 and tokens[0] == "+" and \
                     tokens[2] == "mip" and tokens[4] != "not":
                if tokens[6] != "tree":
                    results.problem.lower_bound = tokens[6]
            elif len(tokens) > 4 and tokens[0] == "+" and \
                     tokens[1] == "mip" and tokens[4] != "not":
                results.problem.lower_bound = tokens[5]
            elif len(tokens) == 6 and tokens[0] == "OPTIMAL" and \
                     tokens[1] == "SOLUTION" and tokens[5] == "PRESOLVER":
                stats.branch_and_bound.number_of_created_subproblems = 0
                stats.branch_and_bound.number_of_bounded_subproblems = 0
                soln.status = SolutionStatus.optimal
            elif len(tokens) == 7 and tokens[1] == "OPTIMAL" and \
                     tokens[2] == "SOLUTION" and tokens[6] == "PRESOLVER":
                stats.branch_and_bound.number_of_created_subproblems = 0
                stats.branch_and_bound.number_of_bounded_subproblems = 0
                soln.status = SolutionStatus.optimal
            elif len(tokens) > 10 and tokens[0] == "+" and \
                     tokens[8] == "empty":
                stats.branch_and_bound.number_of_created_subproblems = tokens[11][:-1]
                stats.branch_and_bound.number_of_bounded_subproblems = tokens[11][:-1]
            elif len(tokens) > 9 and tokens[0] == "+" and \
                     tokens[7] == "empty":
                stats.branch_and_bound.number_of_created_subproblems = tokens[10][:-1]
                stats.branch_and_bound.number_of_bounded_subproblems = tokens[10][:-1]
            elif len(tokens) == 2 and tokens[0] == "sys":
                results.solver.system_time = tokens[1]
            elif len(tokens) == 2 and tokens[0] == "user":
                results.solver.user_time = tokens[1]
            elif len(tokens) > 2 and tokens[0] == "OPTIMAL" and \
                     tokens[1] == "SOLUTION":
                soln.status = SolutionStatus.optimal
            elif len(tokens) > 2 and tokens[0] == "INTEGER" and \
                     tokens[1] == "OPTIMAL":
                soln.status = SolutionStatus.optimal
                stats.branch_and_bound.number_of_created_subproblems = 0
                stats.branch_and_bound.number_of_bounded_subproblems = 0
            elif len(tokens) > 2 and tokens[0] == "TIME" and \
                     tokens[2] == "EXCEEDED;":
                soln.status = SolutionStatus.stoppedByLimit
            if soln.status == SolutionStatus.optimal:
                results.solver.termination_condition = TerminationCondition.optimal
            elif soln.status == SolutionStatus.infeasible:
                results.solver.termination_condition = TerminationCondition.infeasible
        if results.problem.upper_bound == "inf":
            results.problem.upper_bound = 'Infinity'
        if results.problem.lower_bound == "-inf":
            results.problem.lower_bound = "-Infinity"
        try:
            val = results.problem.upper_bound
            tmp = eval(val.strip())
            results.problem.upper_bound = str(tmp)
        except:
            pass
        try:
            val = results.problem.lower_bound
            tmp = eval(val.strip())
            results.problem.lower_bound = str(tmp)
        except:
            pass

        if results.solver.status is SolverStatus.error:
            results.solution.delete(0)
        return results

    def process_soln_file(self, results):

        # the only suffixes that we extract from GLPK are
        # constraint duals. scan through the solver suffix
        # list and throw an exception if the user has
        # specified any others.
        extract_duals = False
        for suffix in self._suffixes:
            flag = False
            if re.match(suffix, "dual"):
                extract_duals = True
                flag = True
            if not flag:
                raise RuntimeError(\
                      "***GLPK solver plugin cannot extract solution " + \
                       "suffix='%s'" % (suffix))

        lp_solution = True  # if false, we're dealing with a MIP!
        if not os.path.exists(self._soln_file):
            return
        soln = results.solution(0)
        INPUT = open(self._soln_file, "r")

        range_duals = {}
        try:

            state = 0  # 0=initial header, 1=constraints, 2=variables, -1=done

            results.problem.number_of_objectives = 1

            # for validation of the total count read and the order
            number_of_constraints_read = 0
            number_of_variables_read = 0

            # constraint names and their value/bounds can be split
            # across multiple lines
            active_constraint_name = ""

            # variable names and their value/bounds can be split across
            # multiple lines
            active_variable_name = ""

            for line in INPUT:
                tokens = re.split('[ \t]+', line.strip())

                if (len(tokens) == 1) and (len(tokens[0]) == 0):
                    pass
                elif state == 0:
                    #
                    # Processing initial header
                    #
                    if len(tokens) == 2 and tokens[0] == "Problem:":
                        # the problem name may be absent, in which case
                        # the "Problem:" line will be skipped.
                        results.problem.name = tokens[1]
                    elif len(tokens) == 2 and tokens[0] == "Rows:":
                        results.problem.number_of_constraints = eval(tokens[1])
                    elif len(tokens) == 2 and tokens[0] == "Columns:":
                        lp_solution = True
                        results.problem.number_of_variables = eval(tokens[1])
                    elif len(tokens) > 2 and tokens[0] == "Columns:":
                        lp_solution = False
                        results.problem.number_of_variables = eval(tokens[1])
                    elif len(tokens) == 2 and tokens[0] == "Non-zeros:":
                        results.problem.number_of_nonzeros = eval(tokens[1])
                    elif len(tokens) >= 2 and tokens[0] == "Status:":
                        if tokens[1] == "OPTIMAL":
                            soln.status = SolutionStatus.optimal
                        elif len(tokens) == 3 and tokens[1] == "INTEGER" and \
                                 tokens[2] == "NON-OPTIMAL":
                            soln.status = SolutionStatus.bestSoFar
                        elif len(tokens) == 3 and tokens[1] == "INTEGER" and \
                                 tokens[2] == "OPTIMAL":
                            soln.status = SolutionStatus.optimal
                        elif len(tokens) == 3 and tokens[1] == "INTEGER" and \
                                 tokens[2] == "UNDEFINED":
                            soln.status = SolutionStatus.stoppedByLimit
                        elif len(tokens) == 3 and tokens[1] == "INTEGER" and \
                                tokens[2] == "EMPTY":
                            soln.status = SolutionStatus.infeasible
                        elif (len(tokens) == 2) and (tokens[1] == "UNDEFINED"):
                            soln.status = SolutionStatus.infeasible
                        else:
                            print("WARNING: Read unknown status while " + \
                                   "parsing GLPK solution file - " + \
                                   "status='%s'") % (" ".join(tokens[1:]))
                    elif len(tokens) >= 2 and tokens[0] == "Objective:":
                        if tokens[4] == "(MINimum)":
                            results.problem.sense = ProblemSense.minimize
                        else:
                            results.problem.sense = ProblemSense.maximize
                        soln.objective[tokens[1]] = {'Value': float(tokens[3])}
                        if soln.status is SolutionStatus.optimal:
                            results.problem.lower_bound = soln.objective[tokens[1]]['Value']
                            results.problem.upper_bound = soln.objective[tokens[1]]['Value']
                        # the objective is the last entry in the problem section - move on to constraints.
                        state = 1

                elif state == 1:
                    #
                    # Process Constraint Info
                    #

                    if (len(tokens) == 2) and (len(active_constraint_name) == 0):
                        number_of_constraints_read = number_of_constraints_read + 1
                        active_constraint_name = tokens[1].strip()
                        index = eval(tokens[0].strip())

                        # sanity check - the indices should be in sequence.
                        if index != number_of_constraints_read:
                            raise ValueError(\
                                  ("***ERROR: Unexpected constraint index " + \
                                   "encountered on line=%s; expected " + \
                                   "value=%s; actual value=%s") % \
                                   (line, str(number_of_consrtaints_read),
                                    str(index)))
                    else:
                        index = None
                        activity = None
                        lower_bound = None
                        upper_bound = None
                        marginal = None

                        # extract the field names and process accordingly. there
                        # is some wasted processing w.r.t. single versus double-line
                        # entries, but it's not significant enough to worry about.

                        index_string = line[0:6].strip()
                        name_string = line[7:19].strip()
                        activity_string = line[23:36].strip()
                        lower_bound_string = line[37:50].strip()
                        upper_bound_string = line[51:64].strip()

                        state_string = None
                        marginal_string = None

                        # skip any headers
                        if (index_string == "------") or (index_string == "No."):
                            continue

                        if len(index_string) > 0:
                            index = eval(index_string)

                        if lp_solution is True:
                            state_string = line[20:22].strip()
                            marginal_string = line[65:78].strip()
                            if (activity_string != "< eps") and (len(activity_string) > 0):
                                activity = eval(activity_string)
                            else:
                                activity = 0.0
                            if (lower_bound_string != "< eps") and (len(lower_bound_string) > 0):
                                lower_bound = eval(lower_bound_string)
                            else:
                                lower_bound = 0.0
                            if state_string != "NS" and upper_bound_string != '=':
                                if (upper_bound_string != "< eps") and (len(upper_bound_string) > 0):
                                    upper_bound = eval(upper_bound_string)
                                else:
                                    upper_bound = 0.0
                            if (marginal_string != "< eps") and (len(marginal_string) > 0):
                                marginal = eval(marginal_string)
                            else:
                                marginal = 0.0

                        else:
                            # no constraint-related attributes/values are extracted currently for MIPs.
                            pass

                        constraint_name = None
                        if len(active_constraint_name) > 0:
                            # if there is an active constraint name, the identifier was
                            # too long for everything to be on a single line; the second
                            # line contains all of the value information.
                            constraint_name = active_constraint_name
                            active_constraint_name = ""
                        else:
                            # everything is on a single line.
                            constraint_name = name_string
                            number_of_constraints_read = number_of_constraints_read + 1
                            # sanity check - the indices should be in sequence.
                            if index != number_of_constraints_read:
                                raise ValueError("***ERROR: Unexpected constraint index encountered on line="+line+"; expected value="+str(number_of_constraints_read)+"; actual value="+str(index))

                        if (lp_solution is True) and (extract_duals is True):
                            # GLPK doesn't report slacks directly.
                            constraint_dual = activity
                            if state_string == "B":
                                constraint_dual = 0.0
                            elif (state_string == "NS") or (state_string == "NL") or (state_string == "NU"):
                                constraint_dual = marginal
                            else:
                                raise ValueError("Unknown status="+tokens[0]+" encountered "
                                                 "for constraint="+active_constraint_name+" "
                                                 "in line="+line+" of solution file="+self._soln_file)

                            if constraint_name.startswith('c_'):
                                soln.constraint[constraint_name] = {"Dual" : float(constraint_dual)}
                            elif constraint_name.startswith('r_l_'):
                                range_duals.setdefault(constraint_name[4:],[0,0])[0] = float(constraint_dual)
                            elif constraint_name.startswith('r_u_'):
                                range_duals.setdefault(constraint_name[4:],[0,0])[1] = float(constraint_dual)

                        else:
                            # there isn't anything interesting to do with constraints in the MIP case.
                            pass

                        # if all of the constraints have been read, exit.
                        if number_of_constraints_read == results.problem.number_of_constraints:
                            state = 2

                elif state == 2:
                    #
                    # Process Variable Info
                    #

                    if (len(tokens) == 2) and (len(active_variable_name) == 0):

                        # in the case of name over-flow, there are only two tokens
                        # on the first of two lines for the variable entry.
                        number_of_variables_read = number_of_variables_read + 1
                        active_variable_name = tokens[1].strip()
                        index = eval(tokens[0].strip())

                        # sanity check - the indices should be in sequence.
                        if index != number_of_variables_read:
                            raise ValueError("***ERROR: Unexpected variable index encountered on line="+line+"; expected value="+str(number_of_variables_read)+"; actual value="+str(index))

                    else:

                        index = None
                        activity = None
                        lower_bound = None
                        upper_bound = None
                        marginal = None

                        # extract the field names and process accordingly. there
                        # is some wasted processing w.r.t. single versus double-line
                        # entries, but it's not significant enough to worry about.

                        index_string = line[0:6].strip()
                        name_string = line[7:19].strip()
                        activity_string = line[23:36].strip()
                        lower_bound_string = line[37:50].strip()
                        upper_bound_string = line[51:64].strip()

                        state_string = None
                        marginal_string = None

                        # skip any headers
                        if (index_string == "------") or (index_string == "No."):
                            continue

                        if len(index_string) > 0:
                            index = eval(index_string)

                        if lp_solution is True:

                            state_string = line[20:22].strip()
                            marginal_string = line[65:78].strip()

                            if (activity_string != "< eps") and (len(activity_string) > 0):
                                activity = eval(activity_string)
                            else:
                                activity = 0.0
                                if (lower_bound_string != "< eps") and (len(lower_bound_string) > 0):
                                    lower_bound = eval(lower_bound_string)
                                else:
                                    lower_bound = 0.0
                            if state_string != "NS":
                                if (upper_bound_string != "< eps") and (len(upper_bound_string) > 0):
                                    upper_bound = eval(upper_bound_string)
                                else:
                                    upper_bound = 0.0
                            if (marginal_string != "< eps") and (len(marginal_string) > 0):
                                marginal = eval(marginal_string)
                            else:
                                marginal = 0.0
                        else:
                            if (activity_string != "< eps") and (len(activity_string) > 0):
                                activity = eval(activity_string)
                            else:
                                activity = 0.0

                        variable_name = None
                        if len(active_variable_name) > 0:
                            # if there is an active variable name, the identifier was
                            # too long for everything to be on a single line; the second
                            # line contains all of the value information.
                            variable_name = active_variable_name
                            active_variable_name = ""
                        else:
                            # everything is on a single line.
                            variable_name = name_string
                            number_of_variables_read = number_of_variables_read + 1
                            # sanity check - the indices should be in sequence.
                            if index != number_of_variables_read:
                                raise ValueError("***ERROR: Unexpected variable index encountered on line="+line+"; expected value="+str(number_of_variables_read)+"; actual value="+str(index))

                        if lp_solution is True:
                            # the "activity" column always specifies the variable value.
                            # embedding the if-then-else to validate the basis status.
                            # we are currently ignoring all bound-related information.
                            variable_value = None
                            if state_string in ('B', 'NL', 'NS', 'NU', 'NF'):
                                # NF = non-basic free (unbounded) variable
                                # NL = non-basic variable at its lower bound
                                # NU = non-basic variable at its upper bound
                                # NS = non-basic fixed variable
                                variable_value = activity
                            else:
                                raise ValueError("Unknown status="+state_string+" encountered "
                                                 "for variable="+variable_name+" in the "
                                                 "following line of the GLPK solution file="
                                                 +self._soln_file+":\n"+line)

                            variable = soln.variable[variable_name] = {"Value" : variable_value}
                        else:
                            variable = soln.variable[variable_name] = {"Value" : activity}

                    # if all of the variables have been read, exit.
                    if number_of_variables_read == results.problem.number_of_variables:
                        state = -1

                if state==-1:
                    break

            INPUT.close()

        except ValueError:
            msg = sys.exc_info()[1]
            INPUT.close()
            raise RuntimeError(msg)
        except Exception:
            msg = sys.exc_info()[1]
            INPUT.close()
            raise

        # For the range constraints, supply only the dual with the largest
        # magnitude (at least one should always be numerically zero)
        scon = soln.Constraint
        for key,(ld,ud) in range_duals.items():
            if abs(ld) > abs(ud):
                scon['r_l_'+key] = {"Dual" : ld}
            else:
                scon['r_l_'+key] = {"Dual" : ud}        # Use the same key

        #
        if soln.status is SolutionStatus.optimal:
            soln.gap = 0.0
        elif soln.status is SolutionStatus.stoppedByLimit:
            soln.gap = "Infinity"  # until proven otherwise
            if "lower_bound" in dir(results.problem):
                if results.problem.lower_bound is "-Infinity":
                    soln.gap = "Infinity"
                elif not results.problem.lower_bound is None:
                    if "upper_bound" not in dir(results.problem):
                        gap = "Infinity"
                    elif results.problem.upper_bound is None:
                        gap = "Infinity"
                    else:
                        soln.gap = eval(soln.objective(0)) - \
                                   eval(results.problem.lower_bound)
            elif "upper_bound" in dir(results.problem):
                if results.problem.upper_bound is "Infinity":
                    soln.gap = "Infinity"
                elif not results.problem.upper_bound is None:
                    soln.gap = eval(results.problem.upper_bound) - \
                               eval(soln.objective(0))


class MockGLPK(GLPKSHELL_old,MockMIP):
    """A Mock GLPK solver used for testing
    """

    alias('_mock_glpk')

    def __init__(self, **kwds):
        try:
            GLPKSHELL_old.__init__(self, **kwds)
        except ApplicationError: #pragma:nocover
            pass                        #pragma:nocover
        MockMIP.__init__(self,"glpk")

    def available(self, exception_flag=True):
        return GLPKSHELL_old.available(self,exception_flag)

    def create_command_line(self,executable,problem_files):
        command = GLPKSHELL_old.create_command_line(self,executable,problem_files)
        MockMIP.create_command_line(self,executable,problem_files)
        return command

    def executable(self):
        return MockMIP.executable(self)

    def _execute_command(self,cmd):
        return MockMIP._execute_command(self,cmd)

    def _convert_problem(self,args,pformat,valid_pformats):
        if pformat in [ProblemFormat.mps,ProblemFormat.cpxlp]:
            return (args,pformat,None)
        else:
            return (args,ProblemFormat.cpxlp,None)


register_executable( name='glpsol')
