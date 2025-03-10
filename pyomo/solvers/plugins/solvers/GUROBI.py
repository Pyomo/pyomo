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

import io
import os
import sys
import re
import time
import logging
import subprocess

from pyomo.common import Executable
from pyomo.common.collections import Bunch
from pyomo.common.dependencies import attempt_import
from pyomo.common.enums import maximize, minimize
from pyomo.common.errors import ApplicationError
from pyomo.common.fileutils import this_file_dir
from pyomo.common.log import is_debug_set
from pyomo.common.tee import capture_output, TeeStream
from pyomo.common.tempfiles import TempfileManager

from pyomo.opt.base import ProblemFormat, ResultsFormat, OptSolver
from pyomo.opt.base.solvers import _extract_version, SolverFactory
from pyomo.opt.results import (
    SolverStatus,
    TerminationCondition,
    SolutionStatus,
    Solution,
)
from pyomo.opt.solver import ILMLicensedSystemCallSolver
from pyomo.core.kernel.block import IBlock
from pyomo.core import ConcreteModel, Var, Objective

from pyomo.solvers.plugins.solvers.gurobi_direct import gurobipy, gurobipy_available
from pyomo.solvers.plugins.solvers.ASL import ASL

logger = logging.getLogger('pyomo.solvers')
GUROBI_RUN = attempt_import('pyomo.solvers.plugins.solvers.GUROBI_RUN')[0]


@SolverFactory.register('gurobi', doc='The GUROBI LP/MIP solver')
class GUROBI(OptSolver):
    """The GUROBI LP/MIP solver"""

    def __new__(cls, *args, **kwds):
        mode = kwds.pop('solver_io', 'lp')
        if mode is None:
            mode = 'lp'
        #
        if mode == 'lp':
            if gurobipy_available:
                return SolverFactory('_gurobi_file', **kwds)
            else:
                return SolverFactory('_gurobi_shell', **kwds)
        if mode == 'mps':
            if gurobipy_available:
                opt = SolverFactory('_gurobi_file', **kwds)
            else:
                opt = SolverFactory('_gurobi_shell', **kwds)
            opt.set_problem_format(ProblemFormat.mps)
            return opt
        if mode in ['python', 'direct']:
            opt = SolverFactory('gurobi_direct', **kwds)
            if opt is None:
                logger.error('Python API for GUROBI is not installed')
                return
            return opt
        if mode == 'persistent':
            opt = SolverFactory('gurobi_persistent', **kwds)
            if opt is None:
                logger.error('Python API for GUROBI is not installed')
                return
            return opt
        #
        if mode == 'os':
            opt = SolverFactory('_ossolver', **kwds)
        elif mode == 'nl':
            opt = SolverFactory('_gurobi_nl', **kwds)
        else:
            logger.error('Unknown IO type: %s' % mode)
            return
        # The Gurobi ASL solver was 'gurobi_ampl' through Gurobi 11,
        # then was renamed to 'gurobi'.  Check 'gurobi' first, then
        # 'gurobi_ampl'.
        for exe_name in ('gurobi', 'gurobi_ampl'):
            exe = Executable(exe_name)
            if (
                exe.available()
                and b'[-AMPL]'
                in subprocess.run(
                    exe.executable,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=1,
                ).stdout
            ):
                opt.set_options(f'solver={exe_name}')
                break
        else:
            # Fall back on 'gurobi' (matching the current Gurobi
            # release) if neither appears to be available.
            opt.set_options('solver=gurobi')
        return opt


@SolverFactory.register('_gurobi_nl', doc='NL interface to the Gurobi solver')
class GUROBINL(ASL):
    """NL interface to gurobi_ampl."""

    def license_is_valid(self):
        m = ConcreteModel()
        m.x = Var(bounds=(1, 2))
        m.obj = Objective(expr=m.x)
        try:
            with capture_output():
                self.solve(m)
            return abs(m.x.value - 1) <= 1e-4
        except:
            return False


@SolverFactory.register(
    '_gurobi_shell', doc='Shell interface to the GUROBI LP/MIP solver'
)
class GUROBISHELL(ILMLicensedSystemCallSolver):
    """Shell interface to the GUROBI LP/MIP solver"""

    _solver_info_cache = {}

    def __init__(self, **kwds):
        #
        # Call base class constructor
        #
        kwds['type'] = 'gurobi'
        ILMLicensedSystemCallSolver.__init__(self, **kwds)

        # NOTE: eventually both of the following attributes should be
        # migrated to a common base class.  is the current solve
        # warm-started? a transient data member to communicate state
        # information across the _presolve, _apply_solver, and
        # _postsolve methods.
        self._warm_start_solve = False
        # related to the above, the temporary name of the MST warm-start
        # file (if any).
        self._warm_start_file_name = None

        #
        # Define valid problem formats and associated results formats
        #
        self._valid_problem_formats = [ProblemFormat.cpxlp, ProblemFormat.mps]
        self._valid_result_formats = {}
        self._valid_result_formats[ProblemFormat.cpxlp] = [ResultsFormat.soln]
        self._valid_result_formats[ProblemFormat.mps] = [ResultsFormat.soln]
        self.set_problem_format(ProblemFormat.cpxlp)

        # Note: Undefined capabilities default to 'None'
        self._capabilities = Bunch()
        self._capabilities.linear = True
        self._capabilities.quadratic_objective = True
        self._capabilities.quadratic_constraint = True
        self._capabilities.integer = True
        self._capabilities.sos1 = True
        self._capabilities.sos2 = True

    def license_is_valid(self):
        """
        Runs a check for a valid Gurobi license using the
        given executable (default is 'gurobi_cl'). All
        output is hidden. If the test fails for any reason
        (including the executable being invalid), then this
        function will return False.
        """
        solver_exec = self.executable()
        if (solver_exec, 'licensed') in self._solver_info_cache:
            return self._solver_info_cache[(solver_exec, 'licensed')]

        if not solver_exec:
            licensed = False
        else:
            executable = os.path.join(os.path.dirname(solver_exec), 'gurobi_cl')
            try:
                rc = subprocess.call(
                    [executable, "--license"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
            except OSError:
                try:
                    rc = subprocess.run(
                        [solver_exec],
                        input=('import gurobipy; gurobipy.Env().dispose(); quit()'),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        universal_newlines=True,
                    ).returncode
                except OSError:
                    rc = 1
            licensed = not rc

        self._solver_info_cache[(solver_exec, 'licensed')] = licensed
        return licensed

    def _default_results_format(self, prob_format):
        return ResultsFormat.soln

    def warm_start_capable(self):
        return True

    #
    # write a warm-start file in the GUROBI MST format, which is *not*
    # the same as the CPLEX MST format.
    #
    def _warm_start(self, instance):
        from pyomo.core.base import Var

        # for each variable in the symbol_map, add a child to the
        # variables element.  Both continuous and discrete are accepted
        # (and required, depending on other options).
        #
        # **Note**: This assumes that the symbol_map is "clean", i.e.,
        # contains only references to the variables encountered in
        # constraints
        output_index = 0
        if isinstance(instance, IBlock):
            smap = getattr(instance, "._symbol_maps")[self._smap_id]
        else:
            smap = instance.solutions.symbol_map[self._smap_id]
        byObject = smap.byObject
        with open(self._warm_start_file_name, 'w') as mst_file:
            for vdata in instance.component_data_objects(Var):
                if (vdata.value is not None) and (id(vdata) in byObject):
                    name = byObject[id(vdata)]
                    mst_file.write("%s %s\n" % (name, vdata.value))

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

        # the input argument can currently be one of two things: an
        # instance or a filename.  if a filename is provided and a
        # warm-start is indicated, we go ahead and create the temporary
        # file - assuming that the user has already, via some external
        # mechanism, invoked warm_start() with a instance to create the
        # warm start file.
        if self._warm_start_solve and isinstance(args[0], str):
            # we assume the user knows what they are doing...
            pass
        elif self._warm_start_solve and (not isinstance(args[0], str)):
            # assign the name of the warm start file *before* calling
            # the base class presolve - the base class method ends up
            # creating the command line, and the warm start file-name is
            # (obviously) needed there.
            if self._warm_start_file_name is None:
                assert not user_warmstart
                self._warm_start_file_name = TempfileManager.create_tempfile(
                    suffix='.gurobi.mst'
                )

        # let the base class handle any remaining keywords/actions.
        ILMLicensedSystemCallSolver._presolve(self, *args, **kwds)

        # NB: we must let the base class presolve run first so that the
        # symbol_map is actually constructed!

        if (len(args) > 0) and (not isinstance(args[0], str)):
            if len(args) != 1:
                raise ValueError(
                    "GUROBI _presolve method can only handle a single "
                    "problem instance - %s were supplied" % (len(args),)
                )

            # write the warm-start file - currently only supports MIPs.
            # we only know how to deal with a single problem instance.
            if self._warm_start_solve and (not user_warmstart):
                start_time = time.time()
                self._warm_start(args[0])
                end_time = time.time()
                if self._report_timing is True:
                    print(
                        "Warm start write time=%.2f seconds" % (end_time - start_time)
                    )

    def _default_executable(self):
        if sys.platform == 'win32':
            executable = Executable("gurobi.bat")
        else:
            executable = Executable("gurobi.sh")
        if executable:
            return executable.path()
        if gurobipy_available:
            return sys.executable
        logger.warning(
            "Could not locate the 'gurobi' executable, "
            "which is required for solver %s" % self.name
        )
        self.enable = False
        return None

    def _get_version(self):
        """
        Returns a tuple describing the solver executable version.
        """
        solver_exec = self.executable()
        if (solver_exec, 'version') in self._solver_info_cache:
            return self._solver_info_cache[(solver_exec, 'version')]

        if solver_exec is None:
            ver = _extract_version('')
        else:
            results = subprocess.run(
                [solver_exec],
                input=('import gurobipy; print(gurobipy.gurobi.version()); quit()'),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
            )
            ver = None
            try:
                ver = tuple(eval(results.stdout.strip()))
                while len(ver) < 4:
                    ver += (0,)
            except SyntaxError:
                ver = _extract_version('')
        if ver is not None:
            ver = ver[:4]
        self._solver_info_cache[(solver_exec, 'version')] = ver
        return ver

    def create_command_line(self, executable, problem_files):
        #
        # Define log file
        #
        if self._log_file is None:
            self._log_file = TempfileManager.create_tempfile(suffix='.gurobi.log')

        #
        # Define solution file
        # As indicated above, contains (in XML) both the solution and
        # solver status.
        #
        if self._soln_file is None:
            self._soln_file = TempfileManager.create_tempfile(suffix='.gurobi.txt')

        #
        # Write the GUROBI execution script
        #

        problem_filename = self._problem_files[0]
        solution_filename = self._soln_file
        warmstart_filename = self._warm_start_file_name

        # translate the options into a normal python dictionary, from a
        # pyomo.common.collections.Bunch - the gurobi_run function
        # doesn't know about pyomo, so the translation is necessary
        # (`repr(options)` doesn't produce executable python code)
        options_dict = dict(self.options)

        # NOTE: the gurobi shell is independent of Pyomo python
        #       virtualized environment, so any imports - specifically
        #       that required to get GUROBI_RUN - must be handled
        #       explicitly.
        # NOTE: The gurobi plugin (GUROBI.py) and GUROBI_RUN.py live in
        #       the same directory.
        script = "import sys\n"
        script += "sys.path.append(%r)\n" % (this_file_dir(),)
        script += "import GUROBI_RUN\n"
        script += "soln = GUROBI_RUN.gurobi_run("
        mipgap = float(self.options.mipgap) if self.options.mipgap is not None else None
        for x in (
            problem_filename,
            warmstart_filename,
            None,
            options_dict,
            self._suffixes,
        ):
            script += "%r," % x
        script += ")\n"
        script += "GUROBI_RUN.write_result(soln, %r)\n" % solution_filename
        script += "quit()\n"

        # dump the script and warm-start file names for the
        # user if we're keeping files around.
        if self._keepfiles:
            script_fname = TempfileManager.create_tempfile(suffix='.gurobi.script')
            script_file = open(script_fname, 'w')
            script_file.write(script)
            script_file.close()

            print("Solver script file: '%s'" % script_fname)
            if self._warm_start_solve and (self._warm_start_file_name is not None):
                print("Solver warm-start file: " + self._warm_start_file_name)

        #
        # Define command line
        #
        cmd = [executable]
        if self._timer:
            cmd.insert(0, self._timer)
        return Bunch(cmd=cmd, script=script, log_file=self._log_file, env=None)

    def process_soln_file(self, results):
        # the only suffixes that we extract are constraint duals,
        # constraint slacks, and variable reduced-costs. scan through
        # the solver suffix list and throw an exception if the user has
        # specified any others.
        extract_duals = False
        extract_slacks = False
        extract_rc = False
        for suffix in self._suffixes:
            flag = False
            if re.match(suffix, "dual"):
                extract_duals = True
                flag = True
            if re.match(suffix, "slack"):
                extract_slacks = True
                flag = True
            if re.match(suffix, "rc"):
                extract_rc = True
                flag = True
            if not flag:
                raise RuntimeError(
                    "***The GUROBI solver plugin cannot extract solution suffix="
                    + suffix
                )

        # check for existence of the solution file
        # not sure why we just return - would think that we
        # would want to indicate some sort of error
        if not os.path.exists(self._soln_file):
            return

        soln = Solution()

        # caching for efficiency
        soln_variables = soln.variable
        soln_constraints = soln.constraint

        num_variables_read = 0

        # string compares are too expensive, so simply introduce some
        # section IDs.
        # 0 - unknown
        # 1 - problem
        # 2 - solution
        # 3 - solver

        section = 0  # unknown

        solution_seen = False

        range_duals = {}
        range_slacks = {}

        INPUT = open(self._soln_file, "r")
        for line in INPUT:
            line = line.strip()
            tokens = [token.strip() for token in line.split(":")]
            if tokens[0] == 'section':
                if tokens[1] == 'problem':
                    section = 1
                elif tokens[1] == 'solution':
                    section = 2
                    solution_seen = True
                elif tokens[1] == 'solver':
                    section = 3
            else:
                if section == 2:
                    if tokens[0] == 'var':
                        if tokens[1] != "ONE_VAR_CONSTANT":
                            soln_variables[tokens[1]] = {"Value": float(tokens[2])}
                            num_variables_read += 1
                    elif tokens[0] == 'status':
                        soln.status = getattr(SolutionStatus, tokens[1])
                    elif tokens[0] == 'gap':
                        soln.gap = float(tokens[1])
                    elif tokens[0] == 'objective':
                        if tokens[1].strip() != 'None':
                            soln.objective['__default_objective__'] = {
                                'Value': float(tokens[1])
                            }
                            if results.problem.sense == minimize:
                                results.problem.upper_bound = float(tokens[1])
                            else:
                                results.problem.lower_bound = float(tokens[1])
                    elif tokens[0] == 'constraintdual':
                        name = tokens[1]
                        if name != "c_e_ONE_VAR_CONSTANT":
                            if name.startswith('c_'):
                                soln_constraints.setdefault(tokens[1], {})["Dual"] = (
                                    float(tokens[2])
                                )
                            elif name.startswith('r_l_'):
                                range_duals.setdefault(name[4:], [0, 0])[0] = float(
                                    tokens[2]
                                )
                            elif name.startswith('r_u_'):
                                range_duals.setdefault(name[4:], [0, 0])[1] = float(
                                    tokens[2]
                                )
                    elif tokens[0] == 'constraintslack':
                        name = tokens[1]
                        if name != "c_e_ONE_VAR_CONSTANT":
                            if name.startswith('c_'):
                                soln_constraints.setdefault(tokens[1], {})["Slack"] = (
                                    float(tokens[2])
                                )
                            elif name.startswith('r_l_'):
                                range_slacks.setdefault(name[4:], [0, 0])[0] = float(
                                    tokens[2]
                                )
                            elif name.startswith('r_u_'):
                                range_slacks.setdefault(name[4:], [0, 0])[1] = float(
                                    tokens[2]
                                )
                    elif tokens[0] == 'varrc':
                        if tokens[1] != "ONE_VAR_CONSTANT":
                            soln_variables[tokens[1]]["Rc"] = float(tokens[2])
                    else:
                        setattr(soln, tokens[0], tokens[1])
                elif section == 1:
                    if tokens[0] == 'sense':
                        if tokens[1] == 'minimize':
                            results.problem.sense = minimize
                        elif tokens[1] == 'maximize':
                            results.problem.sense = maximize
                    else:
                        try:
                            val = eval(tokens[1])
                        except:
                            val = tokens[1]
                        setattr(results.problem, tokens[0], val)
                elif section == 3:
                    if tokens[0] == 'status':
                        results.solver.status = getattr(SolverStatus, tokens[1])
                    elif tokens[0] == 'termination_condition':
                        try:
                            results.solver.termination_condition = getattr(
                                TerminationCondition, tokens[1]
                            )
                        except AttributeError:
                            results.solver.termination_condition = (
                                TerminationCondition.unknown
                            )
                    else:
                        setattr(results.solver, tokens[0], tokens[1])

        INPUT.close()

        # For the range constraints, supply only the dual with the largest
        # magnitude (at least one should always be numerically zero)
        for key, (ld, ud) in range_duals.items():
            if abs(ld) > abs(ud):
                soln_constraints['r_l_' + key] = {"Dual": ld}
            else:
                # Use the same key
                soln_constraints['r_l_' + key] = {"Dual": ud}
        # slacks
        for key, (ls, us) in range_slacks.items():
            if abs(ls) > abs(us):
                soln_constraints.setdefault('r_l_' + key, {})["Slack"] = ls
            else:
                # Use the same key
                soln_constraints.setdefault('r_l_' + key, {})["Slack"] = us

        if solution_seen:
            results.solution.insert(soln)

    def _postsolve(self):
        # take care of the annoying GUROBI log file in the current
        # directory.  this approach doesn't seem overly efficient, but
        # python os module functions doesn't accept regular expression
        # directly.
        filename_list = os.listdir(".")
        for filename in filename_list:
            # IMPT: trap the possible exception raised by the file not
            #       existing.  this can occur in pyro environments where
            #       > 1 workers are running GUROBI, and were started
            #       from the same directory.  these logs don't matter
            #       anyway (we redirect everything), and are largely an
            #       annoyance.
            try:
                if re.match(r'gurobi\.log', filename) != None:
                    os.remove(filename)
            except OSError:
                pass

        # let the base class deal with returning results.
        results = ILMLicensedSystemCallSolver._postsolve(self)

        # finally, clean any temporary files registered with the temp file
        # manager, created populated *directly* by this plugin. does not
        # include, for example, the execution script. but does include
        # the warm-start file.
        TempfileManager.pop(remove=not self._keepfiles)

        return results


@SolverFactory.register(
    '_gurobi_file', doc='LP/MPS file-based direct interface to the GUROBI LP/MIP solver'
)
class GUROBIFILE(GUROBISHELL):
    """Direct LP/MPS file-based interface to the GUROBI LP/MIP solver"""

    def available(self, exception_flag=False):
        if not gurobipy_available:  # this triggers the deferred import
            if exception_flag:
                raise ApplicationError("gurobipy module not importable")
            return False
        if getattr(self, '_available', None) is None:
            self._check_license()
        ans = self._available[0]
        if exception_flag and not ans:
            raise ApplicationError(msg % self.name)
        return ans

    def license_is_valid(self):
        return self.available(False) and self._available[1]

    def _check_license(self):
        licensed = False
        try:
            # Gurobipy writes out license file information when creating
            # the environment
            with capture_output(capture_fd=True):
                m = gurobipy.Model()
            licensed = True
        except gurobipy.GurobiError:
            licensed = False

        self._available = (True, licensed)

    def _get_version(self):
        return (
            gurobipy.GRB.VERSION_MAJOR,
            gurobipy.GRB.VERSION_MINOR,
            gurobipy.GRB.VERSION_TECHNICAL,
        )

    def _default_executable(self):
        # Bogus, but not None (because the test infrastructure disables
        # solvers where the executable() is None)
        return ""

    def create_command_line(self, executable, problem_files):
        #
        # Define log file
        #
        if self._log_file is None:
            self._log_file = TempfileManager.create_tempfile(suffix='.gurobi.log')

        #
        # Define command line
        #
        return Bunch(cmd=[], script="", log_file=self._log_file, env=None)

    def _apply_solver(self):
        #
        # Execute the command
        #
        if is_debug_set(logger):
            logger.debug("Running %s", self._command.cmd)

        problem_filename = self._problem_files[0]
        warmstart_filename = self._warm_start_file_name

        # translate the options into a normal python dictionary, from a
        # pyutilib SectionWrapper - because the gurobi_run function was
        # originally designed to run in the Python environment
        # distributed in the Gurobi installation (which doesn't know
        # about pyomo) the translation is necessary.
        options_dict = {}
        for key in self.options:
            options_dict[key] = self.options[key]

        # display the log/solver file names prior to execution. this is useful
        # in case something crashes unexpectedly, which is not without precedent.
        if self._keepfiles:
            if self._log_file is not None:
                print("Solver log file: '%s'" % self._log_file)
            if self._problem_files != []:
                print("Solver problem files: %s" % str(self._problem_files))

        sys.stdout.flush()
        ostreams = [io.StringIO()]
        if self._tee:
            ostreams.append(sys.stdout)
        with capture_output(output=TeeStream(*ostreams), capture_fd=False):
            self._soln = GUROBI_RUN.gurobi_run(
                problem_filename, warmstart_filename, None, options_dict, self._suffixes
            )
        self._log = ostreams[0].getvalue()
        self._rc = 0
        sys.stdout.flush()
        return Bunch(rc=self._rc, log=self._log)

    def process_soln_file(self, results):
        # the only suffixes that we extract are constraint duals,
        # constraint slacks, and variable reduced-costs. Scan through
        # the solver suffix list and throw an exception if the user has
        # specified any others.
        extract_duals = False
        extract_slacks = False
        extract_rc = False
        for suffix in self._suffixes:
            flag = False
            if re.match(suffix, "dual"):
                extract_duals = True
                flag = True
            if re.match(suffix, "slack"):
                extract_slacks = True
                flag = True
            if re.match(suffix, "rc"):
                extract_rc = True
                flag = True
            if not flag:
                raise RuntimeError(
                    "***The GUROBI solver plugin cannot extract solution suffix="
                    + suffix
                )

        soln = Solution()

        # caching for efficiency
        soln_variables = soln.variable
        soln_constraints = soln.constraint

        num_variables_read = 0

        # string compares are too expensive, so simply introduce some
        # section IDs.
        # 0 - unknown
        # 1 - problem
        # 2 - solution
        # 3 - solver

        section = 0  # unknown

        solution_seen = False

        range_duals = {}
        range_slacks = {}

        # Copy over the problem info
        for key, val in self._soln['problem'].items():
            setattr(results.problem, key, val)
        if results.problem.sense == 'minimize':
            results.problem.sense = minimize
        elif results.problem.sense == 'maximize':
            results.problem.sense = maximize

        # Copy over the solver info
        for key, val in self._soln['solver'].items():
            setattr(results.solver, key, val)
        results.solver.status = getattr(SolverStatus, results.solver.status)
        try:
            results.solver.termination_condition = getattr(
                TerminationCondition, results.solver.termination_condition
            )
        except AttributeError:
            results.solver.termination_condition = TerminationCondition.unknown

        # Copy over the solution information
        sol = self._soln.get('solution', None)
        if sol:
            if 'status' in sol:
                soln.status = sol['status']
            if 'gap' in sol:
                soln.gap = sol['gap']
            obj = sol.get('objective', None)
            if obj is not None:
                soln.objective['__default_objective__'] = {'Value': obj}
                if results.problem.sense == minimize:
                    results.problem.upper_bound = obj
                else:
                    results.problem.lower_bound = obj
            for name, val in sol.get('var', {}).items():
                if name == "ONE_VAR_CONSTANT":
                    continue
                soln_variables[name] = {"Value": val}
                num_variables_read += 1
            for name, val in sol.get('varrc', {}).items():
                if name == "ONE_VAR_CONSTANT":
                    continue
                soln_variables[name]["Rc"] = val
            for name, val in sol.get('constraintdual', {}).items():
                if name == "c_e_ONE_VAR_CONSTANT":
                    continue
                if name.startswith('c_'):
                    soln_constraints.setdefault(name, {})["Dual"] = val
                elif name.startswith('r_l_'):
                    range_duals.setdefault(name[4:], [0, 0])[0] = val
                elif name.startswith('r_u_'):
                    range_duals.setdefault(name[4:], [0, 0])[1] = val
            for name, val in sol.get('constraintslack', {}).items():
                if name == "c_e_ONE_VAR_CONSTANT":
                    continue
                if name.startswith('c_'):
                    soln_constraints.setdefault(name, {})["Slack"] = val
                elif name.startswith('r_l_'):
                    range_slacks.setdefault(name[4:], [0, 0])[0] = val
                elif name.startswith('r_u_'):
                    range_slacks.setdefault(name[4:], [0, 0])[1] = val

            results.solution.insert(soln)

        # For the range constraints, supply only the dual with the largest
        # magnitude (at least one should always be numerically zero)
        for key, (ld, ud) in range_duals.items():
            if abs(ld) > abs(ud):
                soln_constraints['r_l_' + key] = {"Dual": ld}
            else:
                # Use the same key
                soln_constraints['r_l_' + key] = {"Dual": ud}
        # slacks
        for key, (ls, us) in range_slacks.items():
            if abs(ls) > abs(us):
                soln_constraints.setdefault('r_l_' + key, {})["Slack"] = ls
            else:
                # Use the same key
                soln_constraints.setdefault('r_l_' + key, {})["Slack"] = us
