# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import logging
import os
import subprocess
import datetime
import time
import io
import re
import sys
import time
import threading
from typing import Optional, Tuple, Union, Mapping, List, Dict, Any, Sequence

from pyomo.common import Executable
from pyomo.common.config import (
    ConfigDict,
    ConfigList,
    ConfigValue,
    document_configdict,
    document_class_CONFIG,
    ADVANCED_OPTION,
)
from pyomo.common.errors import (
    ApplicationError,
    InfeasibleConstraintException,
    MouseTrap,
)
from pyomo.common.fileutils import to_legal_filename
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.timing import HierarchicalTimer, default_timer
from pyomo.core.base.var import VarData
from pyomo.core.staleflag import StaleFlagManager
from pyomo.repn.plugins.nl_writer import NLWriter, NLWriterInfo
from pyomo.contrib.solver.common.base import SolverBase, Availability
from pyomo.contrib.solver.common.config import SolverConfig
from pyomo.contrib.solver.common.factory import LegacySolverWrapper
from pyomo.contrib.solver.common.results import (
    Results,
    TerminationCondition,
    SolutionStatus,
)
from pyomo.contrib.solver.solvers.asl_sol_reader import (
    asl_solve_code_to_solution_status,
    parse_asl_sol_file,
    ASLSolFileData,
    ASLSolFileSolutionLoader,
)
from pyomo.contrib.solver.common.util import NoOptimalSolutionError, NoSolutionError
from pyomo.common.tee import TeeStream
from pyomo.core.expr.visitor import replace_expressions
from pyomo.core.expr.numvalue import value
from pyomo.core.base.suffix import Suffix
from pyomo.common.collections import ComponentMap
from pyomo.solvers.amplfunc_merge import amplfunc_merge

logger = logging.getLogger(__name__)

# Acceptable chars for the end of the alpha_pr column
# in ipopt's output, per https://coin-or.github.io/Ipopt/OUTPUT.html
_ALPHA_PR_CHARS = set("fFhHkKnNRwSstTr")


def _option_to_cmd(opt: str, val: str | int | float):
    """Convert a option / value pair into a valid command line argument."""
    if isinstance(val, str):
        if '"' not in val:
            return f'{opt}="{val}"'
        elif "'" not in val:
            return f"{opt}='{val}'"
        else:
            raise ValueError(
                f"solver_option '{opt}' contained value {val!r} with "
                "both single and double quotes.  Ipopt cannot parse "
                "command line options with escaped quote characters."
            )
    else:
        return f'{opt}={val}'


@document_configdict()
class IpoptConfig(SolverConfig):
    def __init__(
        self,
        description=None,
        doc=None,
        implicit=False,
        implicit_domain=None,
        visibility=0,
    ):
        super().__init__(
            description=description,
            doc=doc,
            implicit=implicit,
            implicit_domain=implicit_domain,
            visibility=visibility,
        )

        self.executable: Executable = self.declare(
            'executable',
            ConfigValue(
                domain=Executable,
                default='ipopt',
                description="Preferred executable for ipopt. Defaults to searching "
                "the ``PATH`` for the first available ``ipopt``.",
            ),
        )
        self.writer_config: ConfigDict = self.declare(
            'writer_config', NLWriter.CONFIG()
        )


class IpoptSolutionLoader(ASLSolFileSolutionLoader):
    def get_reduced_costs(
        self, vars_to_load: Optional[Sequence[VarData]] = None
    ) -> Mapping[VarData, float]:
        if self._nl_info.eliminated_vars:
            raise MouseTrap(
                'Complete reduced costs are not available when variables have '
                'been presolved from the model.  Turn presolve off '
                '(solver.config.writer_config.linear_presolve=False) to get '
                'reduced costs.'
            )

        zl_map = self._sol_data.var_suffixes.get('ipopt_zL_out', {})
        zu_map = self._sol_data.var_suffixes.get('ipopt_zU_out', {})
        # TBD: is it an error if Ipopt fails to return RC info?
        # if not (zl_map or zu_map):
        #     raise?
        if self._nl_info.scaling:
            # Unscale the zl and zu maps:
            inv_obj_scale = 1.0
            if self._nl_info.scaling.objectives:
                inv_obj_scale /= self._nl_info.scaling.objectives[self._sol_data.objno]
            var_scale = self._nl_info.scaling.variables
            zl_map = {k: v * var_scale[k] * inv_obj_scale for k, v in zl_map.items()}
            zu_map = {k: v * var_scale[k] * inv_obj_scale for k, v in zu_map.items()}

        rc = ComponentMap()
        for ndx, v in enumerate(self._nl_info.variables):
            _rc = 0.0
            if ndx in zl_map:
                # Note *any* value in zl has an absolute value at least
                # as big as 0.  No need to test and just overwrite _rc:
                _rc = zl_map[ndx]
            if ndx in zu_map:
                zu = zu_map[ndx]
                if abs(zu) > abs(_rc):
                    _rc = zu
            rc[v] = _rc

        if vars_to_load is not None:
            # Note vars_to_load could contain variables that were
            # eliminated (so use get()):
            rc = ComponentMap((v, rc.get(v, 0)) for v in vars_to_load)
        return rc


#: The set of all ipopt options that can be passed to Ipopt on the command line
ipopt_command_line_options = {
    'acceptable_compl_inf_tol',
    'acceptable_constr_viol_tol',
    'acceptable_dual_inf_tol',
    'acceptable_tol',
    'alpha_for_y',
    'bound_frac',
    'bound_mult_init_val',
    'bound_push',
    'bound_relax_factor',
    'compl_inf_tol',
    'constr_mult_init_max',
    'constr_viol_tol',
    'diverging_iterates_tol',
    'dual_inf_tol',
    'expect_infeasible_problem',
    'file_print_level',
    'halt_on_ampl_error',
    'hessian_approximation',
    'honor_original_bounds',
    'linear_scaling_on_demand',
    'linear_solver',
    'linear_system_scaling',
    'ma27_pivtol',
    'ma27_pivtolmax',
    'ma57_pivot_order',
    'ma57_pivtol',
    'ma57_pivtolmax',
    'max_cpu_time',
    'max_iter',
    'max_refinement_steps',
    'max_soc',
    'maxit',
    'min_refinement_steps',
    'mu_init',
    'mu_max',
    'mu_oracle',
    'mu_strategy',
    'nlp_scaling_max_gradient',
    'nlp_scaling_method',
    'obj_scaling_factor',
    'option_file_name',
    'outlev',
    'output_file',
    'pardiso_matching_strategy',
    'print_level',
    'print_options_documentation',
    'print_user_options',
    'required_infeasibility_reduction',
    'slack_bound_frac',
    'slack_bound_push',
    'tol',
    'wantsol',
    'warm_start_bound_push',
    'warm_start_init_point',
    'warm_start_mult_bound_push',
    'watchdog_shortened_iter_trigger',
}

#: The set of options we forbid the user from setting (with reasons)
unallowed_ipopt_options = {
    'wantsol': 'The solver interface requires the sol file to be created',
    'option_file_name': (
        'Pyomo generates the ipopt options file as part of the `solve` '
        'method.  Add all options to config.solver_options instead.'
    ),
}


@document_class_CONFIG(methods=['solve'])
class Ipopt(SolverBase):
    """Interface to the Ipopt NLP solver (NL file based)"""

    #: Global class configuration;
    #: see :ref:`pyomo.contrib.solver.solvers.ipopt.Ipopt::CONFIG`.
    CONFIG = IpoptConfig()

    #: cache of availability / version information
    _exe_cache: dict[str : tuple[int] | None] = {}

    #: default timeout to use when attempting to get the ipopt version number
    _version_timeout = 2

    def __init__(self, **kwds: Any) -> None:
        super().__init__(**kwds)

        #: Instance configuration;
        #: see :ref:`pyomo.contrib.solver.solvers.ipopt.Ipopt::CONFIG`.
        self.config = self.config

    def available(self) -> Availability:
        return (
            Availability.NotFound
            if self.version() is None
            else Availability.FullLicense
        )

    def version(self) -> tuple[int, int, int] | None:
        return self._get_version(self.config.executable.path())

    def _get_version(self, exe):
        try:
            return self._exe_cache[exe]
        except KeyError:
            pass
        if exe is None:
            # No executable (either we couldn't find a matching file, or
            # the file is not executable)
            self._exe_cache[None] = None
            return None
        # Run the executable and look for the version
        results = subprocess.run(
            [str(exe), '--version'],
            timeout=self._version_timeout,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            check=False,
        )
        # Note that we expect the command to run without error, AND that
        # it returns a string starting "ipopt <version>".  That prevents
        # us from trying to use other (even ASL) executables as if they
        # were ipopt
        fields = results.stdout.split(maxsplit=2)
        if results.returncode:
            ver = None
        elif len(fields) != 3 or fields[0].lower() != 'ipopt':
            ver = None
        else:
            try:
                ver = tuple(int(i) for i in fields[1].split('.'))
            except (ValueError, TypeError):
                ver = None
        if ver is None:
            logger.warning(
                f"Failed parsing Ipopt version: '{exe} --version':\n\n{results.stdout}"
            )
        self._exe_cache[exe] = ver
        return ver

    def has_linear_solver(self, linear_solver: str) -> bool:
        """Determine if Ipopt has access to the specified linear solver

        This solves a small problem to detect if the Ipopt executable
        has access to the specified linear solver.

        Parameters
        ----------
        linear_solver : str

            The linear solver to test.  Accepts any string that is valid
            for the ``linear_solver`` Ipopt option.

        """
        import pyomo.core as AML

        m = AML.ConcreteModel()
        m.x = AML.Var()
        m.o = AML.Objective(expr=(m.x - 2) ** 2)
        results = self.solve(
            m,
            tee=False,
            raise_exception_on_nonoptimal_result=False,
            load_solutions=False,
            solver_options={'linear_solver': linear_solver},
        )
        return 'running with linear solver' in results.solver_log

    def solve(self, model, **kwds) -> Results:
        "Solve a model using Ipopt"
        # Begin time tracking
        start_time = default_timer()
        # Allocate the results object so we can populate it as we go
        results = Results()
        results.timing_info.start_timestamp = datetime.datetime.now(
            datetime.timezone.utc
        )
        results.solver_name = self.name

        # Update configuration options, based on keywords passed to solve
        config: IpoptConfig = self.config(value=kwds, preserve_implicit=True)

        timer = config.timer
        if timer is None:
            timer = config.timer = HierarchicalTimer()

        # As we are about to run a solver, update the stale flag
        StaleFlagManager.mark_all_as_stale()

        with TempfileManager.new_context() as tempfile:
            if config.working_dir is None:
                dname = tempfile.mkdtemp()
            else:
                dname = config.working_dir
            if not os.path.exists(dname):
                os.mkdir(dname)
            # Because we are just "making up" a file name, it is better
            # to always generate a consistent and legal name, rather
            # than blindly follow what the user gave us.  We will use
            # `universal=True` here to make sure that double quotes are
            # translated, thereby guaranteeing that we should always
            # generate a legal base name (unless, of course, the user
            # put double quotes somewhere else in the path)
            basename = to_legal_filename(model.name, universal=True)
            nlfd, nl_fname = tempfile.mkstemp(
                suffix='.nl', prefix=basename, dir=dname, text=True, delete=False
            )
            results.extra_info.base_file_name = basename = nl_fname[:-3]
            for ext in ('.row', '.col', '.sol', '.opt'):
                if os.path.exists(basename + ext):
                    raise RuntimeError(
                        f"Solver interface file {basename + ext} already exists!"
                    )
            # Note: the ASL has an issue where string constants written
            # to the NL file (e.g. arguments in external functions) MUST
            # be terminated with '\n' regardless of platform.  We will
            # disable universal newlines in the NL file to prevent
            # Python from mapping those '\n' to '\r\n' on Windows.
            with (
                os.fdopen(nlfd, 'w', newline='\n', encoding='utf-8') as nl_file,
                open(basename + '.row', 'w', encoding='utf-8') as row_file,
                open(basename + '.col', 'w', encoding='utf-8') as col_file,
            ):
                timer.start('write_nl_file')
                try:
                    # Note: this is mapping the top-level
                    # symbolic_solver_labels onto the solver's writer
                    # config, and then that config is being used (in
                    # it's entirety) to set the NLWriter's CONFIG.
                    nl_info = NLWriter().write(
                        model,
                        nl_file,
                        row_file,
                        col_file,
                        config=config.writer_config,
                        symbolic_solver_labels=config.symbolic_solver_labels,
                    )
                    proven_infeasible = False
                except InfeasibleConstraintException:
                    proven_infeasible = True
                    nl_info = NLWriterInfo()
                timer.stop('write_nl_file')

            if proven_infeasible:
                results.termination_condition = TerminationCondition.provenInfeasible
                results.solution_status = SolutionStatus.noSolution
                results.extra_info.iteration_count = 0
            elif not nl_info.variables:
                if nl_info.eliminated_vars:
                    results.termination_condition = (
                        TerminationCondition.convergenceCriteriaSatisfied
                    )
                    results.solution_status = SolutionStatus.optimal
                    results.solution_loader = IpoptSolutionLoader(
                        sol_data=ASLSolFileData(), nl_info=nl_info
                    )
                else:
                    results.termination_condition = TerminationCondition.emptyModel
                    results.solution_status = SolutionStatus.noSolution
                results.extra_info.iteration_count = 0
            else:
                self._run_ipopt(results, config, nl_info, basename, timer)

        if (
            config.raise_exception_on_nonoptimal_result
            and results.solution_status != SolutionStatus.optimal
        ):
            raise NoOptimalSolutionError()

        if config.load_solutions:
            if results.solution_status == SolutionStatus.noSolution:
                raise NoSolutionError()
            results.solution_loader.load_vars()
            if (
                hasattr(model, 'dual')
                and isinstance(model.dual, Suffix)
                and model.dual.import_enabled()
            ):
                model.dual.update(results.solution_loader.get_duals())
            if (
                hasattr(model, 'rc')
                and isinstance(model.rc, Suffix)
                and model.rc.import_enabled()
            ):
                model.rc.update(results.solution_loader.get_reduced_costs())

        if (
            results.solution_status in {SolutionStatus.feasible, SolutionStatus.optimal}
            and len(nl_info.objectives) > 0
        ):
            if config.load_solutions:
                results.incumbent_objective = value(nl_info.objectives[0])
            else:
                results.incumbent_objective = value(
                    replace_expressions(
                        nl_info.objectives[0].expr,
                        substitution_map={
                            id(v): val
                            for v, val in results.solution_loader.get_primals().items()
                        },
                        descend_into_named_expressions=True,
                        remove_named_expressions=True,
                    )
                )

        results.solver_config = config

        # Capture/record end-time / wall-time
        results.timing_info.timer = timer
        results.timing_info.wall_time = default_timer() - start_time
        return results

    def _process_options(
        self, option_fname: str, options: dict[str, str | int | float]
    ) -> list[str]:
        # Look through the solver options and separate the command line
        # options from the options that must be sent via an options
        # file.  Raise an exception for any unallowable options.
        options_file_options = []
        cmd_line_options = []
        for key, val in options.items():
            if key in unallowed_ipopt_options:
                msg = unallowed_ipopt_options[key]
                raise ValueError(f"unallowed Ipopt option '{key}': {msg}")
            elif key in ipopt_command_line_options:
                cmd_line_options.append(_option_to_cmd(key, val))
            else:
                options_file_options.append(f"{key} {val}\n")
        # create the options file (if we need it)
        if options_file_options:
            with open(option_fname, 'w', encoding='utf-8') as OPT_FILE:
                OPT_FILE.writelines(options_file_options)
            cmd_line_options.append(_option_to_cmd('option_file_name', option_fname))
        # Return the (formatted) command line options
        return cmd_line_options

    def _run_ipopt(self, results, config, nl_info, basename, timer):
        # Get a copy of the environment to pass to the subprocess
        env = os.environ.copy()
        if nl_info.external_function_libraries:
            env['AMPLFUNC'] = amplfunc_merge(env, *nl_info.external_function_libraries)

        # Get the Ipopt executable and start building the command line
        exe = config.executable.path()
        if not exe:
            raise ApplicationError('ipopt executable not found')
        cmd = [exe, basename + '.nl', '-AMPL']

        # Process ipopt options (splitting them between command line
        # options and those that must be passed through the opt file)
        options = config.solver_options.value()
        # Map standard Pyomo solver options to Ipopt options: standard
        # options override ipopt-specific options.
        if config.threads and config.threads != 1:
            logger.log(
                logging.WARNING,
                msg=f"The `threads={config.threads}` option was specified, "
                f"but this is not used by {self.__class__.__name__}.",
            )
        if config.time_limit is not None:
            options['max_cpu_time'] = config.time_limit
        cmd.extend(self._process_options(basename + '.opt', options))

        results.solver_version = self._get_version(exe)
        results.extra_info.add(
            'command_line', ConfigValue(cmd, visibility=ADVANCED_OPTION)
        )

        # This seems silly, but we have to give the subprocess slightly
        # longer to finish than ipopt, otherwise we may kill the
        # subprocess before ipopt has a chance to write the SOL file.
        # We will add 1% (with a min of 1 second and max of 100 seconds).
        timeout = config.time_limit
        if timeout is not None:
            timeout = timeout + min(max(1.0, 0.01 * timeout), 100.0)

        # Call ipopt - passing the files via the subprocess
        ostreams = [io.StringIO()] + config.tee
        timer.start('subprocess')
        try:
            with TeeStream(*ostreams) as t:
                process = subprocess.run(
                    cmd,
                    timeout=timeout,
                    env=env,
                    universal_newlines=True,
                    stdout=t.STDOUT,
                    stderr=t.STDERR,
                    check=False,
                )
        except OSError:
            err = sys.exc_info()[1]
            msg = 'Could not execute the command: %s\tError message: %s'
            raise ApplicationError(msg % (cmd, err))
        finally:
            timer.stop('subprocess')

        results.solver_log = ostreams[0].getvalue()
        results.extra_info.return_code = process.returncode
        if process.returncode:
            results.termination_condition = TerminationCondition.error

        # This is the data we need to parse to get the iterations
        # and time
        timer.start('parse_log')
        parsed_output_data = self._parse_ipopt_output(results.solver_log)
        results.extra_info.iteration_count = parsed_output_data.pop('iters', None)
        _timing = parsed_output_data.pop('cpu_seconds', None)
        if _timing:
            results.timing_info.update(_timing)
        # Save the iteration log, but mark it as an "advanced" result
        iter_log = parsed_output_data.pop('iteration_log', None)
        if iter_log is not None:
            results.extra_info.add(
                'iteration_log', ConfigList(iter_log, visibility=ADVANCED_OPTION)
            )
        results.extra_info.update(parsed_output_data)
        timer.stop('parse_log')

        timer.start('parse_sol')
        if os.path.isfile(basename + '.sol'):
            with open(basename + '.sol', 'r', encoding='utf-8') as sol_file:
                sol_data = parse_asl_sol_file(sol_file)
        else:
            sol_data = ASLSolFileData()
        results.solution_loader = IpoptSolutionLoader(
            sol_data=sol_data, nl_info=nl_info
        )
        timer.stop('parse_sol')

        # Initialize the solver message, solution loader solution
        # status and termination condition:
        asl_solve_code_to_solution_status(sol_data, results)

    def _parse_ipopt_output(self, output: str) -> Dict[str, Any]:
        parsed_data = {}

        # Stop parsing if there is nothing to parse
        if not output:
            logger.log(
                logging.WARNING,
                "Returned output from ipopt was empty. Cannot parse for additional data.",
            )
            return parsed_data

        # Extract number of iterations
        iter_match = re.search(r'Number of Iterations.*:\s+(\d+)', output)
        if iter_match:
            parsed_data['iters'] = int(iter_match.group(1))
        # Gather all the iteration data
        iter_table = re.findall(r'^(?:\s*\d+.*?)$', output, re.MULTILINE)
        if iter_table:
            columns = [
                ("iter", int),
                ("objective", float),
                ("inf_pr", float),
                ("inf_du", float),
                ("lg_mu", float),
                ("d_norm", float),
                ("lg_rg", float),
                ("alpha_du", float),
                ("alpha_pr", float),
                ("ls", int),
            ]
            iterations = []
            n_expected_columns = len(columns)
            iter_idx = columns.index(('iter', int))
            alpha_pr_idx = columns.index(('alpha_pr', float))

            for line in iter_table:
                tokens = line.strip().split()
                # IPOPT sometimes mashes the first two column values together
                # (e.g., "2r-4.93e-03"). We need to split them.
                if '-' in tokens[iter_idx]:
                    # This happens rarely, so we are OK with this
                    # portion of the parser being a little less
                    # efficient (e.g., reallocating the tokens list, and
                    # performing index math)
                    tkn = tokens[iter_idx]
                    idx = tkn.index('-')
                    tokens[iter_idx : iter_idx + 1] = tkn[:idx], tkn[idx:]

                # Extract restoration flag from 'iter'
                restoration = tokens[iter_idx].endswith("r")
                if restoration:
                    tokens[iter_idx] = tokens[iter_idx][:-1]

                # Separate alpha_pr into numeric part and optional tag (f, D, R, etc.)
                step_acceptance = tokens[alpha_pr_idx][-1]
                if step_acceptance in _ALPHA_PR_CHARS:
                    tokens[alpha_pr_idx] = tokens[alpha_pr_idx][:-1]
                else:
                    step_acceptance = None

                try:
                    iter_data = {
                        key: None if t == '-' else cast(t)
                        for (key, cast), t in zip(columns, tokens)
                    }
                except (ValueError, TypeError):
                    logger.error(
                        "Error parsing Ipopt log entry:\n"
                        f"\t{sys.exc_info()[1]}\n\t{line}"
                    )
                    # Fall-back on a simpler (but slower) parse: extract
                    # the fields, and cast to float what we can.  The
                    # point here is the parser should never fail with an
                    # exception (even if it fails to parse some of the
                    # log)
                    iter_data = {}
                    for (key, cast), t in zip(columns, tokens):
                        if t == '-':
                            t = None
                        else:
                            try:
                                t = cast(t)
                            except:
                                pass
                        iter_data[key] = t

                iter_data["restoration"] = restoration
                iter_data["step_acceptance"] = step_acceptance

                # Capture optional IPOPT diagnostic tags if present
                if len(tokens) > n_expected_columns:
                    iter_data['diagnostic_tags'] = " ".join(tokens[n_expected_columns:])

                iterations.append(iter_data)

            parsed_data['iteration_log'] = iterations

            if len(iterations) != parsed_data.get('iters', 0) + 1:
                n_iter = parsed_data.get('iters', 0)
                logger.warning(
                    f"Total number of iteration records parsed {len(iterations)} does "
                    f"not match the number of iterations ({n_iter}) plus one."
                )

        # Extract scaled and unscaled table
        scaled_unscaled_match = re.search(
            r'''
            Objective\.*:\s*([-+eE0-9.]+)\s+([-+eE0-9.]+)\s*
            Dual\ infeasibility\.*:\s*([-+eE0-9.]+)\s+([-+eE0-9.]+)\s*
            Constraint\ violation\.*:\s*([-+eE0-9.]+)\s+([-+eE0-9.]+)\s*
            (?:Variable\ bound\ violation:\s*([-+eE0-9.]+)\s+([-+eE0-9.]+)\s*)?
            Complementarity\.*:\s*([-+eE0-9.]+)\s+([-+eE0-9.]+)\s*
            Overall\ NLP\ error\.*:\s*([-+eE0-9.]+)\s+([-+eE0-9.]+)
            ''',
            output,
            re.DOTALL | re.VERBOSE,
        )

        if scaled_unscaled_match:
            groups = scaled_unscaled_match.groups()
            all_fields = [
                "incumbent_objective",
                "dual_infeasibility",
                "constraint_violation",
                "variable_bound_violation",  # optional
                "complementarity_error",
                "overall_nlp_error",
            ]
            # Filter out None values and create final fields and values.
            # Nones occur in old-style IPOPT output (<= 3.13)
            zipped = [
                (field, scaled, unscaled)
                for field, scaled, unscaled in zip(
                    all_fields, groups[0::2], groups[1::2]
                )
                if scaled is not None and unscaled is not None
            ]
            scaled = {k: float(s) for k, s, _ in zipped}
            unscaled = {k: float(u) for k, _, u in zipped}
            parsed_data.update(unscaled)
            parsed_data['final_scaled_results'] = scaled

        # Newer versions of IPOPT no longer separate timing into
        # two different values. This is so we have compatibility with
        # both new and old versions
        parsed_data['cpu_seconds'] = {
            k.strip(): float(v)
            for k, v in re.findall(
                r'Total(?: CPU)? sec(?:ond)?s in ([^=]+)=\s*([0-9.]+)', output
            )
        }

        return parsed_data


class LegacyIpoptSolver(LegacySolverWrapper, Ipopt):
    def _process_options(
        self, option_fname: str, options: dict[str, str | int | float]
    ) -> list[str]:
        # The old Ipopt solver would map solver_options starting with
        # "OF_" to the options file.  That is no longer needed, so we
        # will strip off any "OF_" that we find
        for opt in list(options):
            if opt.startswith('OF_'):
                options[opt[3:]] = options.pop(opt)
        return super()._process_options(option_fname, options)
