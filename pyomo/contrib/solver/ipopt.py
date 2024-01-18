#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import os
import subprocess
import datetime
import io
import sys
from typing import Mapping, Optional, Sequence

from pyomo.common import Executable
from pyomo.common.config import ConfigValue, NonNegativeInt, NonNegativeFloat
from pyomo.common.errors import PyomoException
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.timing import HierarchicalTimer
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.staleflag import StaleFlagManager
from pyomo.repn.plugins.nl_writer import NLWriter, NLWriterInfo
from pyomo.contrib.solver.base import SolverBase
from pyomo.contrib.solver.config import SolverConfig
from pyomo.contrib.solver.factory import SolverFactory
from pyomo.contrib.solver.results import Results, TerminationCondition, SolutionStatus
from .sol_reader import parse_sol_file
from pyomo.contrib.solver.solution import SolSolutionLoader, SolutionLoader
from pyomo.common.tee import TeeStream
from pyomo.common.log import LogStream
from pyomo.core.expr.visitor import replace_expressions
from pyomo.core.expr.numvalue import value
from pyomo.core.base.suffix import Suffix
from pyomo.common.collections import ComponentMap

import logging

logger = logging.getLogger(__name__)


class ipoptSolverError(PyomoException):
    """
    General exception to catch solver system errors
    """


class ipoptConfig(SolverConfig):
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

        self.executable = self.declare(
            'executable', ConfigValue(default=Executable('ipopt'))
        )
        # TODO: Add in a deprecation here for keepfiles
        # M.B.: Is the above TODO still relevant?
        self.temp_dir: str = self.declare(
            'temp_dir', ConfigValue(domain=str, default=None)
        )
        self.writer_config = self.declare(
            'writer_config', ConfigValue(default=NLWriter.CONFIG())
        )


class ipoptResults(Results):
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
        self.timing_info.no_function_solve_time: Optional[
            float
        ] = self.timing_info.declare(
            'no_function_solve_time', ConfigValue(domain=NonNegativeFloat)
        )
        self.timing_info.function_solve_time: Optional[
            float
        ] = self.timing_info.declare(
            'function_solve_time', ConfigValue(domain=NonNegativeFloat)
        )


class ipoptSolutionLoader(SolSolutionLoader):
    def get_reduced_costs(self, vars_to_load: Sequence[_GeneralVarData] | None = None) -> Mapping[_GeneralVarData, float]:
        if self._nl_info.scaling is None:
            scale_list = [1] * len(self._nl_info.variables)
        else:
            scale_list = self._nl_info.scaling.variables
        sol_data = self._sol_data
        nl_info = self._nl_info
        zl_map = sol_data.var_suffixes['ipopt_zL_out']
        zu_map = sol_data.var_suffixes['ipopt_zU_out']
        rc = dict()
        for ndx, v in enumerate(nl_info.variables):
            scale = scale_list[ndx]
            v_id = id(v)
            rc[v_id] = (v, 0)
            if ndx in zl_map:
                zl = zl_map[ndx] * scale
                if abs(zl) > abs(rc[v_id][1]):
                    rc[v_id] = (v, zl)
            if ndx in zu_map:
                zu = zu_map[ndx] * scale
                if abs(zu) > abs(rc[v_id][1]):
                    rc[v_id] = (v, zu)

        if vars_to_load is None:
            res = ComponentMap(rc.values())
            for v, _ in nl_info.eliminated_vars:
                res[v] = 0
        else:
            res = ComponentMap()
            for v in vars_to_load:
                if id(v) in rc:
                    res[v] = rc[id(v)][1]
                else:
                    # eliminated vars
                    res[v] = 0
        return res


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


@SolverFactory.register('ipopt_v2', doc='The ipopt NLP solver (new interface)')
class ipopt(SolverBase):
    CONFIG = ipoptConfig()

    def __init__(self, **kwds):
        super().__init__(**kwds)
        self._writer = NLWriter()
        self._available_cache = None
        self._version_cache = None

    def available(self):
        if self._available_cache is None:
            if self.config.executable.path() is None:
                self._available_cache = self.Availability.NotFound
            else:
                self._available_cache = self.Availability.FullLicense
        return self._available_cache

    def version(self):
        if self._version_cache is None:
            results = subprocess.run(
                [str(self.config.executable), '--version'],
                timeout=1,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
            )
            version = results.stdout.splitlines()[0]
            version = version.split(' ')[1].strip()
            version = tuple(int(i) for i in version.split('.'))
            self._version_cache = version
        return self._version_cache

    def _write_options_file(self, filename: str, options: Mapping):
        # First we need to determine if we even need to create a file.
        # If options is empty, then we return False
        opt_file_exists = False
        if not options:
            return False
        # If it has options in it, parse them and write them to a file.
        # If they are command line options, ignore them; they will be
        # parsed during _create_command_line
        with open(filename + '.opt', 'w') as opt_file:
            for k, val in options.items():
                if k not in ipopt_command_line_options:
                    opt_file_exists = True
                    opt_file.write(str(k) + ' ' + str(val) + '\n')
        return opt_file_exists

    def _create_command_line(self, basename: str, config: ipoptConfig, opt_file: bool):
        cmd = [str(config.executable), basename + '.nl', '-AMPL']
        if opt_file:
            cmd.append('option_file_name=' + basename + '.opt')
        if 'option_file_name' in config.solver_options:
            raise ValueError(
                'Pyomo generates the ipopt options file as part of the solve method. '
                'Add all options to ipopt.config.solver_options instead.'
            )
        if (
            config.time_limit is not None
            and 'max_cpu_time' not in config.solver_options
        ):
            config.solver_options['max_cpu_time'] = config.time_limit
        for k, val in config.solver_options.items():
            if k in ipopt_command_line_options:
                cmd.append(str(k) + '=' + str(val))
        return cmd

    def solve(self, model, **kwds):
        # Begin time tracking
        start_timestamp = datetime.datetime.now(datetime.timezone.utc)
        # Check if solver is available
        avail = self.available()
        if not avail:
            raise ipoptSolverError(
                f'Solver {self.__class__} is not available ({avail}).'
            )
        # Update configuration options, based on keywords passed to solve
        config: ipoptConfig = self.config(value=kwds)
        if config.threads:
            logger.log(
                logging.WARNING,
                msg=f"The `threads` option was specified, but this is not used by {self.__class__}.",
            )
        if config.timer is None:
            timer = HierarchicalTimer()
        else:
            timer = config.timer
        StaleFlagManager.mark_all_as_stale()
        results = ipoptResults()
        with TempfileManager.new_context() as tempfile:
            if config.temp_dir is None:
                dname = tempfile.mkdtemp()
            else:
                dname = config.temp_dir
            if not os.path.exists(dname):
                os.mkdir(dname)
            basename = os.path.join(dname, model.name)
            if os.path.exists(basename + '.nl'):
                raise RuntimeError(
                    f"NL file with the same name {basename + '.nl'} already exists!"
                )
            with open(basename + '.nl', 'w') as nl_file, open(
                basename + '.row', 'w'
            ) as row_file, open(basename + '.col', 'w') as col_file:
                timer.start('write_nl_file')
                self._writer.config.set_value(config.writer_config)
                nl_info = self._writer.write(
                    model,
                    nl_file,
                    row_file,
                    col_file,
                    symbolic_solver_labels=config.symbolic_solver_labels,
                )
                timer.stop('write_nl_file')
            # Get a copy of the environment to pass to the subprocess
            env = os.environ.copy()
            if nl_info.external_function_libraries:
                if env.get('AMPLFUNC'):
                    nl_info.external_function_libraries.append(env.get('AMPLFUNC'))
                env['AMPLFUNC'] = "\n".join(nl_info.external_function_libraries)
            # Write the opt_file, if there should be one; return a bool to say
            # whether or not we have one (so we can correctly build the command line)
            opt_file = self._write_options_file(
                filename=basename, options=config.solver_options
            )
            # Call ipopt - passing the files via the subprocess
            cmd = self._create_command_line(
                basename=basename, config=config, opt_file=opt_file
            )
            # this seems silly, but we have to give the subprocess slightly longer to finish than
            # ipopt
            if config.time_limit is not None:
                timeout = config.time_limit + min(
                    max(1.0, 0.01 * config.time_limit), 100
                )
            else:
                timeout = None

            ostreams = [io.StringIO()]
            if config.tee:
                ostreams.append(sys.stdout)
            if config.log_solver_output:
                ostreams.append(
                    LogStream(
                        level=logging.INFO, logger=logger
                    )
                )
            with TeeStream(*ostreams) as t:
                timer.start('subprocess')
                process = subprocess.run(
                    cmd,
                    timeout=timeout,
                    env=env,
                    universal_newlines=True,
                    stdout=t.STDOUT,
                    stderr=t.STDERR,
                )
                timer.stop('subprocess')
                # This is the stuff we need to parse to get the iterations
                # and time
                iters, ipopt_time_nofunc, ipopt_time_func = self._parse_ipopt_output(
                    ostreams[0]
                )

            if process.returncode != 0:
                results.termination_condition = TerminationCondition.error
                results.solution_loader = SolutionLoader(None, None, None, None)
            else:
                with open(basename + '.sol', 'r') as sol_file:
                    timer.start('parse_sol')
                    results = self._parse_solution(sol_file, nl_info, results)
                    timer.stop('parse_sol')
                results.iteration_count = iters
                results.timing_info.no_function_solve_time = ipopt_time_nofunc
                results.timing_info.function_solve_time = ipopt_time_func
        if (
            config.raise_exception_on_nonoptimal_result
            and results.solution_status != SolutionStatus.optimal
        ):
            raise RuntimeError(
                'Solver did not find the optimal solution. Set opt.config.raise_exception_on_nonoptimal_result = False to bypass this error.'
            )

        results.solver_name = 'ipopt'
        results.solver_version = self.version()
        if (
            config.load_solution
            and results.solution_status == SolutionStatus.noSolution
        ):
            raise RuntimeError(
                'A feasible solution was not found, so no solution can be loaded.'
                'Please set config.load_solution=False to bypass this error.'
            )

        if config.load_solution:
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
            if config.load_solution:
                results.incumbent_objective = value(nl_info.objectives[0])
            else:
                results.incumbent_objective = replace_expressions(
                    nl_info.objectives[0].expr,
                    substitution_map={
                        id(v): val
                        for v, val in results.solution_loader.get_primals().items()
                    },
                    descend_into_named_expressions=True,
                    remove_named_expressions=True,
                )

        results.solver_configuration = config
        results.solver_log = ostreams[0].getvalue()

        # Capture/record end-time / wall-time
        end_timestamp = datetime.datetime.now(datetime.timezone.utc)
        results.timing_info.start_timestamp = start_timestamp
        results.timing_info.wall_time = (
            end_timestamp - start_timestamp
        ).total_seconds()
        return results

    def _parse_ipopt_output(self, stream: io.StringIO):
        """
        Parse an IPOPT output file and return:

        * number of iterations
        * time in IPOPT

        """

        iters = None
        nofunc_time = None
        func_time = None
        # parse the output stream to get the iteration count and solver time
        for line in stream.getvalue().splitlines():
            if line.startswith("Number of Iterations....:"):
                tokens = line.split()
                iters = int(tokens[3])
            elif line.startswith(
                "Total CPU secs in IPOPT (w/o function evaluations)   ="
            ):
                tokens = line.split()
                nofunc_time = float(tokens[9])
            elif line.startswith(
                "Total CPU secs in NLP function evaluations           ="
            ):
                tokens = line.split()
                func_time = float(tokens[8])

        return iters, nofunc_time, func_time

    def _parse_solution(
        self, instream: io.TextIOBase, nl_info: NLWriterInfo, result: ipoptResults
    ):
        res, sol_data = parse_sol_file(
            sol_file=instream, nl_info=nl_info, result=result
        )

        if res.solution_status == SolutionStatus.noSolution:
            res.solution_loader = SolutionLoader(None, None, None, None)
        else:
            res.solution_loader = ipoptSolutionLoader(
                sol_data=sol_data,
                nl_info=nl_info,
            )

        return res