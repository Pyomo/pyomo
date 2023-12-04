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
from typing import Mapping

from pyomo.common import Executable
from pyomo.common.config import ConfigValue, NonNegativeInt
from pyomo.common.errors import PyomoException
from pyomo.common.tempfiles import TempfileManager
from pyomo.core.base.label import NumericLabeler
from pyomo.repn.plugins.nl_writer import NLWriter, NLWriterInfo
from pyomo.solver.base import SolverBase, SymbolMap
from pyomo.solver.config import SolverConfig
from pyomo.solver.factory import SolverFactory
from pyomo.solver.results import (
    Results,
    TerminationCondition,
    SolutionStatus,
    parse_sol_file,
)
from pyomo.solver.solution import SolutionLoaderBase, SolutionLoader
from pyomo.common.tee import TeeStream
from pyomo.common.log import LogStream
from pyomo.core.expr.visitor import replace_expressions
from pyomo.core.expr.numvalue import value
from pyomo.core.base.suffix import Suffix

import logging

logger = logging.getLogger(__name__)


class SolverError(PyomoException):
    """
    General exception to catch solver system errors
    """

    pass


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
        self.save_solver_io: bool = self.declare(
            'save_solver_io', ConfigValue(domain=bool, default=False)
        )
        self.temp_dir: str = self.declare(
            'temp_dir', ConfigValue(domain=str, default=None)
        )
        self.solver_output_logger = self.declare(
            'solver_output_logger', ConfigValue(default=logger)
        )
        self.log_level = self.declare(
            'log_level', ConfigValue(domain=NonNegativeInt, default=logging.INFO)
        )


class ipoptSolutionLoader(SolutionLoaderBase):
    pass


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
        self._config = self.CONFIG(kwds)
        self._writer = NLWriter()
        self.ipopt_options = self._config.solver_options

    def available(self):
        if self.config.executable.path() is None:
            return self.Availability.NotFound
        return self.Availability.FullLicense

    def version(self):
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
        return version

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, val):
        self._config = val

    @property
    def symbol_map(self):
        return self._symbol_map

    def _write_options_file(self, ostream: io.TextIOBase, options: Mapping):
        f = ostream
        for k, val in options.items():
            if k not in ipopt_command_line_options:
                f.write(str(k) + ' ' + str(val) + '\n')

    def _create_command_line(self, basename: str, config: ipoptConfig):
        cmd = [
            str(config.executable),
            basename + '.nl',
            '-AMPL',
            'option_file_name=' + basename + '.opt',
        ]
        if 'option_file_name' in config.solver_options:
            raise ValueError(
                'Use ipopt.config.temp_dir to specify the name of the options file. '
                'Do not use ipopt.config.solver_options["option_file_name"].'
            )
        self.ipopt_options = dict(config.solver_options)
        if config.time_limit is not None and 'max_cpu_time' not in self.ipopt_options:
            self.ipopt_options['max_cpu_time'] = config.time_limit
        for k, v in self.ipopt_options.items():
            cmd.append(str(k) + '=' + str(v))

        return cmd

    def solve(self, model, **kwds):
        # Begin time tracking
        start_timestamp = datetime.datetime.now(datetime.timezone.utc)
        # Check if solver is available
        avail = self.available()
        if not avail:
            raise SolverError(f'Solver {self.__class__} is not available ({avail}).')
        # Update configuration options, based on keywords passed to solve
        config: ipoptConfig = self.config(kwds.pop('options', {}))
        config.set_value(kwds)
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
                nl_info = self._writer.write(
                    model,
                    nl_file,
                    row_file,
                    col_file,
                    symbolic_solver_labels=config.symbolic_solver_labels,
                )
            # Get a copy of the environment to pass to the subprocess
            env = os.environ.copy()
            if nl_info.external_function_libraries:
                if env.get('AMPLFUNC'):
                    nl_info.external_function_libraries.append(env.get('AMPLFUNC'))
                env['AMPLFUNC'] = "\n".join(nl_info.external_function_libraries)
            symbol_map = self._symbol_map = SymbolMap()
            labeler = NumericLabeler('component')
            for v in nl_info.variables:
                symbol_map.getSymbol(v, labeler)
            for c in nl_info.constraints:
                symbol_map.getSymbol(c, labeler)
            with open(basename + '.opt', 'w') as opt_file:
                self._write_options_file(
                    ostream=opt_file, options=config.solver_options
                )
            # Call ipopt - passing the files via the subprocess
            cmd = self._create_command_line(basename=basename, config=config)

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
            else:
                ostreams.append(
                    LogStream(
                        level=config.log_level, logger=config.solver_output_logger
                    )
                )
            with TeeStream(*ostreams) as t:
                process = subprocess.run(
                    cmd,
                    timeout=timeout,
                    env=env,
                    universal_newlines=True,
                    stdout=t.STDOUT,
                    stderr=t.STDERR,
                )
                # This is the stuff we need to parse to get the iterations
                # and time
                iters, solver_time = self._parse_ipopt_output(ostreams[0])

            if process.returncode != 0:
                results = Results()
                results.termination_condition = TerminationCondition.error
                results.solution_loader = SolutionLoader(None, None, None, None)
            else:
                with open(basename + '.sol', 'r') as sol_file:
                    results = self._parse_solution(sol_file, nl_info)
                results.iteration_count = iters
                results.timing_info.solver_wall_time = solver_time

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

        if results.solution_status in {SolutionStatus.feasible, SolutionStatus.optimal}:
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
        time = None
        # parse the output stream to get the iteration count and solver time
        for line in stream.getvalue().splitlines():
            if line.startswith("Number of Iterations....:"):
                tokens = line.split()
                iters = int(tokens[3])
            elif line.startswith(
                "Total CPU secs in IPOPT (w/o function evaluations)   ="
            ):
                tokens = line.split()
                time = float(tokens[9])
            elif line.startswith(
                "Total CPU secs in NLP function evaluations           ="
            ):
                tokens = line.split()
                time += float(tokens[8])

        return iters, time

    def _parse_solution(self, instream: io.TextIOBase, nl_info: NLWriterInfo):
        suffixes_to_read = ['dual', 'ipopt_zL_out', 'ipopt_zU_out']
        res, sol_data = parse_sol_file(
            sol_file=instream, nl_info=nl_info, suffixes_to_read=suffixes_to_read
        )

        if res.solution_status == SolutionStatus.noSolution:
            res.solution_loader = SolutionLoader(None, None, None, None)
        else:
            rc = dict()
            for v in nl_info.variables:
                v_id = id(v)
                rc[v_id] = (v, 0)
                if v_id in sol_data.var_suffixes['ipopt_zL_out']:
                    zl = sol_data.var_suffixes['ipopt_zL_out'][v_id][1]
                    if abs(zl) > abs(rc[v_id][1]):
                        rc[v_id] = (v, zl)
                if v_id in sol_data.var_suffixes['ipopt_zU_out']:
                    zu = sol_data.var_suffixes['ipopt_zU_out'][v_id][1]
                    if abs(zu) > abs(rc[v_id][1]):
                        rc[v_id] = (v, zu)

            res.solution_loader = SolutionLoader(
                primals=sol_data.primals,
                duals=sol_data.duals,
                slacks=None,
                reduced_costs=rc,
            )

        return res
