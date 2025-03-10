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

from pyomo.common.tempfiles import TempfileManager
from pyomo.common.fileutils import Executable
from pyomo.contrib.appsi.base import (
    PersistentSolver,
    Results,
    TerminationCondition,
    SolverConfig,
    PersistentSolutionLoader,
)
from pyomo.contrib.appsi.writers import NLWriter
from pyomo.common.log import LogStream
import logging
import subprocess
from pyomo.core.kernel.objective import minimize
import math
from pyomo.common.collections import ComponentMap
from pyomo.core.expr.numvalue import value
from pyomo.core.expr.visitor import replace_expressions
from typing import Optional, Sequence, NoReturn, List, Mapping
from pyomo.core.base.var import VarData
from pyomo.core.base.constraint import ConstraintData
from pyomo.core.base.block import BlockData
from pyomo.core.base.param import ParamData
from pyomo.core.base.objective import ObjectiveData
from pyomo.common.timing import HierarchicalTimer
from pyomo.common.tee import TeeStream
import sys
from typing import Dict
from pyomo.common.config import ConfigValue, NonNegativeInt
from pyomo.common.errors import PyomoException
import os
from pyomo.contrib.appsi.cmodel import cmodel_available
from pyomo.core.staleflag import StaleFlagManager


logger = logging.getLogger(__name__)


class IpoptConfig(SolverConfig):
    def __init__(
        self,
        description=None,
        doc=None,
        implicit=False,
        implicit_domain=None,
        visibility=0,
    ):
        super(IpoptConfig, self).__init__(
            description=description,
            doc=doc,
            implicit=implicit,
            implicit_domain=implicit_domain,
            visibility=visibility,
        )

        self.declare('executable', ConfigValue())
        self.declare('filename', ConfigValue(domain=str))
        self.declare('keepfiles', ConfigValue(domain=bool))
        self.declare('solver_output_logger', ConfigValue())
        self.declare('log_level', ConfigValue(domain=NonNegativeInt))

        self.executable = Executable('ipopt')
        self.filename = None
        self.keepfiles = False
        self.solver_output_logger = logger
        self.log_level = logging.INFO


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


class Ipopt(PersistentSolver):
    def __init__(self, only_child_vars=False):
        self._config = IpoptConfig()
        self._solver_options = dict()
        self._writer = NLWriter(only_child_vars=only_child_vars)
        self._filename = None
        self._dual_sol = dict()
        self._primal_sol = ComponentMap()
        self._reduced_costs = ComponentMap()
        self._last_results_object: Optional[Results] = None
        self._version_timeout = 2

    def available(self):
        if self.config.executable.path() is None:
            return self.Availability.NotFound
        elif not cmodel_available:
            return self.Availability.NeedsCompiledExtension
        return self.Availability.FullLicense

    def version(self):
        results = subprocess.run(
            [str(self.config.executable), '--version'],
            timeout=self._version_timeout,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        version = results.stdout.splitlines()[0]
        version = version.split(' ')[1]
        version = version.strip()
        version = tuple(int(i) for i in version.split('.'))
        return version

    def nl_filename(self):
        if self._filename is None:
            return None
        else:
            return self._filename + '.nl'

    def sol_filename(self):
        if self._filename is None:
            return None
        else:
            return self._filename + '.sol'

    def options_filename(self):
        if self._filename is None:
            return None
        else:
            return self._filename + '.opt'

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, val):
        self._config = val

    @property
    def ipopt_options(self):
        """
        A dictionary mapping solver options to values for those options. These are solver specific.

        Returns
        -------
            dict
                A dictionary mapping solver options to values for those options
        """
        return self._solver_options

    @ipopt_options.setter
    def ipopt_options(self, val: Dict):
        self._solver_options = val

    @property
    def update_config(self):
        return self._writer.update_config

    @property
    def writer(self):
        return self._writer

    @property
    def symbol_map(self):
        return self._writer.symbol_map

    def set_instance(self, model):
        self._writer.config.symbolic_solver_labels = self.config.symbolic_solver_labels
        self._writer.set_instance(model)

    def add_variables(self, variables: List[VarData]):
        self._writer.add_variables(variables)

    def add_params(self, params: List[ParamData]):
        self._writer.add_params(params)

    def add_constraints(self, cons: List[ConstraintData]):
        self._writer.add_constraints(cons)

    def add_block(self, block: BlockData):
        self._writer.add_block(block)

    def remove_variables(self, variables: List[VarData]):
        self._writer.remove_variables(variables)

    def remove_params(self, params: List[ParamData]):
        self._writer.remove_params(params)

    def remove_constraints(self, cons: List[ConstraintData]):
        self._writer.remove_constraints(cons)

    def remove_block(self, block: BlockData):
        self._writer.remove_block(block)

    def set_objective(self, obj: ObjectiveData):
        self._writer.set_objective(obj)

    def update_variables(self, variables: List[VarData]):
        self._writer.update_variables(variables)

    def update_params(self):
        self._writer.update_params()

    def _write_options_file(self):
        f = open(self._filename + '.opt', 'w')
        for k, val in self.ipopt_options.items():
            if k not in ipopt_command_line_options:
                f.write(str(k) + ' ' + str(val) + '\n')
        f.close()

    def solve(self, model, timer: HierarchicalTimer = None):
        StaleFlagManager.mark_all_as_stale()
        avail = self.available()
        if not avail:
            raise PyomoException(f'Solver {self.__class__} is not available ({avail}).')
        if self._last_results_object is not None:
            self._last_results_object.solution_loader.invalidate()
        if timer is None:
            timer = HierarchicalTimer()
        try:
            TempfileManager.push()
            if self.config.filename is None:
                nl_filename = TempfileManager.create_tempfile(suffix='.nl')
                self._filename = nl_filename.split('.')[0]
            else:
                self._filename = self.config.filename
                TempfileManager.add_tempfile(self._filename + '.nl', exists=False)
            TempfileManager.add_tempfile(self._filename + '.sol', exists=False)
            TempfileManager.add_tempfile(self._filename + '.opt', exists=False)
            self._write_options_file()
            timer.start('write nl file')
            self._writer.write(model, self._filename + '.nl', timer=timer)
            timer.stop('write nl file')
            res = self._apply_solver(timer)
            self._last_results_object = res
            if self.config.report_timing:
                logger.info('\n' + str(timer))
            return res
        finally:
            # finally, clean any temporary files registered with the
            # temp file manager, created/populated *directly* by this
            # plugin.
            TempfileManager.pop(remove=not self.config.keepfiles)
            if not self.config.keepfiles:
                self._filename = None

    def _parse_sol(self):
        solve_vars = self._writer.get_ordered_vars()
        solve_cons = self._writer.get_ordered_cons()
        results = Results()

        f = open(self._filename + '.sol', 'r')
        all_lines = list(f.readlines())
        f.close()

        termination_line = all_lines[1]
        if 'Optimal Solution Found' in termination_line:
            results.termination_condition = TerminationCondition.optimal
        elif 'Problem may be infeasible' in termination_line:
            results.termination_condition = TerminationCondition.infeasible
        elif 'problem might be unbounded' in termination_line:
            results.termination_condition = TerminationCondition.unbounded
        elif 'Maximum Number of Iterations Exceeded' in termination_line:
            results.termination_condition = TerminationCondition.maxIterations
        elif 'Maximum CPU Time Exceeded' in termination_line:
            results.termination_condition = TerminationCondition.maxTimeLimit
        else:
            results.termination_condition = TerminationCondition.unknown

        n_cons = len(solve_cons)
        n_vars = len(solve_vars)
        dual_lines = all_lines[12 : 12 + n_cons]
        primal_lines = all_lines[12 + n_cons : 12 + n_cons + n_vars]

        rc_upper_info_line = all_lines[12 + n_cons + n_vars + 1]
        assert rc_upper_info_line.startswith('suffix')
        n_rc_upper = int(rc_upper_info_line.split()[2])
        assert 'ipopt_zU_out' in all_lines[12 + n_cons + n_vars + 2]
        upper_rc_lines = all_lines[
            12 + n_cons + n_vars + 3 : 12 + n_cons + n_vars + 3 + n_rc_upper
        ]

        rc_lower_info_line = all_lines[12 + n_cons + n_vars + 3 + n_rc_upper]
        assert rc_lower_info_line.startswith('suffix')
        n_rc_lower = int(rc_lower_info_line.split()[2])
        assert 'ipopt_zL_out' in all_lines[12 + n_cons + n_vars + 3 + n_rc_upper + 1]
        lower_rc_lines = all_lines[
            12
            + n_cons
            + n_vars
            + 3
            + n_rc_upper
            + 2 : 12
            + n_cons
            + n_vars
            + 3
            + n_rc_upper
            + 2
            + n_rc_lower
        ]

        self._dual_sol = dict()
        self._primal_sol = ComponentMap()
        self._reduced_costs = ComponentMap()

        for ndx, dual in enumerate(dual_lines):
            dual = float(dual)
            con = solve_cons[ndx]
            self._dual_sol[con] = dual

        for ndx, primal in enumerate(primal_lines):
            primal = float(primal)
            var = solve_vars[ndx]
            self._primal_sol[var] = primal

        for rcu_line in upper_rc_lines:
            split_line = rcu_line.split()
            var_ndx = int(split_line[0])
            rcu = float(split_line[1])
            var = solve_vars[var_ndx]
            self._reduced_costs[var] = rcu

        for rcl_line in lower_rc_lines:
            split_line = rcl_line.split()
            var_ndx = int(split_line[0])
            rcl = float(split_line[1])
            var = solve_vars[var_ndx]
            if var in self._reduced_costs:
                if abs(rcl) > abs(self._reduced_costs[var]):
                    self._reduced_costs[var] = rcl
            else:
                self._reduced_costs[var] = rcl

        for var in solve_vars:
            if var not in self._reduced_costs:
                self._reduced_costs[var] = 0

        if (
            results.termination_condition == TerminationCondition.optimal
            and self.config.load_solution
        ):
            for v, val in self._primal_sol.items():
                v.set_value(val, skip_validation=True)
            if self._writer.get_active_objective() is None:
                results.best_feasible_objective = None
            else:
                results.best_feasible_objective = value(
                    self._writer.get_active_objective().expr
                )
        elif results.termination_condition == TerminationCondition.optimal:
            if self._writer.get_active_objective() is None:
                results.best_feasible_objective = None
            else:
                obj_expr_evaluated = replace_expressions(
                    self._writer.get_active_objective().expr,
                    substitution_map={
                        id(v): val for v, val in self._primal_sol.items()
                    },
                    descend_into_named_expressions=True,
                    remove_named_expressions=True,
                )
                results.best_feasible_objective = value(obj_expr_evaluated)
        elif self.config.load_solution:
            raise RuntimeError(
                'A feasible solution was not found, so no solution can be loaded. '
                'If using the appsi.solvers.Ipopt interface, you can '
                'set opt.config.load_solution=False. If using the environ.SolverFactory '
                'interface, you can set opt.solve(model, load_solutions = False). '
                'Then you can check results.termination_condition and '
                'results.best_feasible_objective before loading a solution.'
            )

        return results

    def _apply_solver(self, timer: HierarchicalTimer):
        config = self.config

        if config.time_limit is not None:
            timeout = config.time_limit + min(max(1.0, 0.01 * config.time_limit), 100)
        else:
            timeout = None

        ostreams = [
            LogStream(
                level=self.config.log_level, logger=self.config.solver_output_logger
            )
        ]
        if self.config.stream_solver:
            ostreams.append(sys.stdout)

        cmd = [
            str(config.executable),
            self._filename + '.nl',
            '-AMPL',
            'option_file_name=' + self._filename + '.opt',
        ]
        if 'option_file_name' in self.ipopt_options:
            raise ValueError(
                'Use Ipopt.config.filename to specify the name of the options file. '
                'Do not use Ipopt.ipopt_options["option_file_name"].'
            )
        ipopt_options = dict(self.ipopt_options)
        if config.time_limit is not None:
            ipopt_options['max_cpu_time'] = config.time_limit
        for k, v in ipopt_options.items():
            cmd.append(str(k) + '=' + str(v))

        env = os.environ.copy()
        if 'PYOMO_AMPLFUNC' in env:
            env['AMPLFUNC'] = "\n".join(
                filter(
                    None, (env.get('AMPLFUNC', None), env.get('PYOMO_AMPLFUNC', None))
                )
            )

        with TeeStream(*ostreams) as t:
            timer.start('subprocess')
            cp = subprocess.run(
                cmd,
                timeout=timeout,
                stdout=t.STDOUT,
                stderr=t.STDERR,
                env=env,
                universal_newlines=True,
            )
            timer.stop('subprocess')

        if cp.returncode != 0:
            if self.config.load_solution:
                raise RuntimeError(
                    'A feasible solution was not found, so no solution can be loaded.'
                    'Please set opt.config.load_solution=False and check '
                    'results.termination_condition and '
                    'results.best_feasible_objective before loading a solution.'
                )
            results = Results()
            results.termination_condition = TerminationCondition.error
            results.best_feasible_objective = None
        else:
            timer.start('parse solution')
            results = self._parse_sol()
            timer.stop('parse solution')

        if self._writer.get_active_objective() is None:
            results.best_objective_bound = None
        else:
            if self._writer.get_active_objective().sense == minimize:
                results.best_objective_bound = -math.inf
            else:
                results.best_objective_bound = math.inf

        results.solution_loader = PersistentSolutionLoader(solver=self)

        return results

    def get_primals(
        self, vars_to_load: Optional[Sequence[VarData]] = None
    ) -> Mapping[VarData, float]:
        if (
            self._last_results_object is None
            or self._last_results_object.best_feasible_objective is None
        ):
            raise RuntimeError(
                'Solver does not currently have a valid solution. Please '
                'check the termination condition.'
            )

        res = ComponentMap()
        if vars_to_load is None:
            for v, val in self._primal_sol.items():
                res[v] = val
        else:
            for v in vars_to_load:
                res[v] = self._primal_sol[v]
        return res

    def get_duals(self, cons_to_load: Optional[Sequence[ConstraintData]] = None):
        if (
            self._last_results_object is None
            or self._last_results_object.termination_condition
            != TerminationCondition.optimal
        ):
            raise RuntimeError(
                'Solver does not currently have valid duals. Please '
                'check the termination condition.'
            )

        if cons_to_load is None:
            return {k: v for k, v in self._dual_sol.items()}
        else:
            return {c: self._dual_sol[c] for c in cons_to_load}

    def get_reduced_costs(
        self, vars_to_load: Optional[Sequence[VarData]] = None
    ) -> Mapping[VarData, float]:
        if (
            self._last_results_object is None
            or self._last_results_object.termination_condition
            != TerminationCondition.optimal
        ):
            raise RuntimeError(
                'Solver does not currently have valid reduced costs. Please '
                'check the termination condition.'
            )

        if vars_to_load is None:
            return ComponentMap((k, v) for k, v in self._reduced_costs.items())
        else:
            return ComponentMap((v, self._reduced_costs[v]) for v in vars_to_load)

    def has_linear_solver(self, linear_solver):
        import pyomo.core as AML
        from pyomo.common.tee import capture_output

        m = AML.ConcreteModel()
        m.x = AML.Var()
        m.o = AML.Objective(expr=(m.x - 2) ** 2)
        with capture_output() as OUT:
            solver = self.__class__()
            solver.config.stream_solver = True
            solver.config.load_solution = False
            solver.ipopt_options['linear_solver'] = linear_solver
            try:
                solver.solve(m)
            except FileNotFoundError:
                # The APPSI interface always tries to open the SOL file,
                # and will generate a FileNotFoundError if ipopt didn't
                # generate one
                return False
        return 'running with linear solver' in OUT.getvalue()
