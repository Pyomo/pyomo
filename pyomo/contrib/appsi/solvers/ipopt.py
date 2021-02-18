from pyutilib.services import TempfileManager
from pyomo.common.fileutils import Executable
from pyomo.contrib.appsi.base import Solver, Results, TerminationCondition, SolverConfig
from pyomo.contrib.appsi.writers import NLWriter
from pyomo.contrib.appsi.utils import TeeThread
import logging
import subprocess
from pyomo.core.kernel.objective import minimize
import math
from pyomo.common.collections import ComponentMap
from pyomo.core.expr.numvalue import value
from pyomo.core.expr.visitor import replace_expressions
from typing import Optional, Sequence, NoReturn, List, Mapping
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.base.constraint import _GeneralConstraintData
from pyomo.core.base.block import _BlockData
from pyomo.core.base.param import _ParamData
from pyomo.core.base.objective import _GeneralObjectiveData
from pyomo.common.timing import HierarchicalTimer


logger = logging.getLogger(__name__)


class IpoptConfig(SolverConfig):
    def __init__(self):
        super(IpoptConfig, self).__init__()
        self.executable = Executable('ipopt')
        self.filename = None
        self.keepfiles = False


class Ipopt(Solver):
    def __init__(self):
        self._config = IpoptConfig()
        self._solver_options = dict()
        self._writer = NLWriter()
        self._filename = None
        self._dual_sol = dict()
        self._primal_sol = ComponentMap()
        self._reduced_costs = ComponentMap()

    def available(self):
        if self.config.executable.path() is None:
            return False
        return True

    def version(self):
        cp = subprocess.run([str(self.config.executable), '--version'],
                            capture_output=True, text=True)

    def nl_filename(self):
        if self._filename is None:
            return None
        else:
            return self._filename + '.nl'

    def row_filename(self):
        if self._filename is None:
            return None
        else:
            return self._filename + '.row'

    def col_filename(self):
        if self._filename is None:
            return None
        else:
            return self._filename + '.col'

    def log_filename(self):
        if self._filename is None:
            return None
        else:
            return self._filename + '.log'

    def sol_filename(self):
        if self._filename is None:
            return None
        else:
            return self._filename + '.sol'

    @property
    def config(self):
        return self._config

    @property
    def solver_options(self):
        return self._solver_options

    @property
    def update_config(self):
        return self._writer.update_config

    def set_instance(self, model):
        self._writer.set_instance(model)

    def add_variables(self, variables: List[_GeneralVarData]):
        self._writer.add_variables(variables)

    def add_params(self, params: List[_ParamData]):
        self._writer.add_params(params)

    def add_constraints(self, cons: List[_GeneralConstraintData]):
        self._writer.add_constraints(cons)

    def add_block(self, block: _BlockData):
        self._writer.add_block(block)

    def remove_variables(self, variables: List[_GeneralVarData]):
        self._writer.remove_variables(variables)

    def remove_params(self, params: List[_ParamData]):
        self._writer.remove_params(params)

    def remove_constraints(self, cons: List[_GeneralConstraintData]):
        self._writer.remove_constraints(cons)

    def remove_block(self, block: _BlockData):
        self._writer.remove_block(block)

    def set_objective(self, obj: _GeneralObjectiveData):
        self._writer.set_objective(obj)

    def update_variables(self, variables: List[_GeneralVarData]):
        self._writer.update_variables(variables)

    def update_params(self):
        self._writer.update_params()

    def _write_options_file(self):
        f = open('ipopt.opt', 'w')
        for k, val in self.solver_options.items():
            f.write(str(k) + ' ' + str(val) + '\n')
        f.close()

    def solve(self, model, timer: HierarchicalTimer = None):
        if not self.available():
            raise RuntimeError('Could not find Ipopt executable')
        if timer is None:
            timer = HierarchicalTimer()
        try:
            TempfileManager.push()
            if self.config.filename is None:
                self._filename = TempfileManager.create_tempfile()
            else:
                self._filename = self.config.filename
            TempfileManager.add_tempfile(self._filename + '.nl', exists=False)
            TempfileManager.add_tempfile(self._filename + '.row', exists=False)
            TempfileManager.add_tempfile(self._filename + '.col', exists=False)
            TempfileManager.add_tempfile(self._filename + '.sol', exists=False)
            TempfileManager.add_tempfile(self._filename + '.log', exists=False)
            TempfileManager.add_tempfile('ipopt.opt', exists=False)
            self._write_options_file()
            timer.start('write nl file')
            self._writer.write(model, self._filename+'.nl', timer=timer)
            timer.stop('write nl file')
            res = self._apply_solver(timer)
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
        dual_lines = all_lines[12:12+n_cons]
        primal_lines = all_lines[12+n_cons:12+n_cons+n_vars]

        rc_upper_info_line = all_lines[12+n_cons+n_vars+1]
        assert rc_upper_info_line.startswith('suffix')
        n_rc_upper = int(rc_upper_info_line.split()[2])
        assert 'ipopt_zU_out' in all_lines[12+n_cons+n_vars+2]
        upper_rc_lines = all_lines[12+n_cons+n_vars+3:12+n_cons+n_vars+3+n_rc_upper]

        rc_lower_info_line = all_lines[12+n_cons+n_vars+3+n_rc_upper]
        assert rc_lower_info_line.startswith('suffix')
        n_rc_lower = int(rc_lower_info_line.split()[2])
        assert 'ipopt_zL_out' in all_lines[12+n_cons+n_vars+3+n_rc_upper+1]
        lower_rc_lines = all_lines[12+n_cons+n_vars+3+n_rc_upper+2:12+n_cons+n_vars+3+n_rc_upper+2+n_rc_lower]

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

        if results.termination_condition == TerminationCondition.optimal and self.config.load_solution:
            for v, val in self._primal_sol.items():
                v.value = val

            if self._writer.get_active_objective() is None:
                results.best_feasible_objective = None
            else:
                results.best_feasible_objective = value(self._writer.get_active_objective().expr)
        elif results.termination_condition == TerminationCondition.optimal:
            if self._writer.get_active_objective() is None:
                results.best_feasible_objective = None
            else:
                obj_expr_evaluated = replace_expressions(self._writer.get_active_objective().expr,
                                                         substitution_map={id(v): val for v, val in self._primal_sol.items()},
                                                         descend_into_named_expressions=True,
                                                         remove_named_expressions=True)
                results.best_feasible_objective = value(obj_expr_evaluated)
        elif self.config.load_solution:
            raise RuntimeError('A feasible solution was not found, so no solution can be loaded.'
                               'Please set opt.config.load_solution=False and check '
                               'results.termination_condition and '
                               'resutls.best_feasible_objective before loading a solution.')

        return results

    def _apply_solver(self, timer: HierarchicalTimer):
        config = self.config

        if config.time_limit is not None:
            timeout = config.time_limit + min(max(1, 0.01 * config.time_limit), 100)
        else:
            timeout = None

        out = open(self._filename + '.log', 'wb')
        err = out
        capture_output = False

        thread = None
        if config.stream_solver:
            thread = TeeThread(self._filename + '.log', stream_to_flush=out)
            thread.start()

        timer.start('subprocess')
        try:
            cp = subprocess.run([str(config.executable),
                                 self._filename + '.nl',
                                 '-AMPL',
                                 'halt_on_ampl_error=yes'],
                                timeout=timeout,
                                stdout=out,
                                stderr=err,
                                capture_output=capture_output)
        finally:
            if thread is not None:
                thread.event.set()
                thread.join()
            out.close()
        timer.stop('subprocess')

        if cp.returncode != 0:
            if self.config.load_solution:
                raise RuntimeError('A feasible solution was not found, so no solution can be loaded.'
                                   'Please set opt.config.load_solution=False and check '
                                   'results.termination_condition and '
                                   'results.best_feasible_objective before loading a solution.')
            results = Results()
            results.termination_condition = TerminationCondition.error
            results.best_feasible_objective = None
            self._primal_sol = None
            self._dual_sol = None
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

        return results

    def load_vars(self, vars_to_load: Optional[Sequence[_GeneralVarData]] = None) -> NoReturn:
        if vars_to_load is None:
            for v, val in self._primal_sol.items():
                v.value = val
        else:
            for v in vars_to_load:
                v.value = self._primal_sol[v]

    def get_duals(self, cons_to_load = None):
        if cons_to_load is None:
            return {k: v for k, v in self._dual_sol.items()}
        else:
            return {c: self._dual_sol[c] for c in cons_to_load}

    def get_reduced_costs(self, vars_to_load: Optional[Sequence[_GeneralVarData]] = None) -> Mapping[_GeneralVarData, float]:
        if vars_to_load is None:
            return ComponentMap((k, v) for k, v in self._reduced_costs.items())
        else:
            return ComponentMap((v, self._reduced_costs[v]) for v in vars_to_load)
