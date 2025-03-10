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
from pyomo.contrib.appsi.base import (
    PersistentSolver,
    Results,
    TerminationCondition,
    MIPSolverConfig,
    PersistentSolutionLoader,
)
from pyomo.contrib.appsi.writers import LPWriter
import logging
import math
from pyomo.common.collections import ComponentMap
from typing import Optional, Sequence, NoReturn, List, Mapping, Dict
from pyomo.core.base.var import VarData
from pyomo.core.base.constraint import ConstraintData
from pyomo.core.base.block import BlockData
from pyomo.core.base.param import ParamData
from pyomo.core.base.objective import ObjectiveData
from pyomo.common.timing import HierarchicalTimer
import sys
import time
from pyomo.common.log import LogStream
from pyomo.common.config import ConfigValue, NonNegativeInt
from pyomo.common.errors import PyomoException
from pyomo.contrib.appsi.cmodel import cmodel_available
from pyomo.core.staleflag import StaleFlagManager


logger = logging.getLogger(__name__)


class CplexConfig(MIPSolverConfig):
    def __init__(
        self,
        description=None,
        doc=None,
        implicit=False,
        implicit_domain=None,
        visibility=0,
    ):
        super(CplexConfig, self).__init__(
            description=description,
            doc=doc,
            implicit=implicit,
            implicit_domain=implicit_domain,
            visibility=visibility,
        )

        self.declare('filename', ConfigValue(domain=str))
        self.declare('keepfiles', ConfigValue(domain=bool))
        self.declare('solver_output_logger', ConfigValue())
        self.declare('log_level', ConfigValue(domain=NonNegativeInt))

        self.filename = None
        self.keepfiles = False
        self.solver_output_logger = logger
        self.log_level = logging.INFO


class CplexResults(Results):
    def __init__(self, solver):
        super(CplexResults, self).__init__()
        self.wallclock_time = None
        self.solution_loader = PersistentSolutionLoader(solver=solver)


class Cplex(PersistentSolver):
    _available = None

    def __init__(self, only_child_vars=False):
        self._config = CplexConfig()
        self._solver_options = dict()
        self._writer = LPWriter(only_child_vars=only_child_vars)
        self._filename = None
        self._last_results_object: Optional[CplexResults] = None

        try:
            import cplex

            self._cplex = cplex
            self._cplex_model: Optional[cplex.Cplex] = None
            self._cplex_available = True
        except ImportError:
            self._cplex = None
            self._cplex_model = None
            self._cplex_available = False

    @property
    def writer(self):
        return self._writer

    @property
    def symbol_map(self):
        return self._writer.symbol_map

    def available(self):
        if Cplex._available is None:
            self._check_license()
        return Cplex._available

    def _check_license(self):
        if self._cplex_available:
            if not cmodel_available:
                Cplex._available = self.Availability.NeedsCompiledExtension
            else:
                try:
                    m = self._cplex.Cplex()
                    m.set_results_stream(None)
                    m.variables.add(lb=[0] * 1001)
                    m.solve()
                    Cplex._available = self.Availability.FullLicense
                except self._cplex.exceptions.errors.CplexSolverError:
                    try:
                        m = self._cplex.Cplex()
                        m.set_results_stream(None)
                        m.variables.add(lb=[0])
                        m.solve()
                        Cplex._available = self.Availability.LimitedLicense
                    except:
                        Cplex._available = self.Availability.BadLicense
        else:
            Cplex._available = self.Availability.NotFound

    def version(self):
        return tuple(int(k) for k in self._cplex.Cplex().get_version().split('.'))

    def lp_filename(self):
        if self._filename is None:
            return None
        else:
            return self._filename + '.lp'

    def log_filename(self):
        if self._filename is None:
            return None
        else:
            return self._filename + '.log'

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, val):
        self._config = val

    @property
    def cplex_options(self):
        """
        A dictionary mapping solver options to values for those options. These
        are solver specific.

        Returns
        -------
        dict
            A dictionary mapping solver options to values for those options
        """
        return self._solver_options

    @cplex_options.setter
    def cplex_options(self, val: Dict):
        self._solver_options = val

    @property
    def update_config(self):
        return self._writer.update_config

    def set_instance(self, model):
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
                self._filename = TempfileManager.create_tempfile()
            else:
                self._filename = self.config.filename
            TempfileManager.add_tempfile(self._filename + '.lp', exists=False)
            TempfileManager.add_tempfile(self._filename + '.log', exists=False)
            timer.start('write lp file')
            self._writer.write(model, self._filename + '.lp', timer=timer)
            timer.stop('write lp file')
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

    def _apply_solver(self, timer: HierarchicalTimer):
        config = self.config

        timer.start('cplex read lp')
        self._cplex_model = cplex_model = self._cplex.Cplex()
        cplex_model.read(self._filename + '.lp')
        timer.stop('cplex read lp')

        log_stream = LogStream(
            level=self.config.log_level, logger=self.config.solver_output_logger
        )
        if config.stream_solver:

            def _process_stream(arg):
                sys.stdout.write(arg)
                return arg

            cplex_model.set_results_stream(log_stream, _process_stream)
        else:
            cplex_model.set_results_stream(log_stream)

        for key, option in self.cplex_options.items():
            opt_cmd = cplex_model.parameters
            key_pieces = key.split('_')
            for key_piece in key_pieces:
                opt_cmd = getattr(opt_cmd, key_piece)
            opt_cmd.set(option)

        if config.time_limit is not None:
            cplex_model.parameters.timelimit.set(config.time_limit)
        if config.mip_gap is not None:
            cplex_model.parameters.mip.tolerances.mipgap.set(config.mip_gap)

        timer.start('cplex solve')
        t0 = time.time()
        cplex_model.solve()
        t1 = time.time()
        timer.stop('cplex solve')

        return self._postsolve(timer, t1 - t0)

    def _postsolve(self, timer: HierarchicalTimer, solve_time):
        config = self.config
        cpxprob = self._cplex_model

        results = CplexResults(solver=self)
        results.wallclock_time = solve_time
        status = cpxprob.solution.get_status()

        if status in [1, 101, 102]:
            results.termination_condition = TerminationCondition.optimal
        elif status in [2, 40, 118, 133, 134]:
            results.termination_condition = TerminationCondition.unbounded
        elif status in [4, 119, 134]:
            results.termination_condition = TerminationCondition.infeasibleOrUnbounded
        elif status in [3, 103]:
            results.termination_condition = TerminationCondition.infeasible
        elif status in [10]:
            results.termination_condition = TerminationCondition.maxIterations
        elif status in [11, 25, 107, 131]:
            results.termination_condition = TerminationCondition.maxTimeLimit
        else:
            results.termination_condition = TerminationCondition.unknown

        if self._writer.get_active_objective() is None:
            results.best_feasible_objective = None
            results.best_objective_bound = None
        else:
            if cpxprob.solution.get_solution_type() != cpxprob.solution.type.none:
                if (
                    cpxprob.variables.get_num_binary()
                    + cpxprob.variables.get_num_integer()
                ) == 0:
                    results.best_feasible_objective = (
                        cpxprob.solution.get_objective_value()
                    )
                    results.best_objective_bound = (
                        cpxprob.solution.get_objective_value()
                    )
                else:
                    results.best_feasible_objective = (
                        cpxprob.solution.get_objective_value()
                    )
                    results.best_objective_bound = (
                        cpxprob.solution.MIP.get_best_objective()
                    )
            else:
                results.best_feasible_objective = None
                if cpxprob.objective.get_sense() == cpxprob.objective.sense.minimize:
                    results.best_objective_bound = -math.inf
                else:
                    results.best_objective_bound = math.inf

        if config.load_solution:
            if cpxprob.solution.get_solution_type() == cpxprob.solution.type.none:
                raise RuntimeError(
                    'A feasible solution was not found, so no solution can be loaded. '
                    'If using the appsi.solvers.Cplex interface, you can '
                    'set opt.config.load_solution=False. If using the environ.SolverFactory '
                    'interface, you can set opt.solve(model, load_solutions = False). '
                    'Then you can check results.termination_condition and '
                    'results.best_feasible_objective before loading a solution.'
                )
            else:
                if results.termination_condition != TerminationCondition.optimal:
                    logger.warning(
                        'Loading a feasible but suboptimal solution. '
                        'Please set load_solution=False and check '
                        'results.termination_condition before loading a solution.'
                    )
                timer.start('load solution')
                self.load_vars()
                timer.stop('load solution')

        return results

    def get_primals(
        self, vars_to_load: Optional[Sequence[VarData]] = None
    ) -> Mapping[VarData, float]:
        if (
            self._cplex_model.solution.get_solution_type()
            == self._cplex_model.solution.type.none
        ):
            raise RuntimeError(
                'Solver does not currently have a valid solution. Please '
                'check the termination condition.'
            )

        symbol_map = self._writer.symbol_map
        if vars_to_load is None:
            var_names = self._cplex_model.variables.get_names()
        else:
            var_names = [symbol_map.byObject[id(v)] for v in vars_to_load]
        var_vals = self._cplex_model.solution.get_values(var_names)
        res = ComponentMap()
        for name, val in zip(var_names, var_vals):
            if name == 'obj_const':
                continue
            v = symbol_map.bySymbol[name]
            if self._writer._referenced_variables[id(v)]:
                res[v] = val
        return res

    def get_duals(
        self, cons_to_load: Optional[Sequence[ConstraintData]] = None
    ) -> Dict[ConstraintData, float]:
        if (
            self._cplex_model.solution.get_solution_type()
            == self._cplex_model.solution.type.none
        ):
            raise RuntimeError(
                'Solver does not currently have valid duals. Please '
                'check the termination condition.'
            )

        if self._cplex_model.get_problem_type() in [
            self._cplex_model.problem_type.MILP,
            self._cplex_model.problem_type.MIQP,
            self._cplex_model.problem_type.MIQCP,
        ]:
            raise RuntimeError('Cannot get duals for mixed-integer problems')

        symbol_map = self._writer.symbol_map

        if cons_to_load is None:
            con_names = self._cplex_model.linear_constraints.get_names()
            dual_values = self._cplex_model.solution.get_dual_values()
        else:
            con_names = list()
            for con in cons_to_load:
                orig_name = symbol_map.byObject[id(con)]
                if con.equality:
                    con_names.append(orig_name + '_eq')
                else:
                    if con.lower is not None:
                        con_names.append(orig_name + '_lb')
                    if con.upper is not None:
                        con_names.append(orig_name + '_ub')
            dual_values = self._cplex_model.solution.get_dual_values(con_names)

        res = dict()
        for name, val in zip(con_names, dual_values):
            orig_name = name[:-3]
            if orig_name == 'obj_const_con':
                continue
            _con = symbol_map.bySymbol[orig_name]
            if _con in res:
                if abs(val) > abs(res[_con]):
                    res[_con] = val
            else:
                res[_con] = val

        return res

    def get_reduced_costs(
        self, vars_to_load: Optional[Sequence[VarData]] = None
    ) -> Mapping[VarData, float]:
        if (
            self._cplex_model.solution.get_solution_type()
            == self._cplex_model.solution.type.none
        ):
            raise RuntimeError(
                'Solver does not currently have valid reduced costs. Please '
                'check the termination condition.'
            )

        if self._cplex_model.get_problem_type() in [
            self._cplex_model.problem_type.MILP,
            self._cplex_model.problem_type.MIQP,
            self._cplex_model.problem_type.MIQCP,
        ]:
            raise RuntimeError('Cannot get reduced costs for mixed-integer problems')

        symbol_map = self._writer.symbol_map
        if vars_to_load is None:
            var_names = self._cplex_model.variables.get_names()
        else:
            var_names = [symbol_map.byObject[id(v)] for v in vars_to_load]
        rc = self._cplex_model.solution.get_reduced_costs(var_names)
        res = ComponentMap()
        for name, val in zip(var_names, rc):
            if name == 'obj_const':
                continue
            v = symbol_map.bySymbol[name]
            res[v] = val
        return res
