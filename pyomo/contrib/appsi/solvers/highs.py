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

import logging
from typing import List, Dict, Optional
from pyomo.common.collections import ComponentMap
from pyomo.common.dependencies import attempt_import
from pyomo.common.errors import PyomoException
from pyomo.common.flags import NOTSET
from pyomo.common.timing import HierarchicalTimer
from pyomo.common.config import ConfigValue, NonNegativeInt
from pyomo.common.tee import TeeStream, capture_output
from pyomo.common.log import LogStream
from pyomo.core.kernel.objective import minimize, maximize
from pyomo.core.base import SymbolMap
from pyomo.core.base.var import VarData
from pyomo.core.base.constraint import ConstraintData
from pyomo.core.base.sos import SOSConstraintData
from pyomo.core.base.param import ParamData
from pyomo.core.expr.numvalue import value, is_constant
from pyomo.repn import generate_standard_repn
from pyomo.core.expr.numeric_expr import NPV_MaxExpression, NPV_MinExpression
from pyomo.contrib.appsi.base import (
    PersistentSolver,
    Results,
    TerminationCondition,
    MIPSolverConfig,
    PersistentBase,
    PersistentSolutionLoader,
)
from pyomo.contrib.appsi.cmodel import cmodel, cmodel_available
from pyomo.common.dependencies import numpy as np
from pyomo.core.staleflag import StaleFlagManager
import sys

logger = logging.getLogger(__name__)

highspy, highspy_available = attempt_import('highspy')


class DegreeError(PyomoException):
    pass


class HighsConfig(MIPSolverConfig):
    def __init__(
        self,
        description=None,
        doc=None,
        implicit=False,
        implicit_domain=None,
        visibility=0,
    ):
        super(HighsConfig, self).__init__(
            description=description,
            doc=doc,
            implicit=implicit,
            implicit_domain=implicit_domain,
            visibility=visibility,
        )

        self.declare('logfile', ConfigValue(domain=str))
        self.declare('solver_output_logger', ConfigValue())
        self.declare('log_level', ConfigValue(domain=NonNegativeInt))

        self.logfile = ''
        self.solver_output_logger = logger
        self.log_level = logging.INFO


class HighsResults(Results):
    def __init__(self, solver):
        super().__init__()
        self.wallclock_time = None
        self.solution_loader = PersistentSolutionLoader(solver=solver)


class _MutableVarBounds(object):
    def __init__(self, lower_expr, upper_expr, pyomo_var_id, var_map, highs):
        self.pyomo_var_id = pyomo_var_id
        self.lower_expr = lower_expr
        self.upper_expr = upper_expr
        self.var_map = var_map
        self.highs = highs

    def update(self):
        col_ndx = self.var_map[self.pyomo_var_id]
        lb = value(self.lower_expr)
        ub = value(self.upper_expr)
        self.highs.changeColBounds(col_ndx, lb, ub)


class _MutableLinearCoefficient(object):
    def __init__(self, pyomo_con, pyomo_var_id, con_map, var_map, expr, highs):
        self.expr = expr
        self.highs = highs
        self.pyomo_var_id = pyomo_var_id
        self.pyomo_con = pyomo_con
        self.con_map = con_map
        self.var_map = var_map

    def update(self):
        row_ndx = self.con_map[self.pyomo_con]
        col_ndx = self.var_map[self.pyomo_var_id]
        self.highs.changeCoeff(row_ndx, col_ndx, value(self.expr))


class _MutableObjectiveCoefficient(object):
    def __init__(self, pyomo_var_id, var_map, expr, highs):
        self.expr = expr
        self.highs = highs
        self.pyomo_var_id = pyomo_var_id
        self.var_map = var_map

    def update(self):
        col_ndx = self.var_map[self.pyomo_var_id]
        self.highs.changeColCost(col_ndx, value(self.expr))


class _MutableObjectiveOffset(object):
    def __init__(self, expr, highs):
        self.expr = expr
        self.highs = highs

    def update(self):
        self.highs.changeObjectiveOffset(value(self.expr))


class _MutableConstraintBounds(object):
    def __init__(self, lower_expr, upper_expr, pyomo_con, con_map, highs):
        self.lower_expr = lower_expr
        self.upper_expr = upper_expr
        self.con = pyomo_con
        self.con_map = con_map
        self.highs = highs

    def update(self):
        row_ndx = self.con_map[self.con]
        lb = value(self.lower_expr)
        ub = value(self.upper_expr)
        self.highs.changeRowBounds(row_ndx, lb, ub)


class Highs(PersistentBase, PersistentSolver):
    """
    Interface to HiGHS
    """

    _available = None

    def __init__(self, only_child_vars=False):
        super().__init__(only_child_vars=only_child_vars)
        self._config = HighsConfig()
        self._solver_options = dict()
        self._solver_model = None
        self._pyomo_var_to_solver_var_map = dict()
        self._pyomo_con_to_solver_con_map = dict()
        self._solver_con_to_pyomo_con_map = dict()
        self._mutable_helpers = dict()
        self._mutable_bounds = dict()
        self._objective_helpers = list()
        self._last_results_object: Optional[HighsResults] = None
        self._sol = None

    def available(self):
        if highspy_available:
            return self.Availability.FullLicense
        else:
            return self.Availability.NotFound

    def version(self):
        try:
            version = (
                highspy.HIGHS_VERSION_MAJOR,
                highspy.HIGHS_VERSION_MINOR,
                highspy.HIGHS_VERSION_PATCH,
            )
        except AttributeError:
            # Older versions of Highs do not have the above attributes
            # and the solver version can only be obtained by making
            # an instance of the solver class.
            tmp = highspy.Highs()
            version = (tmp.versionMajor(), tmp.versionMinor(), tmp.versionPatch())

        return version

    @property
    def config(self) -> HighsConfig:
        return self._config

    @config.setter
    def config(self, val: HighsConfig):
        self._config = val

    @property
    def highs_options(self):
        """
        A dictionary mapping solver options to values for those options. These
        are solver specific.

        Returns
        -------
        dict
           A dictionary mapping solver options to values for those options
        """
        return self._solver_options

    @highs_options.setter
    def highs_options(self, val: Dict):
        self._solver_options = val

    @property
    def symbol_map(self):
        return SymbolMap()
        # raise RuntimeError('Highs interface does not have a symbol map')

    def warm_start_capable(self):
        return True

    def _warm_start(self):
        # Collect all variable values
        col_value = np.zeros(len(self._pyomo_var_to_solver_var_map))
        has_values = False

        for var_id, col_ndx in self._pyomo_var_to_solver_var_map.items():
            var = self._vars[var_id][0]
            if var.value is not None:
                col_value[col_ndx] = value(var)
                has_values = True

        if has_values:
            solution = highspy.HighsSolution()
            solution.col_value = col_value
            solution.value_valid = True
            solution.dual_valid = False

            # Set the solution as a MIP start
            self._solver_model.setSolution(solution)

    def _solve(self, timer: HierarchicalTimer):
        config = self.config
        options = self.highs_options

        ostreams = [
            LogStream(
                level=self.config.log_level, logger=self.config.solver_output_logger
            )
        ]
        if self.config.stream_solver:
            ostreams.append(sys.stdout)

        with capture_output(output=TeeStream(*ostreams), capture_fd=True):
            self._solver_model.setOptionValue('log_to_console', True)
            if config.logfile != '':
                self._solver_model.setOptionValue('log_file', config.logfile)

            if config.time_limit is not None:
                self._solver_model.setOptionValue('time_limit', config.time_limit)
            if config.mip_gap is not None:
                self._solver_model.setOptionValue('mip_rel_gap', config.mip_gap)

            for key, option in options.items():
                self._solver_model.setOptionValue(key, option)

            if config.warmstart:
                self._warm_start()
            timer.start('optimize')
            if self.version()[:2] >= (1, 8):
                self._solver_model.HandleKeyboardInterrupt = True
            self._solver_model.run()
            timer.stop('optimize')

        return self._postsolve(timer)

    def solve(self, model, timer: HierarchicalTimer = None) -> Results:
        StaleFlagManager.mark_all_as_stale()
        if self._last_results_object is not None:
            self._last_results_object.solution_loader.invalidate()
        if timer is None:
            timer = HierarchicalTimer()
        if model is not self._model:
            timer.start('set_instance')
            self.set_instance(model)
            timer.stop('set_instance')
        else:
            timer.start('update')
            self.update(timer=timer)
            timer.stop('update')
        res = self._solve(timer)
        self._last_results_object = res
        if self.config.report_timing:
            logger.info('\n' + str(timer))
        return res

    def _process_domain_and_bounds(self, var_id):
        _v, _lb, _ub, _fixed, _domain_interval, _value = self._vars[var_id]
        lb, ub, step = _domain_interval
        if lb is None:
            lb = -highspy.kHighsInf
        if ub is None:
            ub = highspy.kHighsInf
        if step == 0:
            vtype = highspy.HighsVarType.kContinuous
        elif step == 1:
            vtype = highspy.HighsVarType.kInteger
        else:
            raise ValueError(
                f'Unrecognized domain step: {step} (should be either 0 or 1)'
            )
        if _fixed:
            lb = _value
            ub = _value
        else:
            if _lb is not None or _ub is not None:
                if not is_constant(_lb) or not is_constant(_ub):
                    if _lb is None:
                        tmp_lb = -highspy.kHighsInf
                    else:
                        tmp_lb = _lb
                    if _ub is None:
                        tmp_ub = highspy.kHighsInf
                    else:
                        tmp_ub = _ub
                    mutable_bound = _MutableVarBounds(
                        lower_expr=NPV_MaxExpression((tmp_lb, lb)),
                        upper_expr=NPV_MinExpression((tmp_ub, ub)),
                        pyomo_var_id=var_id,
                        var_map=self._pyomo_var_to_solver_var_map,
                        highs=self._solver_model,
                    )
                    self._mutable_bounds[var_id] = (_v, mutable_bound)
            if _lb is not None:
                lb = max(value(_lb), lb)
            if _ub is not None:
                ub = min(value(_ub), ub)

        return lb, ub, vtype

    def _add_variables(self, variables: List[VarData]):
        self._sol = None
        if self._last_results_object is not None:
            self._last_results_object.solution_loader.invalidate()
        lbs = list()
        ubs = list()
        indices = list()
        vtypes = list()

        current_num_vars = len(self._pyomo_var_to_solver_var_map)
        for v in variables:
            v_id = id(v)
            lb, ub, vtype = self._process_domain_and_bounds(v_id)
            lbs.append(lb)
            ubs.append(ub)
            vtypes.append(vtype)
            indices.append(current_num_vars)
            self._pyomo_var_to_solver_var_map[v_id] = current_num_vars
            current_num_vars += 1

        self._solver_model.addVars(
            len(lbs), np.array(lbs, dtype=np.double), np.array(ubs, dtype=np.double)
        )
        self._solver_model.changeColsIntegrality(
            len(vtypes), np.array(indices), np.array(vtypes)
        )

    def _add_params(self, params: List[ParamData]):
        pass

    def _reinit(self):
        saved_config = self.config
        saved_options = self.highs_options
        saved_update_config = self.update_config
        self.__init__(only_child_vars=self._only_child_vars)
        self.config = saved_config
        self.highs_options = saved_options
        self.update_config = saved_update_config

    def set_instance(self, model):
        if self._last_results_object is not None:
            self._last_results_object.solution_loader.invalidate()
        if not self.available():
            c = self.__class__
            raise PyomoException(
                f'Solver {c.__module__}.{c.__qualname__} is not available '
                f'({self.available()}).'
            )

        ostreams = [
            LogStream(
                level=self.config.log_level, logger=self.config.solver_output_logger
            )
        ]
        if self.config.stream_solver:
            ostreams.append(sys.stdout)
        with capture_output(output=TeeStream(*ostreams), capture_fd=True):
            self._reinit()
            self._model = model
            if self.use_extensions and cmodel_available:
                self._expr_types = cmodel.PyomoExprTypes()

            self._solver_model = highspy.Highs()
            self.add_block(model)
            if self._objective is None:
                self.set_objective(None)

    def _add_constraints(self, cons: List[ConstraintData]):
        self._sol = None
        if self._last_results_object is not None:
            self._last_results_object.solution_loader.invalidate()
        current_num_cons = len(self._pyomo_con_to_solver_con_map)
        lbs = list()
        ubs = list()
        starts = list()
        var_indices = list()
        coef_values = list()

        for con in cons:
            repn = generate_standard_repn(
                con.body, quadratic=False, compute_values=False
            )
            if repn.nonlinear_expr is not None:
                raise DegreeError(
                    f'Highs interface does not support expressions of degree {repn.polynomial_degree()}'
                )

            starts.append(len(coef_values))
            for ndx, coef in enumerate(repn.linear_coefs):
                v = repn.linear_vars[ndx]
                v_id = id(v)
                coef_val = value(coef)
                if not is_constant(coef):
                    mutable_linear_coefficient = _MutableLinearCoefficient(
                        pyomo_con=con,
                        pyomo_var_id=v_id,
                        con_map=self._pyomo_con_to_solver_con_map,
                        var_map=self._pyomo_var_to_solver_var_map,
                        expr=coef,
                        highs=self._solver_model,
                    )
                    if con not in self._mutable_helpers:
                        self._mutable_helpers[con] = list()
                    self._mutable_helpers[con].append(mutable_linear_coefficient)
                    if coef_val == 0:
                        continue
                var_indices.append(self._pyomo_var_to_solver_var_map[v_id])
                coef_values.append(coef_val)

            if con.has_lb():
                lb = con.lower - repn.constant
            else:
                lb = -highspy.kHighsInf
            if con.has_ub():
                ub = con.upper - repn.constant
            else:
                ub = highspy.kHighsInf

            if not is_constant(lb) or not is_constant(ub):
                mutable_con_bounds = _MutableConstraintBounds(
                    lower_expr=lb,
                    upper_expr=ub,
                    pyomo_con=con,
                    con_map=self._pyomo_con_to_solver_con_map,
                    highs=self._solver_model,
                )
                if con not in self._mutable_helpers:
                    self._mutable_helpers[con] = [mutable_con_bounds]
                else:
                    self._mutable_helpers[con].append(mutable_con_bounds)

            lbs.append(value(lb))
            ubs.append(value(ub))
            self._pyomo_con_to_solver_con_map[con] = current_num_cons
            self._solver_con_to_pyomo_con_map[current_num_cons] = con
            current_num_cons += 1

        self._solver_model.addRows(
            len(lbs),
            np.array(lbs, dtype=np.double),
            np.array(ubs, dtype=np.double),
            len(coef_values),
            np.array(starts),
            np.array(var_indices),
            np.array(coef_values, dtype=np.double),
        )

    def _add_sos_constraints(self, cons: List[SOSConstraintData]):
        if cons:
            raise NotImplementedError(
                'Highs interface does not support SOS constraints'
            )

    def _remove_constraints(self, cons: List[ConstraintData]):
        self._sol = None
        if self._last_results_object is not None:
            self._last_results_object.solution_loader.invalidate()
        indices_to_remove = list()
        for con in cons:
            con_ndx = self._pyomo_con_to_solver_con_map.pop(con)
            del self._solver_con_to_pyomo_con_map[con_ndx]
            indices_to_remove.append(con_ndx)
            self._mutable_helpers.pop(con, None)
        self._solver_model.deleteRows(
            len(indices_to_remove), np.sort(np.array(indices_to_remove))
        )
        con_ndx = 0
        new_con_map = dict()
        for c in self._pyomo_con_to_solver_con_map.keys():
            new_con_map[c] = con_ndx
            con_ndx += 1
        self._pyomo_con_to_solver_con_map.clear()
        self._pyomo_con_to_solver_con_map.update(new_con_map)
        self._solver_con_to_pyomo_con_map.clear()
        self._solver_con_to_pyomo_con_map.update(
            {v: k for k, v in self._pyomo_con_to_solver_con_map.items()}
        )

    def _remove_sos_constraints(self, cons: List[SOSConstraintData]):
        if cons:
            raise NotImplementedError(
                'Highs interface does not support SOS constraints'
            )

    def _remove_variables(self, variables: List[VarData]):
        self._sol = None
        if self._last_results_object is not None:
            self._last_results_object.solution_loader.invalidate()
        indices_to_remove = list()
        for v in variables:
            v_id = id(v)
            v_ndx = self._pyomo_var_to_solver_var_map.pop(v_id)
            indices_to_remove.append(v_ndx)
            self._mutable_bounds.pop(v_id, None)
        indices_to_remove.sort()
        self._solver_model.deleteVars(
            len(indices_to_remove), np.array(indices_to_remove)
        )
        v_ndx = 0
        new_var_map = dict()
        for v_id in self._pyomo_var_to_solver_var_map.keys():
            new_var_map[v_id] = v_ndx
            v_ndx += 1
        self._pyomo_var_to_solver_var_map.clear()
        self._pyomo_var_to_solver_var_map.update(new_var_map)

    def _remove_params(self, params: List[ParamData]):
        pass

    def _update_variables(self, variables: List[VarData]):
        self._sol = None
        if self._last_results_object is not None:
            self._last_results_object.solution_loader.invalidate()
        indices = list()
        lbs = list()
        ubs = list()
        vtypes = list()

        for v in variables:
            v_id = id(v)
            self._mutable_bounds.pop(v_id, None)
            v_ndx = self._pyomo_var_to_solver_var_map[v_id]
            lb, ub, vtype = self._process_domain_and_bounds(v_id)
            lbs.append(lb)
            ubs.append(ub)
            vtypes.append(vtype)
            indices.append(v_ndx)

        self._solver_model.changeColsBounds(
            len(indices),
            np.array(indices),
            np.array(lbs, dtype=np.double),
            np.array(ubs, dtype=np.double),
        )
        self._solver_model.changeColsIntegrality(
            len(indices), np.array(indices), np.array(vtypes)
        )

    def update_params(self):
        self._sol = None
        if self._last_results_object is not None:
            self._last_results_object.solution_loader.invalidate()
        for con, helpers in self._mutable_helpers.items():
            for helper in helpers:
                helper.update()
        for k, (v, helper) in self._mutable_bounds.items():
            helper.update()
        for helper in self._objective_helpers:
            helper.update()

    def _set_objective(self, obj):
        self._sol = None
        if self._last_results_object is not None:
            self._last_results_object.solution_loader.invalidate()
        n = len(self._pyomo_var_to_solver_var_map)
        indices = np.arange(n)
        costs = np.zeros(n, dtype=np.double)
        self._objective_helpers = list()
        if obj is None:
            sense = highspy.ObjSense.kMinimize
            self._solver_model.changeObjectiveOffset(0)
        else:
            if obj.sense == minimize:
                sense = highspy.ObjSense.kMinimize
            elif obj.sense == maximize:
                sense = highspy.ObjSense.kMaximize
            else:
                raise ValueError(
                    'Objective sense is not recognized: {0}'.format(obj.sense)
                )

            repn = generate_standard_repn(
                obj.expr, quadratic=False, compute_values=False
            )
            if repn.nonlinear_expr is not None:
                raise DegreeError(
                    f'Highs interface does not support expressions of degree {repn.polynomial_degree()}'
                )

            for coef, v in zip(repn.linear_coefs, repn.linear_vars):
                v_id = id(v)
                v_ndx = self._pyomo_var_to_solver_var_map[v_id]
                costs[v_ndx] = value(coef)
                if not is_constant(coef):
                    mutable_objective_coef = _MutableObjectiveCoefficient(
                        pyomo_var_id=v_id,
                        var_map=self._pyomo_var_to_solver_var_map,
                        expr=coef,
                        highs=self._solver_model,
                    )
                    self._objective_helpers.append(mutable_objective_coef)

            self._solver_model.changeObjectiveOffset(value(repn.constant))
            if not is_constant(repn.constant):
                mutable_objective_offset = _MutableObjectiveOffset(
                    expr=repn.constant, highs=self._solver_model
                )
                self._objective_helpers.append(mutable_objective_offset)

        self._solver_model.changeObjectiveSense(sense)
        self._solver_model.changeColsCost(n, indices, costs)

    def _postsolve(self, timer: HierarchicalTimer):
        config = self.config

        highs = self._solver_model
        status = highs.getModelStatus()

        results = HighsResults(self)
        results.wallclock_time = highs.getRunTime()

        if status == highspy.HighsModelStatus.kNotset:
            results.termination_condition = TerminationCondition.unknown
        elif status == highspy.HighsModelStatus.kLoadError:
            results.termination_condition = TerminationCondition.error
        elif status == highspy.HighsModelStatus.kModelError:
            results.termination_condition = TerminationCondition.error
        elif status == highspy.HighsModelStatus.kPresolveError:
            results.termination_condition = TerminationCondition.error
        elif status == highspy.HighsModelStatus.kSolveError:
            results.termination_condition = TerminationCondition.error
        elif status == highspy.HighsModelStatus.kPostsolveError:
            results.termination_condition = TerminationCondition.error
        elif status == highspy.HighsModelStatus.kModelEmpty:
            results.termination_condition = TerminationCondition.unknown
        elif status == highspy.HighsModelStatus.kOptimal:
            results.termination_condition = TerminationCondition.optimal
        elif status == highspy.HighsModelStatus.kInfeasible:
            results.termination_condition = TerminationCondition.infeasible
        elif status == highspy.HighsModelStatus.kUnboundedOrInfeasible:
            results.termination_condition = TerminationCondition.infeasibleOrUnbounded
        elif status == highspy.HighsModelStatus.kUnbounded:
            results.termination_condition = TerminationCondition.unbounded
        elif status == highspy.HighsModelStatus.kObjectiveBound:
            results.termination_condition = TerminationCondition.objectiveLimit
        elif status == highspy.HighsModelStatus.kObjectiveTarget:
            results.termination_condition = TerminationCondition.objectiveLimit
        elif status == highspy.HighsModelStatus.kTimeLimit:
            results.termination_condition = TerminationCondition.maxTimeLimit
        elif status == highspy.HighsModelStatus.kIterationLimit:
            results.termination_condition = TerminationCondition.maxIterations
        elif status == getattr(highspy.HighsModelStatus, "kSolutionLimit", NOTSET):
            # kSolutionLimit was introduced in HiGHS v1.5.3 for MIP-related limits
            results.termination_condition = TerminationCondition.maxIterations
        elif status == highspy.HighsModelStatus.kUnknown:
            results.termination_condition = TerminationCondition.unknown
        else:
            logger.warning(f'Received unhandled {status=} from solver HiGHS.')
            results.termination_condition = TerminationCondition.unknown

        timer.start('load solution')
        self._sol = highs.getSolution()
        has_feasible_solution = False
        if results.termination_condition == TerminationCondition.optimal:
            has_feasible_solution = True
        elif results.termination_condition in {
            TerminationCondition.objectiveLimit,
            TerminationCondition.maxIterations,
            TerminationCondition.maxTimeLimit,
        }:
            if self._sol.value_valid:
                has_feasible_solution = True

        if config.load_solution:
            if has_feasible_solution:
                if results.termination_condition != TerminationCondition.optimal:
                    logger.warning(
                        'Loading a feasible but suboptimal solution. '
                        'Please set load_solution=False and check '
                        'results.termination_condition and '
                        'results.found_feasible_solution() before loading a solution.'
                    )
                self.load_vars()
            else:
                raise RuntimeError(
                    'A feasible solution was not found, so no solution can be loaded. '
                    'If using the appsi.solvers.Highs interface, you can '
                    'set opt.config.load_solution=False. If using the environ.SolverFactory '
                    'interface, you can set opt.solve(model, load_solutions = False). '
                    'Then you can check results.termination_condition and '
                    'results.best_feasible_objective before loading a solution.'
                )
        timer.stop('load solution')

        info = highs.getInfo()
        results.best_objective_bound = None
        results.best_feasible_objective = None
        if self._objective is not None:
            if has_feasible_solution:
                results.best_feasible_objective = info.objective_function_value
            if info.mip_node_count == -1:
                if has_feasible_solution:
                    results.best_objective_bound = info.objective_function_value
                else:
                    results.best_objective_bound = None
            else:
                results.best_objective_bound = info.mip_dual_bound

        return results

    def load_vars(self, vars_to_load=None):
        for v, val in self.get_primals(vars_to_load=vars_to_load).items():
            v.set_value(val, skip_validation=True)
        StaleFlagManager.mark_all_as_stale(delayed=True)

    def get_primals(self, vars_to_load=None, solution_number=0):
        if self._sol is None or not self._sol.value_valid:
            raise RuntimeError(
                'Solver does not currently have a valid solution. Please '
                'check the termination condition.'
            )

        res = ComponentMap()
        if vars_to_load is None:
            var_ids_to_load = list()
            for v, ref_info in self._referenced_variables.items():
                using_cons, using_sos, using_obj = ref_info
                if using_cons or using_sos or (using_obj is not None):
                    var_ids_to_load.append(v)
        else:
            var_ids_to_load = [id(v) for v in vars_to_load]

        var_vals = self._sol.col_value

        for v_id in var_ids_to_load:
            v = self._vars[v_id][0]
            v_ndx = self._pyomo_var_to_solver_var_map[v_id]
            res[v] = var_vals[v_ndx]

        return res

    def get_reduced_costs(self, vars_to_load=None):
        if self._sol is None or not self._sol.dual_valid:
            raise RuntimeError(
                'Solver does not currently have valid reduced costs. Please '
                'check the termination condition.'
            )
        res = ComponentMap()
        if vars_to_load is None:
            var_ids_to_load = list(self._vars.keys())
        else:
            var_ids_to_load = [id(v) for v in vars_to_load]

        var_vals = self._sol.col_dual

        for v_id in var_ids_to_load:
            v = self._vars[v_id][0]
            v_ndx = self._pyomo_var_to_solver_var_map[v_id]
            res[v] = var_vals[v_ndx]

        return res

    def get_duals(self, cons_to_load=None):
        if self._sol is None or not self._sol.dual_valid:
            raise RuntimeError(
                'Solver does not currently have valid duals. Please '
                'check the termination condition.'
            )

        res = dict()
        if cons_to_load is None:
            cons_to_load = list(self._pyomo_con_to_solver_con_map.keys())

        duals = self._sol.row_dual

        for c in cons_to_load:
            c_ndx = self._pyomo_con_to_solver_con_map[c]
            res[c] = duals[c_ndx]

        return res

    def get_slacks(self, cons_to_load=None):
        if self._sol is None or not self._sol.value_valid:
            raise RuntimeError(
                'Solver does not currently have valid slacks. Please '
                'check the termination condition.'
            )

        res = dict()
        if cons_to_load is None:
            cons_to_load = list(self._pyomo_con_to_solver_con_map.keys())

        slacks = self._sol.row_value

        for c in cons_to_load:
            c_ndx = self._pyomo_con_to_solver_con_map[c]
            res[c] = slacks[c_ndx]

        return res
