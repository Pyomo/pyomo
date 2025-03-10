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

from pyomo.contrib.appsi.base import (
    PersistentBase,
    PersistentSolver,
    SolverConfig,
    Results,
    TerminationCondition,
    PersistentSolutionLoader,
)
from pyomo.core.expr.numeric_expr import (
    ProductExpression,
    DivisionExpression,
    PowExpression,
    SumExpression,
    MonomialTermExpression,
    NegationExpression,
    UnaryFunctionExpression,
    LinearExpression,
    AbsExpression,
    NPV_ProductExpression,
    NPV_DivisionExpression,
    NPV_PowExpression,
    NPV_SumExpression,
    NPV_NegationExpression,
    NPV_UnaryFunctionExpression,
    NPV_AbsExpression,
)
from pyomo.common.errors import PyomoException
from pyomo.common.collections import ComponentMap
from pyomo.core.expr.numvalue import native_numeric_types
from typing import Dict, Optional, List
from pyomo.core.base.block import BlockData
from pyomo.core.base.var import VarData
from pyomo.core.base.param import ParamData
from pyomo.core.base.constraint import ConstraintData
from pyomo.common.timing import HierarchicalTimer
from pyomo.core.base import SymbolMap, NumericLabeler, TextLabeler
from pyomo.common.dependencies import attempt_import
from pyomo.core.staleflag import StaleFlagManager
from pyomo.contrib.appsi.cmodel import cmodel, cmodel_available

wntr, wntr_available = attempt_import('wntr')
import logging
import time
import sys
from pyomo.core.expr.visitor import ExpressionValueVisitor


logger = logging.getLogger(__name__)


class WntrConfig(SolverConfig):
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


class WntrResults(Results):
    def __init__(self, solver):
        super().__init__()
        self.wallclock_time = None
        self.solution_loader = PersistentSolutionLoader(solver=solver)


class Wntr(PersistentBase, PersistentSolver):
    def __init__(self, only_child_vars=True):
        super().__init__(only_child_vars=only_child_vars)
        self._config = WntrConfig()
        self._solver_options = dict()
        self._solver_model = None
        self._symbol_map = SymbolMap()
        self._labeler = None
        self._pyomo_var_to_solver_var_map = dict()
        self._pyomo_con_to_solver_con_map = dict()
        self._pyomo_param_to_solver_param_map = dict()
        self._needs_updated = True
        self._last_results_object: Optional[WntrResults] = None
        self._pyomo_to_wntr_visitor = PyomoToWntrVisitor(
            self._pyomo_var_to_solver_var_map, self._pyomo_param_to_solver_param_map
        )

    def available(self):
        if wntr_available:
            return self.Availability.FullLicense
        else:
            return self.Availability.NotFound

    def version(self):
        return tuple(int(i) for i in wntr.__version__.split('.'))

    @property
    def config(self) -> WntrConfig:
        return self._config

    @config.setter
    def config(self, val: WntrConfig):
        self._config = val

    @property
    def wntr_options(self):
        return self._solver_options

    @wntr_options.setter
    def wntr_options(self, val: Dict):
        self._solver_options = val

    @property
    def symbol_map(self):
        return self._symbol_map

    def _solve(self, timer: HierarchicalTimer):
        options = dict()
        if self.config.time_limit is not None:
            options['TIME_LIMIT'] = self.config.time_limit
        options.update(self.wntr_options)
        opt = wntr.sim.solvers.NewtonSolver(options)

        if self.config.stream_solver:
            ostream = sys.stdout
        else:
            ostream = None

        t0 = time.time()
        if self._needs_updated:
            timer.start('set_structure')
            self._solver_model.set_structure()
            timer.stop('set_structure')
            self._needs_updated = False
        timer.start('newton solve')
        status, msg, num_iter = opt.solve(self._solver_model, ostream)
        timer.stop('newton solve')
        tf = time.time()

        results = WntrResults(self)
        results.wallclock_time = tf - t0
        if status == wntr.sim.solvers.SolverStatus.converged:
            results.termination_condition = TerminationCondition.optimal
        else:
            results.termination_condition = TerminationCondition.error
        results.best_feasible_objective = None
        results.best_objective_bound = None

        if self.config.load_solution:
            if status == wntr.sim.solvers.SolverStatus.converged:
                timer.start('load solution')
                self.load_vars()
                timer.stop('load solution')
            else:
                raise RuntimeError(
                    'A feasible solution was not found, so no solution can be loaded. '
                    'If using the appsi.solvers.Wntr interface, you can '
                    'set opt.config.load_solution=False. If using the environ.SolverFactory '
                    'interface, you can set opt.solve(model, load_solutions = False). '
                    'Then you can check results.termination_condition and '
                    'results.best_feasible_objective before loading a solution.'
                )
        return results

    def solve(self, model: BlockData, timer: HierarchicalTimer = None) -> Results:
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
            timer.start('initial values')
            for v_id, solver_v in self._pyomo_var_to_solver_var_map.items():
                pyomo_v = self._vars[v_id][0]
                val = pyomo_v.value
                if val is not None:
                    solver_v.value = val
            timer.stop('initial values')
            timer.stop('update')
        res = self._solve(timer)
        self._last_results_object = res
        if self.config.report_timing:
            logger.info('\n' + str(timer))
        return res

    def _reinit(self):
        saved_config = self.config
        saved_options = self.wntr_options
        saved_update_config = self.update_config
        self.__init__(only_child_vars=self._only_child_vars)
        self.config = saved_config
        self.wntr_options = saved_options
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
        self._reinit()
        self._model = model
        if self.use_extensions and cmodel_available:
            self._expr_types = cmodel.PyomoExprTypes()

        if self.config.symbolic_solver_labels:
            self._labeler = TextLabeler()
        else:
            self._labeler = NumericLabeler('x')

        self._solver_model = wntr.sim.aml.aml.Model()
        self._solver_model._wntr_fixed_var_params = wntr.sim.aml.aml.ParamDict()
        self._solver_model._wntr_fixed_var_cons = wntr.sim.aml.aml.ConstraintDict()

        self.add_block(model)

    def _add_variables(self, variables: List[VarData]):
        aml = wntr.sim.aml.aml
        for var in variables:
            varname = self._symbol_map.getSymbol(var, self._labeler)
            _v, _lb, _ub, _fixed, _domain_interval, _value = self._vars[id(var)]
            lb, ub, step = _domain_interval
            if (
                _lb is not None
                or _ub is not None
                or lb is not None
                or ub is not None
                or step != 0
            ):
                raise ValueError(
                    f"WNTR's newton solver only supports continuous variables without bounds: {var.name}"
                )
            if _value is None:
                _value = 0
            wntr_var = aml.Var(_value)
            setattr(self._solver_model, varname, wntr_var)
            self._pyomo_var_to_solver_var_map[id(var)] = wntr_var
            if _fixed:
                self._solver_model._wntr_fixed_var_params[id(var)] = param = aml.Param(
                    _value
                )
                wntr_expr = wntr_var - param
                self._solver_model._wntr_fixed_var_cons[id(var)] = aml.Constraint(
                    wntr_expr
                )
            self._needs_updated = True

    def _add_params(self, params: List[ParamData]):
        aml = wntr.sim.aml.aml
        for p in params:
            pname = self._symbol_map.getSymbol(p, self._labeler)
            wntr_p = aml.Param(p.value)
            setattr(self._solver_model, pname, wntr_p)
            self._pyomo_param_to_solver_param_map[id(p)] = wntr_p

    def _add_constraints(self, cons: List[ConstraintData]):
        aml = wntr.sim.aml.aml
        for con in cons:
            if not con.equality:
                raise ValueError(
                    f"WNTR's newtwon solver only supports equality constraints: {con.name}"
                )
            conname = self._symbol_map.getSymbol(con, self._labeler)
            wntr_expr = self._pyomo_to_wntr_visitor.dfs_postorder_stack(
                con.body - con.upper
            )
            wntr_con = aml.Constraint(wntr_expr)
            setattr(self._solver_model, conname, wntr_con)
            self._pyomo_con_to_solver_con_map[con] = wntr_con
            self._needs_updated = True

    def _remove_constraints(self, cons: List[ConstraintData]):
        for con in cons:
            solver_con = self._pyomo_con_to_solver_con_map[con]
            delattr(self._solver_model, solver_con.name)
            self._symbol_map.removeSymbol(con)
            del self._pyomo_con_to_solver_con_map[con]
            self._needs_updated = True

    def _remove_variables(self, variables: List[VarData]):
        for var in variables:
            v_id = id(var)
            solver_var = self._pyomo_var_to_solver_var_map[v_id]
            delattr(self._solver_model, solver_var.name)
            self._symbol_map.removeSymbol(var)
            del self._pyomo_var_to_solver_var_map[v_id]
            if v_id in self._solver_model._wntr_fixed_var_params:
                del self._solver_model._wntr_fixed_var_params[v_id]
                del self._solver_model._wntr_fixed_var_cons[v_id]
            self._needs_updated = True

    def _remove_params(self, params: List[ParamData]):
        for p in params:
            p_id = id(p)
            solver_param = self._pyomo_param_to_solver_param_map[p_id]
            delattr(self._solver_model, solver_param.name)
            self._symbol_map.removeSymbol(p)
            del self._pyomo_param_to_solver_param_map[p_id]

    def _update_variables(self, variables: List[VarData]):
        aml = wntr.sim.aml.aml
        for var in variables:
            v_id = id(var)
            solver_var = self._pyomo_var_to_solver_var_map[v_id]
            _v, _lb, _ub, _fixed, _domain_interval, _value = self._vars[v_id]
            lb, ub, step = _domain_interval
            if (
                _lb is not None
                or _ub is not None
                or lb is not None
                or ub is not None
                or step != 0
            ):
                raise ValueError(
                    f"WNTR's newton solver only supports continuous variables without bounds: {var.name}"
                )
            if _value is None:
                _value = 0
            solver_var.value = _value
            if _fixed:
                if v_id not in self._solver_model._wntr_fixed_var_params:
                    self._solver_model._wntr_fixed_var_params[v_id] = param = aml.Param(
                        _value
                    )
                    wntr_expr = solver_var - param
                    self._solver_model._wntr_fixed_var_cons[v_id] = aml.Constraint(
                        wntr_expr
                    )
                    self._needs_updated = True
                else:
                    self._solver_model._wntr_fixed_var_params[v_id].value = _value
            else:
                if v_id in self._solver_model._wntr_fixed_var_params:
                    del self._solver_model._wntr_fixed_var_params[v_id]
                    del self._solver_model._wntr_fixed_var_cons[v_id]
                    self._needs_updated = True

    def update_params(self):
        for p_id, solver_p in self._pyomo_param_to_solver_param_map.items():
            p = self._params[p_id]
            solver_p.value = p.value

    def _set_objective(self, obj):
        raise NotImplementedError(
            f"WNTR's newton solver can only solve square problems"
        )

    def load_vars(self, vars_to_load=None):
        if vars_to_load is None:
            vars_to_load = [i[0] for i in self._vars.values()]
        for v in vars_to_load:
            v_id = id(v)
            solver_v = self._pyomo_var_to_solver_var_map[v_id]
            v.value = solver_v.value

    def get_primals(self, vars_to_load=None):
        if vars_to_load is None:
            vars_to_load = [i[0] for i in self._vars.values()]
        res = ComponentMap()
        for v in vars_to_load:
            v_id = id(v)
            solver_v = self._pyomo_var_to_solver_var_map[v_id]
            res[v] = solver_v.value
        return res

    def _add_sos_constraints(self, cons):
        if len(cons) > 0:
            raise NotImplementedError(
                f"WNTR's newton solver does not support SOS constraints"
            )

    def _remove_sos_constraints(self, cons):
        if len(cons) > 0:
            raise NotImplementedError(
                f"WNTR's newton solver does not support SOS constraints"
            )


def _handle_product_expression(node, values):
    arg1, arg2 = values
    return arg1 * arg2


def _handle_sum_expression(node, values):
    return sum(values)


def _handle_division_expression(node, values):
    arg1, arg2 = values
    return arg1 / arg2


def _handle_pow_expression(node, values):
    arg1, arg2 = values
    return arg1**arg2


def _handle_negation_expression(node, values):
    return -values[0]


def _handle_exp_expression(node, values):
    return wntr.sim.aml.exp(values[0])


def _handle_log_expression(node, values):
    return wntr.sim.aml.log(values[0])


def _handle_sin_expression(node, values):
    return wntr.sim.aml.sin(values[0])


def _handle_cos_expression(node, values):
    return wntr.sim.aml.cos(values[0])


def _handle_tan_expression(node, values):
    return wntr.sim.aml.tan(values[0])


def _handle_asin_expression(node, values):
    return wntr.sim.aml.asin(values[0])


def _handle_acos_expression(node, values):
    return wntr.sim.aml.acos(values[0])


def _handle_atan_expression(node, values):
    return wntr.sim.aml.atan(values[0])


def _handle_sqrt_expression(node, values):
    return (values[0]) ** 0.5


def _handle_abs_expression(node, values):
    return wntr.sim.aml.abs(values[0])


_unary_handler_map = dict()
_unary_handler_map['exp'] = _handle_exp_expression
_unary_handler_map['log'] = _handle_log_expression
_unary_handler_map['sin'] = _handle_sin_expression
_unary_handler_map['cos'] = _handle_cos_expression
_unary_handler_map['tan'] = _handle_tan_expression
_unary_handler_map['asin'] = _handle_asin_expression
_unary_handler_map['acos'] = _handle_acos_expression
_unary_handler_map['atan'] = _handle_atan_expression
_unary_handler_map['sqrt'] = _handle_sqrt_expression
_unary_handler_map['abs'] = _handle_abs_expression


def _handle_unary_function_expression(node, values):
    if node.getname() in _unary_handler_map:
        return _unary_handler_map[node.getname()](node, values)
    else:
        raise NotImplementedError(
            f'Unrecognized unary function expression: {node.getname()}'
        )


_handler_map = dict()
_handler_map[ProductExpression] = _handle_product_expression
_handler_map[DivisionExpression] = _handle_division_expression
_handler_map[PowExpression] = _handle_pow_expression
_handler_map[SumExpression] = _handle_sum_expression
_handler_map[MonomialTermExpression] = _handle_product_expression
_handler_map[NegationExpression] = _handle_negation_expression
_handler_map[UnaryFunctionExpression] = _handle_unary_function_expression
_handler_map[LinearExpression] = _handle_sum_expression
_handler_map[AbsExpression] = _handle_abs_expression
_handler_map[NPV_ProductExpression] = _handle_product_expression
_handler_map[NPV_DivisionExpression] = _handle_division_expression
_handler_map[NPV_PowExpression] = _handle_pow_expression
_handler_map[NPV_SumExpression] = _handle_sum_expression
_handler_map[NPV_NegationExpression] = _handle_negation_expression
_handler_map[NPV_UnaryFunctionExpression] = _handle_unary_function_expression
_handler_map[NPV_AbsExpression] = _handle_abs_expression


class PyomoToWntrVisitor(ExpressionValueVisitor):
    def __init__(self, var_map, param_map):
        self.var_map = var_map
        self.param_map = param_map

    def visit(self, node, values):
        if node.__class__ in _handler_map:
            return _handler_map[node.__class__](node, values)
        else:
            raise NotImplementedError(f'Unrecognized expression type: {node.__class__}')

    def visiting_potential_leaf(self, node):
        if node.__class__ in native_numeric_types:
            return True, node

        if node.is_variable_type():
            return True, self.var_map[id(node)]

        if node.is_parameter_type():
            return True, self.param_map[id(node)]

        return False, None
