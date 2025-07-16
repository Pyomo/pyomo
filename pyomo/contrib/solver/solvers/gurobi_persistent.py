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
import logging
import math
from typing import List, Optional
from collections.abc import Iterable

from pyomo.common.collections import ComponentSet, ComponentMap, OrderedSet
from pyomo.common.dependencies import attempt_import
from pyomo.common.errors import ApplicationError
from pyomo.common.tee import capture_output, TeeStream
from pyomo.common.timing import HierarchicalTimer
from pyomo.common.shutdown import python_is_shutting_down
from pyomo.core.kernel.objective import minimize, maximize
from pyomo.core.base import SymbolMap, NumericLabeler, TextLabeler
from pyomo.core.base.var import VarData
from pyomo.core.base.constraint import ConstraintData
from pyomo.core.base.sos import SOSConstraintData
from pyomo.core.base.param import ParamData
from pyomo.core.expr.numvalue import value, is_constant, is_fixed, native_numeric_types
from pyomo.repn import generate_standard_repn
from pyomo.core.expr.numeric_expr import NPV_MaxExpression, NPV_MinExpression
from pyomo.contrib.solver.common.base import PersistentSolverBase, Availability
from pyomo.contrib.solver.common.results import (
    Results,
    TerminationCondition,
    SolutionStatus,
)
from pyomo.contrib.solver.common.config import PersistentBranchAndBoundConfig
from pyomo.contrib.solver.solvers.gurobi_direct import (
    GurobiConfigMixin,
    GurobiSolverMixin,
)
from pyomo.contrib.solver.common.util import (
    NoFeasibleSolutionError,
    NoOptimalSolutionError,
    NoDualsError,
    NoReducedCostsError,
    NoSolutionError,
    IncompatibleModelError,
)
from pyomo.contrib.solver.common.persistent import (
    PersistentSolverUtils,
    PersistentSolverMixin,
)
from pyomo.contrib.solver.common.solution_loader import PersistentSolutionLoader
from pyomo.core.staleflag import StaleFlagManager


logger = logging.getLogger(__name__)


def _import_gurobipy():
    try:
        import gurobipy
    except ImportError:
        GurobiPersistent._available = Availability.NotFound
        raise
    if gurobipy.GRB.VERSION_MAJOR < 7:
        GurobiPersistent._available = Availability.BadVersion
        raise ImportError('The Persistent Gurobi interface requires gurobipy>=7.0.0')
    return gurobipy


gurobipy, gurobipy_available = attempt_import('gurobipy', importer=_import_gurobipy)


class GurobiConfig(PersistentBranchAndBoundConfig, GurobiConfigMixin):
    def __init__(
        self,
        description=None,
        doc=None,
        implicit=False,
        implicit_domain=None,
        visibility=0,
    ):
        PersistentBranchAndBoundConfig.__init__(
            self,
            description=description,
            doc=doc,
            implicit=implicit,
            implicit_domain=implicit_domain,
            visibility=visibility,
        )
        GurobiConfigMixin.__init__(self)


class GurobiSolutionLoader(PersistentSolutionLoader):
    def load_vars(self, vars_to_load=None, solution_number=0):
        self._assert_solution_still_valid()
        self._solver._load_vars(
            vars_to_load=vars_to_load, solution_number=solution_number
        )

    def get_primals(self, vars_to_load=None, solution_number=0):
        self._assert_solution_still_valid()
        return self._solver._get_primals(
            vars_to_load=vars_to_load, solution_number=solution_number
        )


class _MutableLowerBound:
    def __init__(self, expr):
        self.var = None
        self.expr = expr

    def update(self):
        self.var.setAttr('lb', value(self.expr))


class _MutableUpperBound:
    def __init__(self, expr):
        self.var = None
        self.expr = expr

    def update(self):
        self.var.setAttr('ub', value(self.expr))


class _MutableLinearCoefficient:
    def __init__(self):
        self.expr = None
        self.var = None
        self.con = None
        self.gurobi_model = None

    def update(self):
        self.gurobi_model.chgCoeff(self.con, self.var, value(self.expr))


class _MutableRangeConstant:
    def __init__(self):
        self.lhs_expr = None
        self.rhs_expr = None
        self.con = None
        self.slack_name = None
        self.gurobi_model = None

    def update(self):
        rhs_val = value(self.rhs_expr)
        lhs_val = value(self.lhs_expr)
        self.con.rhs = rhs_val
        slack = self.gurobi_model.getVarByName(self.slack_name)
        slack.ub = rhs_val - lhs_val


class _MutableConstant:
    def __init__(self):
        self.expr = None
        self.con = None

    def update(self):
        self.con.rhs = value(self.expr)


class _MutableQuadraticConstraint:
    def __init__(
        self, gurobi_model, gurobi_con, constant, linear_coefs, quadratic_coefs
    ):
        self.con = gurobi_con
        self.gurobi_model = gurobi_model
        self.constant = constant
        self.last_constant_value = value(self.constant.expr)
        self.linear_coefs = linear_coefs
        self.last_linear_coef_values = [value(i.expr) for i in self.linear_coefs]
        self.quadratic_coefs = quadratic_coefs
        self.last_quadratic_coef_values = [value(i.expr) for i in self.quadratic_coefs]

    def get_updated_expression(self):
        gurobi_expr = self.gurobi_model.getQCRow(self.con)
        for ndx, coef in enumerate(self.linear_coefs):
            current_coef_value = value(coef.expr)
            incremental_coef_value = (
                current_coef_value - self.last_linear_coef_values[ndx]
            )
            gurobi_expr += incremental_coef_value * coef.var
            self.last_linear_coef_values[ndx] = current_coef_value
        for ndx, coef in enumerate(self.quadratic_coefs):
            current_coef_value = value(coef.expr)
            incremental_coef_value = (
                current_coef_value - self.last_quadratic_coef_values[ndx]
            )
            gurobi_expr += incremental_coef_value * coef.var1 * coef.var2
            self.last_quadratic_coef_values[ndx] = current_coef_value
        return gurobi_expr

    def get_updated_rhs(self):
        return value(self.constant.expr)


class _MutableObjective:
    def __init__(self, gurobi_model, constant, linear_coefs, quadratic_coefs):
        self.gurobi_model = gurobi_model
        self.constant = constant
        self.linear_coefs = linear_coefs
        self.quadratic_coefs = quadratic_coefs
        self.last_quadratic_coef_values = [value(i.expr) for i in self.quadratic_coefs]

    def get_updated_expression(self):
        for ndx, coef in enumerate(self.linear_coefs):
            coef.var.obj = value(coef.expr)
        self.gurobi_model.ObjCon = value(self.constant.expr)

        gurobi_expr = None
        for ndx, coef in enumerate(self.quadratic_coefs):
            if value(coef.expr) != self.last_quadratic_coef_values[ndx]:
                if gurobi_expr is None:
                    self.gurobi_model.update()
                    gurobi_expr = self.gurobi_model.getObjective()
                current_coef_value = value(coef.expr)
                incremental_coef_value = (
                    current_coef_value - self.last_quadratic_coef_values[ndx]
                )
                gurobi_expr += incremental_coef_value * coef.var1 * coef.var2
                self.last_quadratic_coef_values[ndx] = current_coef_value
        return gurobi_expr


class _MutableQuadraticCoefficient:
    def __init__(self):
        self.expr = None
        self.var1 = None
        self.var2 = None


class GurobiPersistent(
    GurobiSolverMixin,
    PersistentSolverMixin,
    PersistentSolverUtils,
    PersistentSolverBase,
):
    """
    Interface to Gurobi persistent
    """

    CONFIG = GurobiConfig()
    _gurobipy_available = gurobipy_available

    def __init__(self, **kwds):
        treat_fixed_vars_as_params = kwds.pop('treat_fixed_vars_as_params', True)
        PersistentSolverBase.__init__(self, **kwds)
        PersistentSolverUtils.__init__(
            self, treat_fixed_vars_as_params=treat_fixed_vars_as_params
        )
        self._register_env_client()
        self._solver_model = None
        self._symbol_map = SymbolMap()
        self._labeler = None
        self._pyomo_var_to_solver_var_map = {}
        self._pyomo_con_to_solver_con_map = {}
        self._solver_con_to_pyomo_con_map = {}
        self._pyomo_sos_to_solver_sos_map = {}
        self._range_constraints = OrderedSet()
        self._mutable_helpers = {}
        self._mutable_bounds = {}
        self._mutable_quadratic_helpers = {}
        self._mutable_objective = None
        self._needs_updated = True
        self._callback = None
        self._callback_func = None
        self._constraints_added_since_update = OrderedSet()
        self._vars_added_since_update = ComponentSet()
        self._last_results_object: Optional[Results] = None

    def release_license(self):
        self._reinit()
        self.__class__.release_license()

    def __del__(self):
        if not python_is_shutting_down():
            self._release_env_client()

    @property
    def symbol_map(self):
        return self._symbol_map

    def _solve(self):
        config = self._active_config
        timer = config.timer
        ostreams = [io.StringIO()] + config.tee

        with capture_output(TeeStream(*ostreams), capture_fd=False):
            options = config.solver_options

            self._solver_model.setParam('LogToConsole', 1)

            if config.threads is not None:
                self._solver_model.setParam('Threads', config.threads)
            if config.time_limit is not None:
                self._solver_model.setParam('TimeLimit', config.time_limit)
            if config.rel_gap is not None:
                self._solver_model.setParam('MIPGap', config.rel_gap)
            if config.abs_gap is not None:
                self._solver_model.setParam('MIPGapAbs', config.abs_gap)

            if config.use_mipstart:
                for (
                    pyomo_var_id,
                    gurobi_var,
                ) in self._pyomo_var_to_solver_var_map.items():
                    pyomo_var = self._vars[pyomo_var_id][0]
                    if pyomo_var.is_integer() and pyomo_var.value is not None:
                        self.set_var_attr(pyomo_var, 'Start', pyomo_var.value)

            for key, option in options.items():
                self._solver_model.setParam(key, option)

            timer.start('optimize')
            self._solver_model.optimize(self._callback)
            timer.stop('optimize')

        self._needs_updated = False
        res = self._postsolve(timer)
        res.solver_config = config
        res.solver_name = 'Gurobi'
        res.solver_version = self.version()
        res.solver_log = ostreams[0].getvalue()
        return res

    def _process_domain_and_bounds(
        self, var, var_id, mutable_lbs, mutable_ubs, ndx, gurobipy_var
    ):
        _v, _lb, _ub, _fixed, _domain_interval, _value = self._vars[id(var)]
        lb, ub, step = _domain_interval
        if lb is None:
            lb = -gurobipy.GRB.INFINITY
        if ub is None:
            ub = gurobipy.GRB.INFINITY
        if step == 0:
            vtype = gurobipy.GRB.CONTINUOUS
        elif step == 1:
            if lb == 0 and ub == 1:
                vtype = gurobipy.GRB.BINARY
            else:
                vtype = gurobipy.GRB.INTEGER
        else:
            raise ValueError(
                f'Unrecognized domain step: {step} (should be either 0 or 1)'
            )
        if _fixed:
            lb = _value
            ub = _value
        else:
            if _lb is not None:
                if not is_constant(_lb):
                    mutable_bound = _MutableLowerBound(NPV_MaxExpression((_lb, lb)))
                    if gurobipy_var is None:
                        mutable_lbs[ndx] = mutable_bound
                    else:
                        mutable_bound.var = gurobipy_var
                    self._mutable_bounds[var_id, 'lb'] = (var, mutable_bound)
                lb = max(value(_lb), lb)
            if _ub is not None:
                if not is_constant(_ub):
                    mutable_bound = _MutableUpperBound(NPV_MinExpression((_ub, ub)))
                    if gurobipy_var is None:
                        mutable_ubs[ndx] = mutable_bound
                    else:
                        mutable_bound.var = gurobipy_var
                    self._mutable_bounds[var_id, 'ub'] = (var, mutable_bound)
                ub = min(value(_ub), ub)

        return lb, ub, vtype

    def _add_variables(self, variables: List[VarData]):
        var_names = []
        vtypes = []
        lbs = []
        ubs = []
        mutable_lbs = {}
        mutable_ubs = {}
        for ndx, var in enumerate(variables):
            varname = self._symbol_map.getSymbol(var, self._labeler)
            lb, ub, vtype = self._process_domain_and_bounds(
                var, id(var), mutable_lbs, mutable_ubs, ndx, None
            )
            var_names.append(varname)
            vtypes.append(vtype)
            lbs.append(lb)
            ubs.append(ub)

        gurobi_vars = self._solver_model.addVars(
            len(variables), lb=lbs, ub=ubs, vtype=vtypes, name=var_names
        )

        for ndx, pyomo_var in enumerate(variables):
            gurobi_var = gurobi_vars[ndx]
            self._pyomo_var_to_solver_var_map[id(pyomo_var)] = gurobi_var
        for ndx, mutable_bound in mutable_lbs.items():
            mutable_bound.var = gurobi_vars[ndx]
        for ndx, mutable_bound in mutable_ubs.items():
            mutable_bound.var = gurobi_vars[ndx]
        self._vars_added_since_update.update(variables)
        self._needs_updated = True

    def _add_parameters(self, params: List[ParamData]):
        pass

    def _reinit(self):
        saved_config = self.config
        saved_tmp_config = self._active_config
        self.__init__(treat_fixed_vars_as_params=self._treat_fixed_vars_as_params)
        # Note that __init__ registers a new env client, so we need to
        # release it here:
        self._release_env_client()
        self.config = saved_config
        self._active_config = saved_tmp_config

    def set_instance(self, model):
        if self._last_results_object is not None:
            self._last_results_object.solution_loader.invalidate()
        if not self.available():
            c = self.__class__
            raise ApplicationError(
                f'Solver {c.__module__}.{c.__qualname__} is not available '
                f'({self.available()}).'
            )
        self._reinit()
        self._model = model

        if self.config.symbolic_solver_labels:
            self._labeler = TextLabeler()
        else:
            self._labeler = NumericLabeler('x')

        self._solver_model = gurobipy.Model(name=model.name or '', env=self.env())

        self.add_block(model)
        if self._objective is None:
            self.set_objective(None)

    def _get_expr_from_pyomo_expr(self, expr):
        mutable_linear_coefficients = []
        mutable_quadratic_coefficients = []
        repn = generate_standard_repn(expr, quadratic=True, compute_values=False)

        degree = repn.polynomial_degree()
        if (degree is None) or (degree > 2):
            raise IncompatibleModelError(
                f'GurobiAuto does not support expressions of degree {degree}.'
            )

        if len(repn.linear_vars) > 0:
            linear_coef_vals = []
            for ndx, coef in enumerate(repn.linear_coefs):
                if not is_constant(coef):
                    mutable_linear_coefficient = _MutableLinearCoefficient()
                    mutable_linear_coefficient.expr = coef
                    mutable_linear_coefficient.var = self._pyomo_var_to_solver_var_map[
                        id(repn.linear_vars[ndx])
                    ]
                    mutable_linear_coefficients.append(mutable_linear_coefficient)
                linear_coef_vals.append(value(coef))
            new_expr = gurobipy.LinExpr(
                linear_coef_vals,
                [self._pyomo_var_to_solver_var_map[id(i)] for i in repn.linear_vars],
            )
        else:
            new_expr = 0.0

        for ndx, v in enumerate(repn.quadratic_vars):
            x, y = v
            gurobi_x = self._pyomo_var_to_solver_var_map[id(x)]
            gurobi_y = self._pyomo_var_to_solver_var_map[id(y)]
            coef = repn.quadratic_coefs[ndx]
            if not is_constant(coef):
                mutable_quadratic_coefficient = _MutableQuadraticCoefficient()
                mutable_quadratic_coefficient.expr = coef
                mutable_quadratic_coefficient.var1 = gurobi_x
                mutable_quadratic_coefficient.var2 = gurobi_y
                mutable_quadratic_coefficients.append(mutable_quadratic_coefficient)
            coef_val = value(coef)
            new_expr += coef_val * gurobi_x * gurobi_y

        return (
            new_expr,
            repn.constant,
            mutable_linear_coefficients,
            mutable_quadratic_coefficients,
        )

    def _add_constraints(self, cons: List[ConstraintData]):
        for con in cons:
            conname = self._symbol_map.getSymbol(con, self._labeler)
            (
                gurobi_expr,
                repn_constant,
                mutable_linear_coefficients,
                mutable_quadratic_coefficients,
            ) = self._get_expr_from_pyomo_expr(con.body)

            if (
                gurobi_expr.__class__ in {gurobipy.LinExpr, gurobipy.Var}
                or gurobi_expr.__class__ in native_numeric_types
            ):
                if con.equality:
                    rhs_expr = con.lower - repn_constant
                    rhs_val = value(rhs_expr)
                    gurobipy_con = self._solver_model.addLConstr(
                        gurobi_expr, gurobipy.GRB.EQUAL, rhs_val, name=conname
                    )
                    if not is_constant(rhs_expr):
                        mutable_constant = _MutableConstant()
                        mutable_constant.expr = rhs_expr
                        mutable_constant.con = gurobipy_con
                        self._mutable_helpers[con] = [mutable_constant]
                elif con.has_lb() and con.has_ub():
                    lhs_expr = con.lower - repn_constant
                    rhs_expr = con.upper - repn_constant
                    lhs_val = value(lhs_expr)
                    rhs_val = value(rhs_expr)
                    gurobipy_con = self._solver_model.addRange(
                        gurobi_expr, lhs_val, rhs_val, name=conname
                    )
                    self._range_constraints.add(con)
                    if not is_constant(lhs_expr) or not is_constant(rhs_expr):
                        mutable_range_constant = _MutableRangeConstant()
                        mutable_range_constant.lhs_expr = lhs_expr
                        mutable_range_constant.rhs_expr = rhs_expr
                        mutable_range_constant.con = gurobipy_con
                        mutable_range_constant.slack_name = 'Rg' + conname
                        mutable_range_constant.gurobi_model = self._solver_model
                        self._mutable_helpers[con] = [mutable_range_constant]
                elif con.has_lb():
                    rhs_expr = con.lower - repn_constant
                    rhs_val = value(rhs_expr)
                    gurobipy_con = self._solver_model.addLConstr(
                        gurobi_expr, gurobipy.GRB.GREATER_EQUAL, rhs_val, name=conname
                    )
                    if not is_constant(rhs_expr):
                        mutable_constant = _MutableConstant()
                        mutable_constant.expr = rhs_expr
                        mutable_constant.con = gurobipy_con
                        self._mutable_helpers[con] = [mutable_constant]
                elif con.has_ub():
                    rhs_expr = con.upper - repn_constant
                    rhs_val = value(rhs_expr)
                    gurobipy_con = self._solver_model.addLConstr(
                        gurobi_expr, gurobipy.GRB.LESS_EQUAL, rhs_val, name=conname
                    )
                    if not is_constant(rhs_expr):
                        mutable_constant = _MutableConstant()
                        mutable_constant.expr = rhs_expr
                        mutable_constant.con = gurobipy_con
                        self._mutable_helpers[con] = [mutable_constant]
                else:
                    raise ValueError(
                        "Constraint does not have a lower "
                        f"or an upper bound: {con} \n"
                    )
                for tmp in mutable_linear_coefficients:
                    tmp.con = gurobipy_con
                    tmp.gurobi_model = self._solver_model
                if len(mutable_linear_coefficients) > 0:
                    if con not in self._mutable_helpers:
                        self._mutable_helpers[con] = mutable_linear_coefficients
                    else:
                        self._mutable_helpers[con].extend(mutable_linear_coefficients)
            elif gurobi_expr.__class__ is gurobipy.QuadExpr:
                if con.equality:
                    rhs_expr = con.lower - repn_constant
                    rhs_val = value(rhs_expr)
                    gurobipy_con = self._solver_model.addQConstr(
                        gurobi_expr, gurobipy.GRB.EQUAL, rhs_val, name=conname
                    )
                elif con.has_lb() and con.has_ub():
                    raise NotImplementedError(
                        'Quadratic range constraints are not supported'
                    )
                elif con.has_lb():
                    rhs_expr = con.lower - repn_constant
                    rhs_val = value(rhs_expr)
                    gurobipy_con = self._solver_model.addQConstr(
                        gurobi_expr, gurobipy.GRB.GREATER_EQUAL, rhs_val, name=conname
                    )
                elif con.has_ub():
                    rhs_expr = con.upper - repn_constant
                    rhs_val = value(rhs_expr)
                    gurobipy_con = self._solver_model.addQConstr(
                        gurobi_expr, gurobipy.GRB.LESS_EQUAL, rhs_val, name=conname
                    )
                else:
                    raise ValueError(
                        "Constraint does not have a lower "
                        f"or an upper bound: {con} \n"
                    )
                if (
                    len(mutable_linear_coefficients) > 0
                    or len(mutable_quadratic_coefficients) > 0
                    or not is_constant(repn_constant)
                ):
                    mutable_constant = _MutableConstant()
                    mutable_constant.expr = rhs_expr
                    mutable_quadratic_constraint = _MutableQuadraticConstraint(
                        self._solver_model,
                        gurobipy_con,
                        mutable_constant,
                        mutable_linear_coefficients,
                        mutable_quadratic_coefficients,
                    )
                    self._mutable_quadratic_helpers[con] = mutable_quadratic_constraint
            else:
                raise ValueError(
                    f'Unrecognized Gurobi expression type: {str(gurobi_expr.__class__)}'
                )

            self._pyomo_con_to_solver_con_map[con] = gurobipy_con
            self._solver_con_to_pyomo_con_map[id(gurobipy_con)] = con
        self._constraints_added_since_update.update(cons)
        self._needs_updated = True

    def _add_sos_constraints(self, cons: List[SOSConstraintData]):
        for con in cons:
            conname = self._symbol_map.getSymbol(con, self._labeler)
            level = con.level
            if level == 1:
                sos_type = gurobipy.GRB.SOS_TYPE1
            elif level == 2:
                sos_type = gurobipy.GRB.SOS_TYPE2
            else:
                raise ValueError(
                    f"Solver does not support SOS level {level} constraints"
                )

            gurobi_vars = []
            weights = []

            for v, w in con.get_items():
                v_id = id(v)
                gurobi_vars.append(self._pyomo_var_to_solver_var_map[v_id])
                weights.append(w)

            gurobipy_con = self._solver_model.addSOS(sos_type, gurobi_vars, weights)
            self._pyomo_sos_to_solver_sos_map[con] = gurobipy_con
        self._constraints_added_since_update.update(cons)
        self._needs_updated = True

    def _remove_constraints(self, cons: List[ConstraintData]):
        for con in cons:
            if con in self._constraints_added_since_update:
                self._update_gurobi_model()
            solver_con = self._pyomo_con_to_solver_con_map[con]
            self._solver_model.remove(solver_con)
            self._symbol_map.removeSymbol(con)
            del self._pyomo_con_to_solver_con_map[con]
            del self._solver_con_to_pyomo_con_map[id(solver_con)]
            self._range_constraints.discard(con)
            self._mutable_helpers.pop(con, None)
            self._mutable_quadratic_helpers.pop(con, None)
        self._needs_updated = True

    def _remove_sos_constraints(self, cons: List[SOSConstraintData]):
        for con in cons:
            if con in self._constraints_added_since_update:
                self._update_gurobi_model()
            solver_sos_con = self._pyomo_sos_to_solver_sos_map[con]
            self._solver_model.remove(solver_sos_con)
            self._symbol_map.removeSymbol(con)
            del self._pyomo_sos_to_solver_sos_map[con]
        self._needs_updated = True

    def _remove_variables(self, variables: List[VarData]):
        for var in variables:
            v_id = id(var)
            if var in self._vars_added_since_update:
                self._update_gurobi_model()
            solver_var = self._pyomo_var_to_solver_var_map[v_id]
            self._solver_model.remove(solver_var)
            self._symbol_map.removeSymbol(var)
            del self._pyomo_var_to_solver_var_map[v_id]
            self._mutable_bounds.pop(v_id, None)
        self._needs_updated = True

    def _remove_parameters(self, params: List[ParamData]):
        pass

    def _update_variables(self, variables: List[VarData]):
        for var in variables:
            var_id = id(var)
            if var_id not in self._pyomo_var_to_solver_var_map:
                raise ValueError(
                    f'The Var provided to update_var needs to be added first: {var}'
                )
            self._mutable_bounds.pop((var_id, 'lb'), None)
            self._mutable_bounds.pop((var_id, 'ub'), None)
            gurobipy_var = self._pyomo_var_to_solver_var_map[var_id]
            lb, ub, vtype = self._process_domain_and_bounds(
                var, var_id, None, None, None, gurobipy_var
            )
            gurobipy_var.setAttr('lb', lb)
            gurobipy_var.setAttr('ub', ub)
            gurobipy_var.setAttr('vtype', vtype)
        self._needs_updated = True

    def update_parameters(self):
        for con, helpers in self._mutable_helpers.items():
            for helper in helpers:
                helper.update()
        for k, (v, helper) in self._mutable_bounds.items():
            helper.update()

        for con, helper in self._mutable_quadratic_helpers.items():
            if con in self._constraints_added_since_update:
                self._update_gurobi_model()
            gurobi_con = helper.con
            new_gurobi_expr = helper.get_updated_expression()
            new_rhs = helper.get_updated_rhs()
            new_sense = gurobi_con.qcsense
            pyomo_con = self._solver_con_to_pyomo_con_map[id(gurobi_con)]
            name = self._symbol_map.getSymbol(pyomo_con, self._labeler)
            self._solver_model.remove(gurobi_con)
            new_con = self._solver_model.addQConstr(
                new_gurobi_expr, new_sense, new_rhs, name=name
            )
            self._pyomo_con_to_solver_con_map[id(pyomo_con)] = new_con
            del self._solver_con_to_pyomo_con_map[id(gurobi_con)]
            self._solver_con_to_pyomo_con_map[id(new_con)] = pyomo_con
            helper.con = new_con
            self._constraints_added_since_update.add(con)

        helper = self._mutable_objective
        pyomo_obj = self._objective
        new_gurobi_expr = helper.get_updated_expression()
        if new_gurobi_expr is not None:
            if pyomo_obj.sense == minimize:
                sense = gurobipy.GRB.MINIMIZE
            else:
                sense = gurobipy.GRB.MAXIMIZE
            self._solver_model.setObjective(new_gurobi_expr, sense=sense)

    def _set_objective(self, obj):
        if obj is None:
            sense = gurobipy.GRB.MINIMIZE
            gurobi_expr = 0
            repn_constant = 0
            mutable_linear_coefficients = []
            mutable_quadratic_coefficients = []
        else:
            if obj.sense == minimize:
                sense = gurobipy.GRB.MINIMIZE
            elif obj.sense == maximize:
                sense = gurobipy.GRB.MAXIMIZE
            else:
                raise ValueError(f'Objective sense is not recognized: {obj.sense}')

            (
                gurobi_expr,
                repn_constant,
                mutable_linear_coefficients,
                mutable_quadratic_coefficients,
            ) = self._get_expr_from_pyomo_expr(obj.expr)

        mutable_constant = _MutableConstant()
        mutable_constant.expr = repn_constant
        mutable_objective = _MutableObjective(
            self._solver_model,
            mutable_constant,
            mutable_linear_coefficients,
            mutable_quadratic_coefficients,
        )
        self._mutable_objective = mutable_objective

        # These two lines are needed as a workaround
        # see PR #2454
        self._solver_model.setObjective(0)
        self._solver_model.update()

        self._solver_model.setObjective(gurobi_expr + value(repn_constant), sense=sense)
        self._needs_updated = True

    def _postsolve(self, timer: HierarchicalTimer):
        config = self._active_config

        gprob = self._solver_model
        grb = gurobipy.GRB
        status = gprob.Status

        results = Results()
        results.solution_loader = GurobiSolutionLoader(self)
        results.timing_info.gurobi_time = gprob.Runtime

        if gprob.SolCount > 0:
            if status == grb.OPTIMAL:
                results.solution_status = SolutionStatus.optimal
            else:
                results.solution_status = SolutionStatus.feasible
        else:
            results.solution_status = SolutionStatus.noSolution

        if status == grb.LOADED:  # problem is loaded, but no solution
            results.termination_condition = TerminationCondition.unknown
        elif status == grb.OPTIMAL:  # optimal
            results.termination_condition = (
                TerminationCondition.convergenceCriteriaSatisfied
            )
        elif status == grb.INFEASIBLE:
            results.termination_condition = TerminationCondition.provenInfeasible
        elif status == grb.INF_OR_UNBD:
            results.termination_condition = TerminationCondition.infeasibleOrUnbounded
        elif status == grb.UNBOUNDED:
            results.termination_condition = TerminationCondition.unbounded
        elif status == grb.CUTOFF:
            results.termination_condition = TerminationCondition.objectiveLimit
        elif status == grb.ITERATION_LIMIT:
            results.termination_condition = TerminationCondition.iterationLimit
        elif status == grb.NODE_LIMIT:
            results.termination_condition = TerminationCondition.iterationLimit
        elif status == grb.TIME_LIMIT:
            results.termination_condition = TerminationCondition.maxTimeLimit
        elif status == grb.SOLUTION_LIMIT:
            results.termination_condition = TerminationCondition.unknown
        elif status == grb.INTERRUPTED:
            results.termination_condition = TerminationCondition.interrupted
        elif status == grb.NUMERIC:
            results.termination_condition = TerminationCondition.unknown
        elif status == grb.SUBOPTIMAL:
            results.termination_condition = TerminationCondition.unknown
        elif status == grb.USER_OBJ_LIMIT:
            results.termination_condition = TerminationCondition.objectiveLimit
        else:
            results.termination_condition = TerminationCondition.unknown

        if (
            results.termination_condition
            != TerminationCondition.convergenceCriteriaSatisfied
            and config.raise_exception_on_nonoptimal_result
        ):
            raise NoOptimalSolutionError()

        results.incumbent_objective = None
        results.objective_bound = None
        if self._objective is not None:
            try:
                results.incumbent_objective = gprob.ObjVal
            except (gurobipy.GurobiError, AttributeError):
                results.incumbent_objective = None
            try:
                results.objective_bound = gprob.ObjBound
            except (gurobipy.GurobiError, AttributeError):
                if self._objective.sense == minimize:
                    results.objective_bound = -math.inf
                else:
                    results.objective_bound = math.inf

            if results.incumbent_objective is not None and not math.isfinite(
                results.incumbent_objective
            ):
                results.incumbent_objective = None

        results.iteration_count = gprob.getAttr('IterCount')

        timer.start('load solution')
        if config.load_solutions:
            if gprob.SolCount > 0:
                self._load_vars()
            else:
                raise NoFeasibleSolutionError()
        timer.stop('load solution')

        return results

    def _load_suboptimal_mip_solution(self, vars_to_load, solution_number):
        if (
            self.get_model_attr('NumIntVars') == 0
            and self.get_model_attr('NumBinVars') == 0
        ):
            raise ValueError(
                'Cannot obtain suboptimal solutions for a continuous model'
            )
        var_map = self._pyomo_var_to_solver_var_map
        ref_vars = self._referenced_variables
        original_solution_number = self.get_gurobi_param_info('SolutionNumber')[2]
        self.set_gurobi_param('SolutionNumber', solution_number)
        gurobi_vars_to_load = [var_map[pyomo_var] for pyomo_var in vars_to_load]
        vals = self._solver_model.getAttr("Xn", gurobi_vars_to_load)
        res = ComponentMap()
        for var_id, val in zip(vars_to_load, vals):
            using_cons, using_sos, using_obj = ref_vars[var_id]
            if using_cons or using_sos or (using_obj is not None):
                res[self._vars[var_id][0]] = val
        self.set_gurobi_param('SolutionNumber', original_solution_number)
        return res

    def _load_vars(self, vars_to_load=None, solution_number=0):
        for v, val in self._get_primals(
            vars_to_load=vars_to_load, solution_number=solution_number
        ).items():
            v.set_value(val, skip_validation=True)
        StaleFlagManager.mark_all_as_stale(delayed=True)

    def _get_primals(self, vars_to_load=None, solution_number=0):
        if self._needs_updated:
            self._update_gurobi_model()  # this is needed to ensure that solutions cannot be loaded after the model has been changed

        if self._solver_model.SolCount == 0:
            raise NoSolutionError()

        var_map = self._pyomo_var_to_solver_var_map
        ref_vars = self._referenced_variables
        if vars_to_load is None:
            vars_to_load = self._pyomo_var_to_solver_var_map.keys()
        else:
            vars_to_load = [id(v) for v in vars_to_load]

        if solution_number != 0:
            return self._load_suboptimal_mip_solution(
                vars_to_load=vars_to_load, solution_number=solution_number
            )

        gurobi_vars_to_load = [var_map[pyomo_var_id] for pyomo_var_id in vars_to_load]
        vals = self._solver_model.getAttr("X", gurobi_vars_to_load)

        res = ComponentMap()
        for var_id, val in zip(vars_to_load, vals):
            using_cons, using_sos, using_obj = ref_vars[var_id]
            if using_cons or using_sos or (using_obj is not None):
                res[self._vars[var_id][0]] = val
        return res

    def _get_reduced_costs(self, vars_to_load=None):
        if self._needs_updated:
            self._update_gurobi_model()

        if self._solver_model.Status != gurobipy.GRB.OPTIMAL:
            raise NoReducedCostsError()

        var_map = self._pyomo_var_to_solver_var_map
        ref_vars = self._referenced_variables
        res = ComponentMap()
        if vars_to_load is None:
            vars_to_load = self._pyomo_var_to_solver_var_map.keys()
        else:
            vars_to_load = [id(v) for v in vars_to_load]

        gurobi_vars_to_load = [var_map[pyomo_var_id] for pyomo_var_id in vars_to_load]
        vals = self._solver_model.getAttr("Rc", gurobi_vars_to_load)

        for var_id, val in zip(vars_to_load, vals):
            using_cons, using_sos, using_obj = ref_vars[var_id]
            if using_cons or using_sos or (using_obj is not None):
                res[self._vars[var_id][0]] = val

        return res

    def _get_duals(self, cons_to_load=None):
        if self._needs_updated:
            self._update_gurobi_model()

        if self._solver_model.Status != gurobipy.GRB.OPTIMAL:
            raise NoDualsError()

        con_map = self._pyomo_con_to_solver_con_map
        reverse_con_map = self._solver_con_to_pyomo_con_map
        dual = {}

        if cons_to_load is None:
            linear_cons_to_load = self._solver_model.getConstrs()
            quadratic_cons_to_load = self._solver_model.getQConstrs()
        else:
            gurobi_cons_to_load = OrderedSet(
                [con_map[pyomo_con] for pyomo_con in cons_to_load]
            )
            linear_cons_to_load = list(
                gurobi_cons_to_load.intersection(
                    OrderedSet(self._solver_model.getConstrs())
                )
            )
            quadratic_cons_to_load = list(
                gurobi_cons_to_load.intersection(
                    OrderedSet(self._solver_model.getQConstrs())
                )
            )
        linear_vals = self._solver_model.getAttr("Pi", linear_cons_to_load)
        quadratic_vals = self._solver_model.getAttr("QCPi", quadratic_cons_to_load)

        for gurobi_con, val in zip(linear_cons_to_load, linear_vals):
            pyomo_con = reverse_con_map[id(gurobi_con)]
            dual[pyomo_con] = val
        for gurobi_con, val in zip(quadratic_cons_to_load, quadratic_vals):
            pyomo_con = reverse_con_map[id(gurobi_con)]
            dual[pyomo_con] = val

        return dual

    def update(self, timer: HierarchicalTimer = None):
        if self._needs_updated:
            self._update_gurobi_model()
        super().update(timer=timer)
        self._update_gurobi_model()

    def _update_gurobi_model(self):
        self._solver_model.update()
        self._constraints_added_since_update = OrderedSet()
        self._vars_added_since_update = ComponentSet()
        self._needs_updated = False

    def get_model_attr(self, attr):
        """
        Get the value of an attribute on the Gurobi model.

        Parameters
        ----------
        attr: str
            The attribute to get. See Gurobi documentation for descriptions of the attributes.
        """
        if self._needs_updated:
            self._update_gurobi_model()
        return self._solver_model.getAttr(attr)

    def write(self, filename):
        """
        Write the model to a file (e.g., and lp file).

        Parameters
        ----------
        filename: str
            Name of the file to which the model should be written.
        """
        self._solver_model.write(filename)
        self._constraints_added_since_update = OrderedSet()
        self._vars_added_since_update = ComponentSet()
        self._needs_updated = False

    def set_linear_constraint_attr(self, con, attr, val):
        """
        Set the value of an attribute on a gurobi linear constraint.

        Parameters
        ----------
        con: pyomo.core.base.constraint.ConstraintData
            The pyomo constraint for which the corresponding gurobi constraint attribute
            should be modified.
        attr: str
            The attribute to be modified. Options are:
                CBasis
                DStart
                Lazy
        val: any
            See gurobi documentation for acceptable values.
        """
        if attr in {'Sense', 'RHS', 'ConstrName'}:
            raise ValueError(
                f'Linear constraint attr {attr} cannot be set with'
                ' the set_linear_constraint_attr method. Please use'
                ' the remove_constraint and add_constraint methods.'
            )
        self._pyomo_con_to_solver_con_map[con].setAttr(attr, val)
        self._needs_updated = True

    def set_var_attr(self, var, attr, val):
        """
        Set the value of an attribute on a gurobi variable.

        Parameters
        ----------
        var: pyomo.core.base.var.VarData
            The pyomo var for which the corresponding gurobi var attribute
            should be modified.
        attr: str
            The attribute to be modified. Options are:
                Start
                VarHintVal
                VarHintPri
                BranchPriority
                VBasis
                PStart
        val: any
            See gurobi documentation for acceptable values.
        """
        if attr in {'LB', 'UB', 'VType', 'VarName'}:
            raise ValueError(
                f'Var attr {attr} cannot be set with'
                ' the set_var_attr method. Please use'
                ' the update_var method.'
            )
        if attr == 'Obj':
            raise ValueError(
                'Var attr Obj cannot be set with'
                ' the set_var_attr method. Please use'
                ' the set_objective method.'
            )
        self._pyomo_var_to_solver_var_map[id(var)].setAttr(attr, val)
        self._needs_updated = True

    def get_var_attr(self, var, attr):
        """
        Get the value of an attribute on a gurobi var.

        Parameters
        ----------
        var: pyomo.core.base.var.VarData
            The pyomo var for which the corresponding gurobi var attribute
            should be retrieved.
        attr: str
            The attribute to get. See gurobi documentation
        """
        if self._needs_updated:
            self._update_gurobi_model()
        return self._pyomo_var_to_solver_var_map[id(var)].getAttr(attr)

    def get_linear_constraint_attr(self, con, attr):
        """
        Get the value of an attribute on a gurobi linear constraint.

        Parameters
        ----------
        con: pyomo.core.base.constraint.ConstraintData
            The pyomo constraint for which the corresponding gurobi constraint attribute
            should be retrieved.
        attr: str
            The attribute to get. See the Gurobi documentation
        """
        if self._needs_updated:
            self._update_gurobi_model()
        return self._pyomo_con_to_solver_con_map[con].getAttr(attr)

    def get_sos_attr(self, con, attr):
        """
        Get the value of an attribute on a gurobi sos constraint.

        Parameters
        ----------
        con: pyomo.core.base.sos.SOSConstraintData
            The pyomo SOS constraint for which the corresponding gurobi SOS constraint attribute
            should be retrieved.
        attr: str
            The attribute to get. See the Gurobi documentation
        """
        if self._needs_updated:
            self._update_gurobi_model()
        return self._pyomo_sos_to_solver_sos_map[con].getAttr(attr)

    def get_quadratic_constraint_attr(self, con, attr):
        """
        Get the value of an attribute on a gurobi quadratic constraint.

        Parameters
        ----------
        con: pyomo.core.base.constraint.ConstraintData
            The pyomo constraint for which the corresponding gurobi constraint attribute
            should be retrieved.
        attr: str
            The attribute to get. See the Gurobi documentation
        """
        if self._needs_updated:
            self._update_gurobi_model()
        return self._pyomo_con_to_solver_con_map[con].getAttr(attr)

    def set_gurobi_param(self, param, val):
        """
        Set a gurobi parameter.

        Parameters
        ----------
        param: str
            The gurobi parameter to set. Options include any gurobi parameter.
            Please see the Gurobi documentation for options.
        val: any
            The value to set the parameter to. See Gurobi documentation for possible values.
        """
        self._solver_model.setParam(param, val)

    def get_gurobi_param_info(self, param):
        """
        Get information about a gurobi parameter.

        Parameters
        ----------
        param: str
            The gurobi parameter to get info for. See Gurobi documentation for possible options.

        Returns
        -------
        six-tuple containing the parameter name, type, value, minimum value, maximum value, and default value.
        """
        return self._solver_model.getParamInfo(param)

    def _intermediate_callback(self):
        def f(gurobi_model, where):
            self._callback_func(self._model, self, where)

        return f

    def set_callback(self, func=None):
        """
        Specify a callback for gurobi to use.

        Parameters
        ----------
        func: function
            The function to call. The function should have three arguments. The first will be the pyomo model being
            solved. The second will be the GurobiPersistent instance. The third will be an enum member of
            gurobipy.GRB.Callback. This will indicate where in the branch and bound algorithm gurobi is at. For
            example, suppose we want to solve

            .. math::

                min 2*x + y

                s.t.

                    y >= (x-2)**2

                    0 <= x <= 4

                    y >= 0

                    y integer

            as an MILP using extended cutting planes in callbacks.

                >>> from gurobipy import GRB # doctest:+SKIP
                >>> import pyomo.environ as pyo
                >>> from pyomo.core.expr.taylor_series import taylor_series_expansion
                >>> from pyomo.contrib import appsi
                >>>
                >>> m = pyo.ConcreteModel()
                >>> m.x = pyo.Var(bounds=(0, 4))
                >>> m.y = pyo.Var(within=pyo.Integers, bounds=(0, None))
                >>> m.obj = pyo.Objective(expr=2*m.x + m.y)
                >>> m.cons = pyo.ConstraintList()  # for the cutting planes
                >>>
                >>> def _add_cut(xval):
                ...     # a function to generate the cut
                ...     m.x.value = xval
                ...     return m.cons.add(m.y >= taylor_series_expansion((m.x - 2)**2))
                ...
                >>> _c = _add_cut(0)  # start with 2 cuts at the bounds of x
                >>> _c = _add_cut(4)  # this is an arbitrary choice
                >>>
                >>> opt = appsi.solvers.Gurobi()
                >>> opt.config.stream_solver = True
                >>> opt.set_instance(m) # doctest:+SKIP
                >>> opt.gurobi_options['PreCrush'] = 1
                >>> opt.gurobi_options['LazyConstraints'] = 1
                >>>
                >>> def my_callback(cb_m, cb_opt, cb_where):
                ...     if cb_where == GRB.Callback.MIPSOL:
                ...         cb_opt.cbGetSolution(variables=[m.x, m.y])
                ...         if m.y.value < (m.x.value - 2)**2 - 1e-6:
                ...             cb_opt.cbLazy(_add_cut(m.x.value))
                ...
                >>> opt.set_callback(my_callback)
                >>> res = opt.solve(m) # doctest:+SKIP

        """
        if func is not None:
            self._callback_func = func
            self._callback = self._intermediate_callback()
        else:
            self._callback = None
            self._callback_func = None

    def cbCut(self, con):
        """
        Add a cut within a callback.

        Parameters
        ----------
        con: pyomo.core.base.constraint.ConstraintData
            The cut to add
        """
        if not con.active:
            raise ValueError('cbCut expected an active constraint.')

        if is_fixed(con.body):
            raise ValueError('cbCut expected a non-trivial constraint')

        (
            gurobi_expr,
            repn_constant,
            mutable_linear_coefficients,
            mutable_quadratic_coefficients,
        ) = self._get_expr_from_pyomo_expr(con.body)

        if con.has_lb():
            if con.has_ub():
                raise ValueError('Range constraints are not supported in cbCut.')
            if not is_fixed(con.lower):
                raise ValueError(f'Lower bound of constraint {con} is not constant.')
        if con.has_ub():
            if not is_fixed(con.upper):
                raise ValueError(f'Upper bound of constraint {con} is not constant.')

        if con.equality:
            self._solver_model.cbCut(
                lhs=gurobi_expr,
                sense=gurobipy.GRB.EQUAL,
                rhs=value(con.lower - repn_constant),
            )
        elif con.has_lb() and (value(con.lower) > -float('inf')):
            self._solver_model.cbCut(
                lhs=gurobi_expr,
                sense=gurobipy.GRB.GREATER_EQUAL,
                rhs=value(con.lower - repn_constant),
            )
        elif con.has_ub() and (value(con.upper) < float('inf')):
            self._solver_model.cbCut(
                lhs=gurobi_expr,
                sense=gurobipy.GRB.LESS_EQUAL,
                rhs=value(con.upper - repn_constant),
            )
        else:
            raise ValueError(
                f'Constraint does not have a lower or an upper bound {con} \n'
            )

    def cbGet(self, what):
        return self._solver_model.cbGet(what)

    def cbGetNodeRel(self, variables):
        """
        Parameters
        ----------
        variables: Var or iterable of Var
        """
        if not isinstance(variables, Iterable):
            variables = [variables]
        gurobi_vars = [self._pyomo_var_to_solver_var_map[id(i)] for i in variables]
        var_values = self._solver_model.cbGetNodeRel(gurobi_vars)
        for i, v in enumerate(variables):
            v.set_value(var_values[i], skip_validation=True)

    def cbGetSolution(self, variables):
        """
        Parameters
        ----------
        variables: iterable of vars
        """
        if not isinstance(variables, Iterable):
            variables = [variables]
        gurobi_vars = [self._pyomo_var_to_solver_var_map[id(i)] for i in variables]
        var_values = self._solver_model.cbGetSolution(gurobi_vars)
        for i, v in enumerate(variables):
            v.set_value(var_values[i], skip_validation=True)

    def cbLazy(self, con):
        """
        Parameters
        ----------
        con: pyomo.core.base.constraint.ConstraintData
            The lazy constraint to add
        """
        if not con.active:
            raise ValueError('cbLazy expected an active constraint.')

        if is_fixed(con.body):
            raise ValueError('cbLazy expected a non-trivial constraint')

        (
            gurobi_expr,
            repn_constant,
            mutable_linear_coefficients,
            mutable_quadratic_coefficients,
        ) = self._get_expr_from_pyomo_expr(con.body)

        if con.has_lb():
            if con.has_ub():
                raise ValueError('Range constraints are not supported in cbLazy.')
            if not is_fixed(con.lower):
                raise ValueError(f'Lower bound of constraint {con} is not constant.')
        if con.has_ub():
            if not is_fixed(con.upper):
                raise ValueError(f'Upper bound of constraint {con} is not constant.')

        if con.equality:
            self._solver_model.cbLazy(
                lhs=gurobi_expr,
                sense=gurobipy.GRB.EQUAL,
                rhs=value(con.lower - repn_constant),
            )
        elif con.has_lb() and (value(con.lower) > -float('inf')):
            self._solver_model.cbLazy(
                lhs=gurobi_expr,
                sense=gurobipy.GRB.GREATER_EQUAL,
                rhs=value(con.lower - repn_constant),
            )
        elif con.has_ub() and (value(con.upper) < float('inf')):
            self._solver_model.cbLazy(
                lhs=gurobi_expr,
                sense=gurobipy.GRB.LESS_EQUAL,
                rhs=value(con.upper - repn_constant),
            )
        else:
            raise ValueError(
                f'Constraint does not have a lower or an upper bound {con} \n'
            )

    def cbSetSolution(self, variables, solution):
        if not isinstance(variables, Iterable):
            variables = [variables]
        gurobi_vars = [self._pyomo_var_to_solver_var_map[id(i)] for i in variables]
        self._solver_model.cbSetSolution(gurobi_vars, solution)

    def cbUseSolution(self):
        return self._solver_model.cbUseSolution()

    def reset(self):
        self._solver_model.reset()
