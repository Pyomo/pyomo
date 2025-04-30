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

from collections.abc import Iterable
import logging
import math
from typing import List, Dict, Optional
from pyomo.common.collections import ComponentSet, ComponentMap, OrderedSet
from pyomo.common.log import LogStream
from pyomo.common.dependencies import attempt_import
from pyomo.common.errors import PyomoException
from pyomo.common.tee import capture_output, TeeStream
from pyomo.common.timing import HierarchicalTimer
from pyomo.common.shutdown import python_is_shutting_down
from pyomo.common.config import ConfigValue, NonNegativeInt
from pyomo.core.kernel.objective import minimize, maximize
from pyomo.core.base import SymbolMap, NumericLabeler, TextLabeler
from pyomo.core.base.var import Var, VarData
from pyomo.core.base.constraint import ConstraintData
from pyomo.core.base.sos import SOSConstraintData
from pyomo.core.base.param import ParamData
from pyomo.core.expr.numvalue import value, is_constant, is_fixed, native_numeric_types
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
from pyomo.core.staleflag import StaleFlagManager
import sys

logger = logging.getLogger(__name__)


def _import_gurobipy():
    try:
        import gurobipy
    except ImportError:
        Gurobi._available = Gurobi.Availability.NotFound
        raise
    if gurobipy.GRB.VERSION_MAJOR < 7:
        Gurobi._available = Gurobi.Availability.BadVersion
        raise ImportError('The APPSI Gurobi interface requires gurobipy>=7.0.0')
    return gurobipy


gurobipy, gurobipy_available = attempt_import('gurobipy', importer=_import_gurobipy)


class DegreeError(PyomoException):
    pass


class GurobiConfig(MIPSolverConfig):
    def __init__(
        self,
        description=None,
        doc=None,
        implicit=False,
        implicit_domain=None,
        visibility=0,
    ):
        super(GurobiConfig, self).__init__(
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


class GurobiSolutionLoader(PersistentSolutionLoader):
    def load_vars(self, vars_to_load=None, solution_number=0):
        self._assert_solution_still_valid()
        self._solver.load_vars(
            vars_to_load=vars_to_load, solution_number=solution_number
        )

    def get_primals(self, vars_to_load=None, solution_number=0):
        self._assert_solution_still_valid()
        return self._solver.get_primals(
            vars_to_load=vars_to_load, solution_number=solution_number
        )


class GurobiResults(Results):
    def __init__(self, solver):
        super(GurobiResults, self).__init__()
        self.wallclock_time = None
        self.solution_loader = GurobiSolutionLoader(solver=solver)


class _MutableLowerBound(object):
    def __init__(self, expr):
        self.var = None
        self.expr = expr

    def update(self):
        self.var.setAttr('lb', value(self.expr))


class _MutableUpperBound(object):
    def __init__(self, expr):
        self.var = None
        self.expr = expr

    def update(self):
        self.var.setAttr('ub', value(self.expr))


class _MutableLinearCoefficient(object):
    def __init__(self):
        self.expr = None
        self.var = None
        self.con = None
        self.gurobi_model = None

    def update(self):
        self.gurobi_model.chgCoeff(self.con, self.var, value(self.expr))


class _MutableRangeConstant(object):
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


class _MutableConstant(object):
    def __init__(self):
        self.expr = None
        self.con = None

    def update(self):
        self.con.rhs = value(self.expr)


class _MutableQuadraticConstraint(object):
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


class _MutableObjective(object):
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


class _MutableQuadraticCoefficient(object):
    def __init__(self):
        self.expr = None
        self.var1 = None
        self.var2 = None


class Gurobi(PersistentBase, PersistentSolver):
    """
    Interface to Gurobi
    """

    _available = None
    _num_instances = 0

    def __init__(self, only_child_vars=False):
        super(Gurobi, self).__init__(only_child_vars=only_child_vars)
        self._num_instances += 1
        self._config = GurobiConfig()
        self._solver_options = dict()
        self._solver_model = None
        self._symbol_map = SymbolMap()
        self._labeler = None
        self._pyomo_var_to_solver_var_map = dict()
        self._pyomo_con_to_solver_con_map = dict()
        self._solver_con_to_pyomo_con_map = dict()
        self._pyomo_sos_to_solver_sos_map = dict()
        self._range_constraints = OrderedSet()
        self._mutable_helpers = dict()
        self._mutable_bounds = dict()
        self._mutable_quadratic_helpers = dict()
        self._mutable_objective = None
        self._needs_updated = True
        self._callback = None
        self._callback_func = None
        self._constraints_added_since_update = OrderedSet()
        self._vars_added_since_update = ComponentSet()
        self._last_results_object: Optional[GurobiResults] = None

    def available(self):
        if not gurobipy_available:  # this triggers the deferred import
            return self.Availability.NotFound
        elif self._available == self.Availability.BadVersion:
            return self.Availability.BadVersion
        else:
            return self._check_license()

    def _check_license(self):
        avail = False
        try:
            # Gurobipy writes out license file information when creating
            # the environment
            with capture_output(capture_fd=True):
                m = gurobipy.Model()
            if self._solver_model is None:
                self._solver_model = m
            avail = True
        except gurobipy.GurobiError:
            avail = False

        if avail:
            if self._available is None:
                res = Gurobi._check_full_license()
                self._available = res
                return res
            else:
                return self._available
        else:
            return self.Availability.BadLicense

    @classmethod
    def _check_full_license(cls):
        m = gurobipy.Model()
        m.setParam('OutputFlag', 0)
        try:
            m.addVars(range(2001))
            m.optimize()
            return cls.Availability.FullLicense
        except gurobipy.GurobiError:
            return cls.Availability.LimitedLicense

    def release_license(self):
        self._reinit()
        if gurobipy_available:
            with capture_output(capture_fd=True):
                gurobipy.disposeDefaultEnv()

    def __del__(self):
        if not python_is_shutting_down():
            self._num_instances -= 1
            if self._num_instances == 0:
                self.release_license()

    def version(self):
        version = (
            gurobipy.GRB.VERSION_MAJOR,
            gurobipy.GRB.VERSION_MINOR,
            gurobipy.GRB.VERSION_TECHNICAL,
        )
        return version

    @property
    def config(self) -> GurobiConfig:
        return self._config

    @config.setter
    def config(self, val: GurobiConfig):
        self._config = val

    @property
    def gurobi_options(self):
        """
        A dictionary mapping solver options to values for those options. These
        are solver specific.

        Returns
        -------
        dict
            A dictionary mapping solver options to values for those options
        """
        return self._solver_options

    @gurobi_options.setter
    def gurobi_options(self, val: Dict):
        self._solver_options = val

    @property
    def symbol_map(self):
        return self._symbol_map

    def _solve(self, timer: HierarchicalTimer):
        ostreams = [
            LogStream(
                level=self.config.log_level, logger=self.config.solver_output_logger
            )
        ]
        if self.config.stream_solver:
            ostreams.append(sys.stdout)

        with capture_output(output=TeeStream(*ostreams), capture_fd=False):
            config = self.config
            options = self.gurobi_options

            self._solver_model.setParam('LogToConsole', 1)
            self._solver_model.setParam('LogFile', config.logfile)

            if config.time_limit is not None:
                self._solver_model.setParam('TimeLimit', config.time_limit)
            if config.mip_gap is not None:
                self._solver_model.setParam('MIPGap', config.mip_gap)

            for key, option in options.items():
                self._solver_model.setParam(key, option)

            timer.start('optimize')
            self._solver_model.optimize(self._callback)
            timer.stop('optimize')

        self._needs_updated = False
        return self._postsolve(timer)

    def solve(self, model, timer: HierarchicalTimer = None) -> Results:
        StaleFlagManager.mark_all_as_stale()
        # Note: solver availability check happens in set_instance(),
        # which will be called (either by the user before this call, or
        # below) before this method calls self._solve.
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
        var_names = list()
        vtypes = list()
        lbs = list()
        ubs = list()
        mutable_lbs = dict()
        mutable_ubs = dict()
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

    def _add_params(self, params: List[ParamData]):
        pass

    def _reinit(self):
        saved_config = self.config
        saved_options = self.gurobi_options
        saved_update_config = self.update_config
        self.__init__(only_child_vars=self._only_child_vars)
        self.config = saved_config
        self.gurobi_options = saved_options
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

        if model.name is not None:
            self._solver_model = gurobipy.Model(model.name)
        else:
            self._solver_model = gurobipy.Model()

        self.add_block(model)
        if self._objective is None:
            self.set_objective(None)

    def _get_expr_from_pyomo_expr(self, expr):
        mutable_linear_coefficients = list()
        mutable_quadratic_coefficients = list()
        repn = generate_standard_repn(expr, quadratic=True, compute_values=False)

        degree = repn.polynomial_degree()
        if (degree is None) or (degree > 2):
            raise DegreeError(
                'GurobiAuto does not support expressions of degree {0}.'.format(degree)
            )

        if len(repn.linear_vars) > 0:
            linear_coef_vals = list()
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
                        "or an upper bound: {0} \n".format(con)
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
                        "or an upper bound: {0} \n".format(con)
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
                    'Unrecognized Gurobi expression type: ' + str(gurobi_expr.__class__)
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
                    "Solver does not support SOS level {0} constraints".format(level)
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

    def _remove_params(self, params: List[ParamData]):
        pass

    def _update_variables(self, variables: List[VarData]):
        for var in variables:
            var_id = id(var)
            if var_id not in self._pyomo_var_to_solver_var_map:
                raise ValueError(
                    'The Var provided to update_var needs to be added first: {0}'.format(
                        var
                    )
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

    def update_params(self):
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
            mutable_linear_coefficients = list()
            mutable_quadratic_coefficients = list()
        else:
            if obj.sense == minimize:
                sense = gurobipy.GRB.MINIMIZE
            elif obj.sense == maximize:
                sense = gurobipy.GRB.MAXIMIZE
            else:
                raise ValueError(
                    'Objective sense is not recognized: {0}'.format(obj.sense)
                )

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
        config = self.config

        gprob = self._solver_model
        grb = gurobipy.GRB
        status = gprob.Status

        results = GurobiResults(self)
        results.wallclock_time = gprob.Runtime

        if status == grb.LOADED:  # problem is loaded, but no solution
            results.termination_condition = TerminationCondition.unknown
        elif status == grb.OPTIMAL:  # optimal
            results.termination_condition = TerminationCondition.optimal
        elif status == grb.INFEASIBLE:
            results.termination_condition = TerminationCondition.infeasible
        elif status == grb.INF_OR_UNBD:
            results.termination_condition = TerminationCondition.infeasibleOrUnbounded
        elif status == grb.UNBOUNDED:
            results.termination_condition = TerminationCondition.unbounded
        elif status == grb.CUTOFF:
            results.termination_condition = TerminationCondition.objectiveLimit
        elif status == grb.ITERATION_LIMIT:
            results.termination_condition = TerminationCondition.maxIterations
        elif status == grb.NODE_LIMIT:
            results.termination_condition = TerminationCondition.maxIterations
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

        results.best_feasible_objective = None
        results.best_objective_bound = None
        if self._objective is not None:
            try:
                results.best_feasible_objective = gprob.ObjVal
            except (gurobipy.GurobiError, AttributeError):
                results.best_feasible_objective = None
            try:
                results.best_objective_bound = gprob.ObjBound
            except (gurobipy.GurobiError, AttributeError):
                if self._objective.sense == minimize:
                    results.best_objective_bound = -math.inf
                else:
                    results.best_objective_bound = math.inf

            if results.best_feasible_objective is not None and not math.isfinite(
                results.best_feasible_objective
            ):
                results.best_feasible_objective = None

        timer.start('load solution')
        if config.load_solution:
            if gprob.SolCount > 0:
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
                    'If using the appsi.solvers.Gurobi interface, you can '
                    'set opt.config.load_solution=False. If using the environ.SolverFactory '
                    'interface, you can set opt.solve(model, load_solutions = False). '
                    'Then you can check results.termination_condition and '
                    'results.best_feasible_objective before loading a solution.'
                )
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

    def load_vars(self, vars_to_load=None, solution_number=0):
        for v, val in self.get_primals(
            vars_to_load=vars_to_load, solution_number=solution_number
        ).items():
            v.set_value(val, skip_validation=True)
        StaleFlagManager.mark_all_as_stale(delayed=True)

    def get_primals(self, vars_to_load=None, solution_number=0):
        if self._needs_updated:
            self._update_gurobi_model()  # this is needed to ensure that solutions cannot be loaded after the model has been changed

        if self._solver_model.SolCount == 0:
            raise RuntimeError(
                'Solver does not currently have a valid solution. Please '
                'check the termination condition.'
            )

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
        else:
            gurobi_vars_to_load = [
                var_map[pyomo_var_id] for pyomo_var_id in vars_to_load
            ]
            vals = self._solver_model.getAttr("X", gurobi_vars_to_load)

            res = ComponentMap()
            for var_id, val in zip(vars_to_load, vals):
                using_cons, using_sos, using_obj = ref_vars[var_id]
                if using_cons or using_sos or (using_obj is not None):
                    res[self._vars[var_id][0]] = val
            return res

    def get_reduced_costs(self, vars_to_load=None):
        if self._needs_updated:
            self._update_gurobi_model()

        if self._solver_model.Status != gurobipy.GRB.OPTIMAL:
            raise RuntimeError(
                'Solver does not currently have valid reduced costs. Please '
                'check the termination condition.'
            )

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

    def get_duals(self, cons_to_load=None):
        if self._needs_updated:
            self._update_gurobi_model()

        if self._solver_model.Status != gurobipy.GRB.OPTIMAL:
            raise RuntimeError(
                'Solver does not currently have valid duals. Please '
                'check the termination condition.'
            )

        con_map = self._pyomo_con_to_solver_con_map
        reverse_con_map = self._solver_con_to_pyomo_con_map
        dual = dict()

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

    def get_slacks(self, cons_to_load=None):
        if self._needs_updated:
            self._update_gurobi_model()

        if self._solver_model.SolCount == 0:
            raise RuntimeError(
                'Solver does not currently have valid slacks. Please '
                'check the termination condition.'
            )

        con_map = self._pyomo_con_to_solver_con_map
        reverse_con_map = self._solver_con_to_pyomo_con_map
        slack = dict()

        gurobi_range_con_vars = OrderedSet(self._solver_model.getVars()) - OrderedSet(
            self._pyomo_var_to_solver_var_map.values()
        )

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
        linear_vals = self._solver_model.getAttr("Slack", linear_cons_to_load)
        quadratic_vals = self._solver_model.getAttr("QCSlack", quadratic_cons_to_load)

        for gurobi_con, val in zip(linear_cons_to_load, linear_vals):
            pyomo_con = reverse_con_map[id(gurobi_con)]
            if pyomo_con in self._range_constraints:
                lin_expr = self._solver_model.getRow(gurobi_con)
                for i in reversed(range(lin_expr.size())):
                    v = lin_expr.getVar(i)
                    if v in gurobi_range_con_vars:
                        Us_ = v.X
                        Ls_ = v.UB - v.X
                        if Us_ > Ls_:
                            slack[pyomo_con] = Us_
                        else:
                            slack[pyomo_con] = -Ls_
                        break
            else:
                slack[pyomo_con] = val
        for gurobi_con, val in zip(quadratic_cons_to_load, quadratic_vals):
            pyomo_con = reverse_con_map[id(gurobi_con)]
            slack[pyomo_con] = val
        return slack

    def update(self, timer: HierarchicalTimer = None):
        if self._needs_updated:
            self._update_gurobi_model()
        super(Gurobi, self).update(timer=timer)
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
                'Linear constraint attr {0} cannot be set with'
                + ' the set_linear_constraint_attr method. Please use'
                + ' the remove_constraint and add_constraint methods.'.format(attr)
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
                'Var attr {0} cannot be set with'
                + ' the set_var_attr method. Please use'
                + ' the update_var method.'.format(attr)
            )
        if attr == 'Obj':
            raise ValueError(
                'Var attr Obj cannot be set with'
                + ' the set_var_attr method. Please use'
                + ' the set_objective method.'
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
                ...         cb_opt.cbGetSolution(vars=[m.x, m.y])
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
                raise ValueError(
                    'Lower bound of constraint {0} is not constant.'.format(con)
                )
        if con.has_ub():
            if not is_fixed(con.upper):
                raise ValueError(
                    'Upper bound of constraint {0} is not constant.'.format(con)
                )

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
                'Constraint does not have a lower or an upper bound {0} \n'.format(con)
            )

    def cbGet(self, what):
        return self._solver_model.cbGet(what)

    def cbGetNodeRel(self, vars):
        """
        Parameters
        ----------
        vars: Var or iterable of Var
        """
        if not isinstance(vars, Iterable):
            vars = [vars]
        gurobi_vars = [self._pyomo_var_to_solver_var_map[id(i)] for i in vars]
        var_values = self._solver_model.cbGetNodeRel(gurobi_vars)
        for i, v in enumerate(vars):
            v.set_value(var_values[i], skip_validation=True)

    def cbGetSolution(self, vars):
        """
        Parameters
        ----------
        vars: iterable of vars
        """
        if not isinstance(vars, Iterable):
            vars = [vars]
        gurobi_vars = [self._pyomo_var_to_solver_var_map[id(i)] for i in vars]
        var_values = self._solver_model.cbGetSolution(gurobi_vars)
        for i, v in enumerate(vars):
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
                raise ValueError(
                    'Lower bound of constraint {0} is not constant.'.format(con)
                )
        if con.has_ub():
            if not is_fixed(con.upper):
                raise ValueError(
                    'Upper bound of constraint {0} is not constant.'.format(con)
                )

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
                'Constraint does not have a lower or an upper bound {0} \n'.format(con)
            )

    def cbSetSolution(self, vars, solution):
        if not isinstance(vars, Iterable):
            vars = [vars]
        gurobi_vars = [self._pyomo_var_to_solver_var_map[id(i)] for i in vars]
        self._solver_model.cbSetSolution(gurobi_vars, solution)

    def cbUseSolution(self):
        return self._solver_model.cbUseSolution()

    def reset(self):
        self._solver_model.reset()
