import logging
import math
from typing import List, Dict, Optional
from pyomo.common.collections import ComponentMap, OrderedSet
from pyomo.common.dependencies import attempt_import
from pyomo.common.errors import PyomoException
from pyomo.common.timing import HierarchicalTimer
from pyomo.common.shutdown import python_is_shutting_down
from pyomo.common.config import ConfigValue
from pyomo.core.kernel.objective import minimize, maximize
from pyomo.core.base import SymbolMap, NumericLabeler, TextLabeler
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.base.constraint import _GeneralConstraintData
from pyomo.core.base.sos import _SOSConstraintData
from pyomo.core.base.param import _ParamData
from pyomo.core.expr.numvalue import value, is_constant, native_numeric_types
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

logger = logging.getLogger(__name__)


coptpy_available = False
try:
    import coptpy

    if not (
        coptpy.COPT.VERSION_MAJOR > 6
        or (coptpy.COPT.VERSION_MAJOR == 6 and coptpy.COPT.VERSION_MINOR >= 5)
    ):
        raise ImportError('The APPSI Copt interface requires coptpy>=6.5.0')
    coptpy_available = True
except:
    pass


class DegreeError(PyomoException):
    pass


class CoptConfig(MIPSolverConfig):
    def __init__(
        self,
        description=None,
        doc=None,
        implicit=False,
        implicit_domain=None,
        visibility=0,
    ):
        super(CoptConfig, self).__init__(
            description=description,
            doc=doc,
            implicit=implicit,
            implicit_domain=implicit_domain,
            visibility=visibility,
        )

        self.declare('logfile', ConfigValue(domain=str))
        self.logfile = ''


class CoptSolutionLoader(PersistentSolutionLoader):
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


class CoptResults(Results):
    def __init__(self, solver):
        super(CoptResults, self).__init__()
        self.wallclock_time = None
        self.solution_loader = CoptSolutionLoader(solver=solver)


class _MutableLowerBound(object):
    def __init__(self, expr):
        self.var = None
        self.expr = expr

    def update(self):
        self.var.lb = value(self.expr)


class _MutableUpperBound(object):
    def __init__(self, expr):
        self.var = None
        self.expr = expr

    def update(self):
        self.var.ub = value(self.expr)


class _MutableLinearCoefficient(object):
    def __init__(self):
        self.expr = None
        self.var = None
        self.con = None
        self.copt_model = None

    def update(self):
        self.copt_model.sefCoeff(self.con, self.var, value(self.expr))


class _MutableRangeConstant(object):
    def __init__(self):
        self.lhs_expr = None
        self.rhs_expr = None
        self.con = None
        self.copt_model = None

    def update(self):
        lhs_val = value(self.lhs_expr)
        rhs_val = value(self.rhs_expr)
        self.con.lb = lhs_val
        self.con.ub = rhs_val


class _MutableConstant(object):
    def __init__(self):
        self.expr = None
        self.con = None

    def update(self):
        if self.con.equality or (self.con.has_lb() and self.con.has_ub()):
            self.con.lb = value(self.expr)
            self.con.ub = value(self.expr)
        elif self.con.has_lb():
            self.con.lb = value(self.expr)
        elif self.con.has_ub():
            self.con.ub = value(self.expr)
        else:
            raise ValueError(
                "Constraint does not has lower/upper bound: {0} \n".format(self.con)
            )


class _MutableQuadraticConstraint(object):
    def __init__(self, copt_model, copt_con, constant, linear_coefs, quadratic_coefs):
        self.con = copt_con
        self.copt_model = copt_model
        self.constant = constant
        self.last_constant_value = value(self.constant.expr)
        self.linear_coefs = linear_coefs
        self.last_linear_coef_values = [value(i.expr) for i in self.linear_coefs]
        self.quadratic_coefs = quadratic_coefs
        self.last_quadratic_coef_values = [value(i.expr) for i in self.quadratic_coefs]

    def get_updated_expression(self):
        copt_expr = self.copt_model.getQuadRow(self.con)
        for ndx, coef in enumerate(self.linear_coefs):
            current_coef_value = value(coef.expr)
            incremental_coef_value = (
                current_coef_value - self.last_linear_coef_values[ndx]
            )
            copt_expr += incremental_coef_value * coef.var
            self.last_linear_coef_values[ndx] = current_coef_value
        for ndx, coef in enumerate(self.quadratic_coefs):
            current_coef_value = value(coef.expr)
            incremental_coef_value = (
                current_coef_value - self.last_quadratic_coef_values[ndx]
            )
            copt_expr += incremental_coef_value * coef.var1 * coef.var2
            self.last_quadratic_coef_values[ndx] = current_coef_value
        return copt_expr

    def get_updated_rhs(self):
        return value(self.constant.expr)


class _MutableObjective(object):
    def __init__(self, copt_model, constant, linear_coefs, quadratic_coefs):
        self.copt_model = copt_model
        self.constant = constant
        self.linear_coefs = linear_coefs
        self.quadratic_coefs = quadratic_coefs
        self.last_quadratic_coef_values = [value(i.expr) for i in self.quadratic_coefs]

    def get_updated_expression(self):
        for ndx, coef in enumerate(self.linear_coefs):
            coef.var.obj = value(coef.expr)
        self.copt_model.objconst = value(self.constant.expr)

        copt_expr = None
        for ndx, coef in enumerate(self.quadratic_coefs):
            if value(coef.expr) != self.last_quadratic_coef_values[ndx]:
                if copt_expr is None:
                    copt_expr = self.copt_model.getObjective()
                current_coef_value = value(coef.expr)
                incremental_coef_value = (
                    current_coef_value - self.last_quadratic_coef_values[ndx]
                )
                copt_expr += incremental_coef_value * coef.var1 * coef.var2
                self.last_quadratic_coef_values[ndx] = current_coef_value
        return copt_expr


class _MutableQuadraticCoefficient(object):
    def __init__(self):
        self.expr = None
        self.var1 = None
        self.var2 = None


class Copt(PersistentBase, PersistentSolver):
    """
    Interface to Copt
    """

    _available = None
    _num_instances = 0

    def __init__(self, only_child_vars=True):
        super(Copt, self).__init__(only_child_vars=only_child_vars)
        self._num_instances += 1
        self._config = CoptConfig()

        self._solver_options = dict()
        self._solver_model = None

        if coptpy_available:
            self._coptenv = coptpy.Envr()
        else:
            self._coptenv = None

        self._symbol_map = SymbolMap()
        self._labeler = None

        self._pyomo_var_to_solver_var_map = dict()
        self._pyomo_con_to_solver_con_map = dict()
        self._solver_con_to_pyomo_con_map = dict()
        self._pyomo_sos_to_solver_sos_map = dict()

        self._mutable_helpers = dict()
        self._mutable_bounds = dict()
        self._mutable_quadratic_helpers = dict()
        self._mutable_objective = None

        self._last_results_object: Optional[CoptResults] = None

    def available(self):
        if self._available is None:
            m = self._coptenv.createModel('checklic')
            m.setParam("Logging", 0)
            try:
                # COPT can solve LP up to 10k variables without license
                m.addVars(10001)
                m.solveLP()
                self._available = Copt.Availability.FullLicense
            except coptpy.CoptError:
                self._available = Copt.Availability.LimitedLicense
        return self._available

    def release_license(self):
        self._reinit()

    def __del__(self):
        if not python_is_shutting_down():
            self._num_instances -= 1
            if self._num_instances == 0:
                self.release_license()

    def version(self):
        version = (
            coptpy.COPT.VERSION_MAJOR,
            coptpy.COPT.VERSION_MINOR,
            coptpy.COPT.VERSION_TECHNICAL,
        )
        return version

    @property
    def config(self) -> CoptConfig:
        return self._config

    @config.setter
    def config(self, val: CoptConfig):
        self._config = val

    @property
    def copt_options(self):
        """
        A dictionary mapping solver options to values for those options.
        These are solver specific.

        Returns
        -------
        dict
            A dictionary mapping solver options to values for those options.
        """
        return self._solver_options

    @copt_options.setter
    def copt_options(self, val: Dict):
        self._solver_options = val

    @property
    def symbol_map(self):
        return self._symbol_map

    def _solve(self, timer: HierarchicalTimer):
        config = self.config
        options = self.copt_options
        if config.stream_solver:
            self._solver_model.setParam('LogToConsole', 1)
        else:
            self._solver_model.setParam('LogToConsole', 0)
        self._solver_model.setLogFile(config.logfile)

        if config.time_limit is not None:
            self._solver_model.setParam('TimeLimit', config.time_limit)
        if config.mip_gap is not None:
            self._solver_model.setParam('RelGap', config.mip_gap)

        for key, option in options.items():
            self._solver_model.setParam(key, option)
        timer.start('solve')
        self._solver_model.solve(self._callback)
        timer.stop('solve')
        self._needs_updated = False
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

    def _process_domain_and_bounds(
        self, var, var_id, mutable_lbs, mutable_ubs, ndx, coptpy_var
    ):
        _v, _lb, _ub, _fixed, _domain_interval, _value = self._vars[id(var)]
        lb, ub, step = _domain_interval
        if lb is None:
            lb = -coptpy.COPT.INFINITY
        if ub is None:
            ub = coptpy.COPT.INFINITY
        if step == 0:
            vtype = coptpy.COPT.CONTINUOUS
        elif step == 1:
            if lb == 0 and ub == 1:
                vtype = coptpy.COPT.BINARY
            else:
                vtype = coptpy.COPT.INTEGER
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
                    if coptpy_var is None:
                        mutable_lbs[ndx] = mutable_bound
                    else:
                        mutable_bound.var = coptpy_var
                    self._mutable_bounds[var_id, 'lb'] = (var, mutable_bound)
                lb = max(value(_lb), lb)
            if _ub is not None:
                if not is_constant(_ub):
                    mutable_bound = _MutableUpperBound(NPV_MinExpression((_ub, ub)))
                    if coptpy_var is None:
                        mutable_ubs[ndx] = mutable_bound
                    else:
                        mutable_bound.var = coptpy_var
                    self._mutable_bounds[var_id, 'ub'] = (var, mutable_bound)
                ub = min(value(_ub), ub)

        return lb, ub, vtype

    def _add_variables(self, variables: List[_GeneralVarData]):
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

        nvars = len(variables)

        copt_vars = list()
        for i in range(nvars):
            copt_vars.append(
                self._solver_model.addVar(
                    lb=lbs[i], ub=ubs[i], vtype=vtypes[i], name=var_names[i]
                )
            )

        for ndx, pyomo_var in enumerate(variables):
            copt_var = copt_vars[ndx]
            self._pyomo_var_to_solver_var_map[id(pyomo_var)] = copt_var
        for ndx, mutable_bound in mutable_lbs.items():
            mutable_bound.var = copt_vars[ndx]
        for ndx, mutable_bound in mutable_ubs.items():
            mutable_bound.var = copt_vars[ndx]

    def _add_params(self, params: List[_ParamData]):
        pass

    def _reinit(self):
        saved_config = self.config
        saved_options = self.copt_options
        saved_update_config = self.update_config
        self.__init__(only_child_vars=self._only_child_vars)
        self.config = saved_config
        self.copt_options = saved_options
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
            self._solver_model = self._coptenv.createModel(model.name)
        else:
            self._solver_model = self._coptenv.createModel()

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
                'CoptAPPSI does not support expressions of degree {0}.'.format(degree)
            )

        if len(repn.quadratic_vars) > 0:
            new_expr = coptpy.QuadExpr(0.0)
        else:
            new_expr = coptpy.LinExpr(0.0)

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
            new_expr += coptpy.LinExpr(
                [self._pyomo_var_to_solver_var_map[id(i)] for i in repn.linear_vars],
                linear_coef_vals,
            )

        for ndx, v in enumerate(repn.quadratic_vars):
            x, y = v
            copt_x = self._pyomo_var_to_solver_var_map[id(x)]
            copt_y = self._pyomo_var_to_solver_var_map[id(y)]
            coef = repn.quadratic_coefs[ndx]
            if not is_constant(coef):
                mutable_quadratic_coefficient = _MutableQuadraticCoefficient()
                mutable_quadratic_coefficient.expr = coef
                mutable_quadratic_coefficient.var1 = copt_x
                mutable_quadratic_coefficient.var2 = copt_y
                mutable_quadratic_coefficients.append(mutable_quadratic_coefficient)
            coef_val = value(coef)
            new_expr += coef_val * copt_x * copt_y

        return (
            new_expr,
            repn.constant,
            mutable_linear_coefficients,
            mutable_quadratic_coefficients,
        )

    def _add_constraints(self, cons: List[_GeneralConstraintData]):
        for con in cons:
            conname = self._symbol_map.getSymbol(con, self._labeler)
            (
                copt_expr,
                repn_constant,
                mutable_linear_coefficients,
                mutable_quadratic_coefficients,
            ) = self._get_expr_from_pyomo_expr(con.body)

            if (
                copt_expr.__class__ in {coptpy.LinExpr, coptpy.Var}
                or copt_expr.__class__ in native_numeric_types
            ):
                if con.equality:
                    rhs_expr = con.lower - repn_constant
                    rhs_val = value(rhs_expr)
                    coptpy_con = self._solver_model.addConstr(
                        copt_expr == rhs_val, name=conname
                    )

                    if not is_constant(rhs_expr):
                        mutable_constant = _MutableConstant()
                        mutable_constant.expr = rhs_expr
                        mutable_constant.con = coptpy_con
                        self._mutable_helpers[con] = [mutable_constant]
                elif con.has_lb() and con.has_ub():
                    lhs_expr = con.lower - repn_constant
                    rhs_expr = con.upper - repn_constant
                    lhs_val = value(lhs_expr)
                    rhs_val = value(rhs_expr)
                    coptpy_con = self._solver_model.addBoundConstr(
                        copt_expr, lhs_val, rhs_val, name=conname
                    )

                    if not is_constant(lhs_expr) or not is_constant(rhs_expr):
                        mutable_range_constant = _MutableRangeConstant()
                        mutable_range_constant.lhs_expr = lhs_expr
                        mutable_range_constant.rhs_expr = rhs_expr
                        mutable_range_constant.con = coptpy_con
                        mutable_range_constant.copt_model = self._solver_model
                        self._mutable_helpers[con] = [mutable_range_constant]
                elif con.has_lb():
                    rhs_expr = con.lower - repn_constant
                    rhs_val = value(rhs_expr)
                    coptpy_con = self._solver_model.addConstr(
                        copt_expr >= rhs_val, name=conname
                    )
                    if not is_constant(rhs_expr):
                        mutable_constant = _MutableConstant()
                        mutable_constant.expr = rhs_expr
                        mutable_constant.con = coptpy_con
                        self._mutable_helpers[con] = [mutable_constant]
                elif con.has_ub():
                    rhs_expr = con.upper - repn_constant
                    rhs_val = value(rhs_expr)
                    coptpy_con = self._solver_model.addConstr(
                        copt_expr <= rhs_val, name=conname
                    )
                    if not is_constant(rhs_expr):
                        mutable_constant = _MutableConstant()
                        mutable_constant.expr = rhs_expr
                        mutable_constant.con = coptpy_con
                        self._mutable_helpers[con] = [mutable_constant]
                else:
                    raise ValueError(
                        "Constraint does not have a lower or an upper bound: {0} \n".format(
                            con
                        )
                    )

                for tmp in mutable_linear_coefficients:
                    tmp.con = coptpy_con
                    tmp.copt_model = self._solver_model
                if len(mutable_linear_coefficients) > 0:
                    if con not in self._mutable_helpers:
                        self._mutable_helpers[con] = mutable_linear_coefficients
                    else:
                        self._mutable_helpers[con].extend(mutable_linear_coefficients)
            elif copt_expr.__class__ is coptpy.QuadExpr:
                if con.equality:
                    rhs_expr = con.lower - repn_constant
                    rhs_val = value(rhs_expr)
                    coptpy_con = self._solver_model.addQConstr(
                        copt_expr == rhs_val, name=conname
                    )
                elif con.has_lb() and con.has_ub():
                    raise NotImplementedError(
                        'Quadratic range constraints are not supported'
                    )
                elif con.has_lb():
                    rhs_expr = con.lower - repn_constant
                    rhs_val = value(rhs_expr)
                    coptpy_con = self._solver_model.addQConstr(
                        copt_expr >= rhs_val, name=conname
                    )
                elif con.has_ub():
                    rhs_expr = con.upper - repn_constant
                    rhs_val = value(rhs_expr)
                    coptpy_con = self._solver_model.addQConstr(
                        copt_expr <= rhs_val, name=conname
                    )
                else:
                    raise ValueError(
                        "Constraint does not have a lower or an upper bound: {0} \n".format(
                            con
                        )
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
                        coptpy_con,
                        mutable_constant,
                        mutable_linear_coefficients,
                        mutable_quadratic_coefficients,
                    )
                    self._mutable_quadratic_helpers[con] = mutable_quadratic_constraint
            else:
                raise ValueError(
                    'Unrecognized COPT expression type: ' + str(copt_expr.__class__)
                )

            self._pyomo_con_to_solver_con_map[con] = coptpy_con
            self._solver_con_to_pyomo_con_map[id(coptpy_con)] = con

    def _add_sos_constraints(self, cons: List[_SOSConstraintData]):
        for con in cons:
            self._symbol_map.getSymbol(con, self._labeler)

            level = con.level
            if level == 1:
                sos_type = coptpy.COPT.SOS_TYPE1
            elif level == 2:
                sos_type = coptpy.COPT.SOS_TYPE2
            else:
                raise ValueError(
                    "Solver does not support SOS level {0} constraints".format(level)
                )

            copt_vars = []
            weights = []

            for v, w in con.get_items():
                v_id = id(v)
                copt_vars.append(self._pyomo_var_to_solver_var_map[v_id])
                weights.append(w)

            coptpy_con = self._solver_model.addSOS(sos_type, copt_vars, weights)
            self._pyomo_sos_to_solver_sos_map[con] = coptpy_con

    def _remove_constraints(self, cons: List[_GeneralConstraintData]):
        for con in cons:
            solver_con = self._pyomo_con_to_solver_con_map[con]
            self._solver_model.remove(solver_con)
            self._symbol_map.removeSymbol(con)
            del self._pyomo_con_to_solver_con_map[con]
            del self._solver_con_to_pyomo_con_map[id(solver_con)]
            self._mutable_helpers.pop(con, None)
            self._mutable_quadratic_helpers.pop(con, None)

    def _remove_sos_constraints(self, cons: List[_SOSConstraintData]):
        for con in cons:
            solver_sos_con = self._pyomo_sos_to_solver_sos_map[con]
            self._solver_model.remove(solver_sos_con)
            self._symbol_map.removeSymbol(con)
            del self._pyomo_sos_to_solver_sos_map[con]

    def _remove_variables(self, variables: List[_GeneralVarData]):
        for var in variables:
            v_id = id(var)
            solver_var = self._pyomo_var_to_solver_var_map[v_id]
            self._solver_model.remove(solver_var)
            self._symbol_map.removeSymbol(var)
            del self._pyomo_var_to_solver_var_map[v_id]
            self._mutable_bounds.pop(v_id, None)

    def _remove_params(self, params: List[_ParamData]):
        pass

    def _update_variables(self, variables: List[_GeneralVarData]):
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
            coptpy_var = self._pyomo_var_to_solver_var_map[var_id]
            lb, ub, vtype = self._process_domain_and_bounds(
                var, var_id, None, None, None, coptpy_var
            )
            coptpy_var.lb = lb
            coptpy_var.ub = ub
            coptpy_var.vtype = vtype

    def update_params(self):
        for con, helpers in self._mutable_helpers.items():
            for helper in helpers:
                helper.update()
        for k, (v, helper) in self._mutable_bounds.items():
            helper.update()

        for con, helper in self._mutable_quadratic_helpers.items():
            copt_con = helper.con
            new_copt_expr = helper.get_updated_expression()
            new_rhs = helper.get_updated_rhs()
            new_sense = copt_con.sense
            pyomo_con = self._solver_con_to_pyomo_con_map[id(copt_con)]
            name = self._symbol_map.getSymbol(pyomo_con, self._labeler)
            self._solver_model.remove(copt_con)
            new_con = self._solver_model.addQConstr(
                new_copt_expr, new_sense, new_rhs, name=name
            )
            self._pyomo_con_to_solver_con_map[id(pyomo_con)] = new_con
            del self._solver_con_to_pyomo_con_map[id(copt_con)]
            self._solver_con_to_pyomo_con_map[id(new_con)] = pyomo_con
            helper.con = new_con

        helper = self._mutable_objective
        pyomo_obj = self._objective
        new_copt_expr = helper.get_updated_expression()
        if new_copt_expr is not None:
            if pyomo_obj.sense == minimize:
                sense = coptpy.COPT.MINIMIZE
            else:
                sense = coptpy.COPT.MAXIMIZE
            self._solver_model.setObjective(new_copt_expr, sense=sense)

    def _set_objective(self, obj):
        if obj is None:
            sense = coptpy.COPT.MINIMIZE
            copt_expr = 0
            repn_constant = 0
            mutable_linear_coefficients = list()
            mutable_quadratic_coefficients = list()
        else:
            if obj.sense == minimize:
                sense = coptpy.COPT.MINIMIZE
            elif obj.sense == maximize:
                sense = coptpy.COPT.MAXIMIZE
            else:
                raise ValueError(
                    'Objective sense is not recognized: {0}'.format(obj.sense)
                )

            (
                copt_expr,
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

        self._solver_model.setObjective(copt_expr + value(repn_constant), sense=sense)

    def _postsolve(self, timer: HierarchicalTimer):
        config = self.config

        status = self._solver_model.status

        results = CoptResults(self)
        results.wallclock_time = self._solver_model.SolvingTime

        if status == coptpy.COPT.UNSTARTED:
            results.termination_condition = TerminationCondition.unknown
        elif status == coptpy.COPT.OPTIMAL:
            results.termination_condition = TerminationCondition.optimal
        elif status == coptpy.COPT.INFEASIBLE:
            results.termination_condition = TerminationCondition.infeasible
        elif status == coptpy.COPT.UNBOUNDED:
            results.termination_condition = TerminationCondition.unbounded
        elif status == coptpy.COPT.INF_OR_UNB:
            results.termination_condition = TerminationCondition.infeasibleOrUnbounded
        elif status == coptpy.COPT.NUMERICAL:
            results.termination_condition = TerminationCondition.error
        elif status == coptpy.COPT.NODELIMIT:
            results.termination_condition = TerminationCondition.maxIterations
        elif status == 7:  # Imprecise
            results.termination_condition = TerminationCondition.optimal
        elif status == coptpy.COPT.TIMEOUT:
            results.termination_condition = TerminationCondition.maxTimeLimit
        elif status == coptpy.COPT.INTERRUPTED:
            results.termination_condition = TerminationCondition.interrupted
        else:
            results.termination_condition = TerminationCondition.unknown

        results.best_feasible_objective = None
        results.best_objective_bound = None
        if self._objective is not None:
            if self._solver_model.ismip:
                results.best_feasible_objective = self._solver_model.objval
                results.best_objective_bound = self._solver_model.bestbnd
            else:
                results.best_feasible_objective = self._solver_model.lpobjval
                results.best_objective_bound = self._solver_model.lpobjval
            if results.best_feasible_objective is not None and not math.isfinite(
                results.best_feasible_objective
            ):
                results.best_feasible_objective = None

        timer.start('load solution')
        if config.load_solution:
            if self._solver_model.haslpsol or self._solver_model.hasmipsol:
                if results.termination_condition != TerminationCondition.optimal:
                    logger.warning(
                        'Loading a feasible but suboptimal solution. '
                        'Please set load_solution=False and check '
                        'results.termination_condition and '
                        'resutls.found_feasible_solution() before loading a solution.'
                    )
                self.load_vars()
            else:
                raise RuntimeError(
                    'A feasible solution was not found, so no solution can be loaded.'
                    'Please set opt.config.load_solution=False and check '
                    'results.termination_condition and '
                    'resutls.best_feasible_objective before loading a solution.'
                )
        timer.stop('load solution')

        return results

    def _load_suboptimal_mip_solution(self, vars_to_load, solution_number):
        if self.get_model_attr("IsMIP") == 0:
            raise ValueError(
                'Cannot obtain suboptimal solutions for a continuous model'
            )
        var_map = self._pyomo_var_to_solver_var_map
        ref_vars = self._referenced_variables
        copt_vars_to_load = [var_map[pyomo_var] for pyomo_var in vars_to_load]
        vals = self._solver_model.getPoolSolution(solution_number, copt_vars_to_load)
        res = ComponentMap()
        for var_id, val in zip(vars_to_load, vals):
            using_cons, using_sos, using_obj = ref_vars[var_id]
            if using_cons or using_sos or (using_obj is not None):
                res[self._vars[var_id][0]] = val
        return res

    def load_vars(self, vars_to_load=None, solution_number=0):
        for v, val in self.get_primals(
            vars_to_load=vars_to_load, solution_number=solution_number
        ).items():
            v.set_value(val, skip_validation=True)
        StaleFlagManager.mark_all_as_stale(delayed=True)

    def get_primals(self, vars_to_load=None, solution_number=0):
        if self._solver_model.haslpsol == 0 and self._solver_model.hasmipsol == 0:
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

        if solution_number != 0 and self._solver_model.poolsols > 0:
            return self._load_suboptimal_mip_solution(
                vars_to_load=vars_to_load, solution_number=solution_number
            )
        else:
            copt_vars_to_load = [var_map[pyomo_var_id] for pyomo_var_id in vars_to_load]
            vals = self._solver_model.getInfo("Value", copt_vars_to_load)

            res = ComponentMap()
            for var_id, val in zip(vars_to_load, vals):
                using_cons, using_sos, using_obj = ref_vars[var_id]
                if using_cons or using_sos or (using_obj is not None):
                    res[self._vars[var_id][0]] = val
            return res

    def get_reduced_costs(self, vars_to_load=None):
        if self._solver_model.status != coptpy.COPT.OPTIMAL:
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

        copt_vars_to_load = [var_map[pyomo_var_id] for pyomo_var_id in vars_to_load]
        vals = self._solver_model.getInfo("RedCost", copt_vars_to_load)

        for var_id, val in zip(vars_to_load, vals):
            using_cons, using_sos, using_obj = ref_vars[var_id]
            if using_cons or using_sos or (using_obj is not None):
                res[self._vars[var_id][0]] = val

        return res

    def get_duals(self, cons_to_load=None):
        if self._solver_model.Status != coptpy.COPT.OPTIMAL:
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
            copt_cons_to_load = OrderedSet(
                [con_map[pyomo_con] for pyomo_con in cons_to_load]
            )
            linear_cons_to_load = list(
                copt_cons_to_load.intersection(
                    OrderedSet(self._solver_model.getConstrs())
                )
            )
            quadratic_cons_to_load = list(
                copt_cons_to_load.intersection(
                    OrderedSet(self._solver_model.getQConstrs())
                )
            )
        linear_vals = self._solver_model.getInfo("Dual", linear_cons_to_load)
        # TODO: Cannot get duals for quadratic constraints so far with COPT
        quadratic_vals = self._solver_model.getInfo("Dual", quadratic_cons_to_load)

        for copt_con, val in zip(linear_cons_to_load, linear_vals):
            pyomo_con = reverse_con_map[id(copt_con)]
            dual[pyomo_con] = val
        for copt_con, val in zip(quadratic_cons_to_load, quadratic_vals):
            pyomo_con = reverse_con_map[id(copt_con)]
            dual[pyomo_con] = val

        return dual

    def get_slacks(self, cons_to_load=None):
        # NOTE: Slacks of COPT are activities of constraints
        if self._solver_model.haslpsol == 0 and self._solver_model.hasmipsol == 0:
            raise RuntimeError(
                'Solver does not currently have valid slacks. Please '
                'check the termination condition.'
            )

        con_map = self._pyomo_con_to_solver_con_map
        reverse_con_map = self._solver_con_to_pyomo_con_map
        slack = dict()

        if cons_to_load is None:
            linear_cons_to_load = self._solver_model.getConstrs()
            quadratic_cons_to_load = self._solver_model.getQConstrs()
        else:
            copt_cons_to_load = OrderedSet(
                [con_map[pyomo_con] for pyomo_con in cons_to_load]
            )
            linear_cons_to_load = list(
                copt_cons_to_load.intersection(
                    OrderedSet(self._solver_model.getConstrs())
                )
            )
            quadratic_cons_to_load = list(
                copt_cons_to_load.intersection(
                    OrderedSet(self._solver_model.getQConstrs())
                )
            )
        linear_vals = self._solver_model.getInfo("Slack", linear_cons_to_load)
        quadratic_vals = self._solver_model.getInfo("Slack", quadratic_cons_to_load)

        for copt_con, val in zip(linear_cons_to_load, linear_vals):
            pyomo_con = reverse_con_map[id(copt_con)]
            slack[pyomo_con] = val
        for copt_con, val in zip(quadratic_cons_to_load, quadratic_vals):
            pyomo_con = reverse_con_map[id(copt_con)]
            slack[pyomo_con] = val
        return slack

    def update(self, timer: HierarchicalTimer = None):
        pass

    def get_model_attr(self, attr):
        """
        Get the value of an attribute on the COPT model.

        Parameters
        ----------
        attr: str
            The attribute to get. See COPT documentation for descriptions of
            the attributes.
        """
        return getattr(self._solver_model, attr)

    def write(self, filename):
        """
        Write the model to a file.

        Parameters
        ----------
        filename: str
            Name of the file to which the model should be written.
        """
        self._solver_model.write(filename)

    def set_linear_constraint_attr(self, con, attr, val):
        """
        Set the value of information on a COPT linear constraint.

        Parameters
        ----------
        con: pyomo.core.base.constraint._GeneralConstraintData
            The pyomo constraint for which the corresponding COPT constraint attribute
            should be modified.
        attr: str
            The information to be modified. See the COPT documentation.
        val: any
            See COPT documentation for acceptable values.
        """
        setattr(self._pyomo_con_to_solver_con_map[con], attr, val)

    def set_var_attr(self, var, attr, val):
        """
        Set the value of information on a COPT variable.

        Parameters
        ----------
        var: pyomo.core.base.var._GeneralVarData
            The pyomo var for which the corresponding COPT variable information
            should be modified.
        attr: str
            The information to be modified. See the COPT documentation.
        val: any
            See COPT documentation for acceptable values.
        """
        setattr(self._pyomo_var_to_solver_var_map[id(var)], attr, val)

    def get_var_attr(self, var, attr):
        """
        Get the value of information on a COPT variable.

        Parameters
        ----------
        var: pyomo.core.base.var._GeneralVarData
            The pyomo var for which the corresponding COPT variable information
            should be retrieved.
        attr: str
            The information to get. See the COPT documentation.
        """
        return getattr(self._pyomo_var_to_solver_var_map[id(var)], attr)

    def get_linear_constraint_attr(self, con, attr):
        """
        Get the value of information on a COPT linear constraint.

        Parameters
        ----------
        con: pyomo.core.base.constraint._GeneralConstraintData
            The pyomo constraint for which the corresponding COPT constraint information
            should be retrieved.
        attr: str
            The information to get. See the COPT documentation.
        """
        return getattr(self._pyomo_con_to_solver_con_map[con], attr)

    def get_sos_attr(self, con, attr):
        """
        Get the value of information on a COPT sos constraint.

        Parameters
        ----------
        con: pyomo.core.base.sos._SOSConstraintData
            The pyomo SOS constraint for which the corresponding COPT SOS constraint
            information should be retrieved.
        attr: str
            The information to get. See the COPT documentation.
        """
        return getattr(self._pyomo_sos_to_solver_sos_map[con], attr)

    def get_quadratic_constraint_attr(self, con, attr):
        """
        Get the value of information on a COPT quadratic constraint.

        Parameters
        ----------
        con: pyomo.core.base.constraint._GeneralConstraintData
            The pyomo constraint for which the corresponding COPT constraint information
            should be retrieved.
        attr: str
            The information to get. See the COPT documentation.
        """
        return getattr(self._pyomo_con_to_solver_con_map[con], attr)

    def set_copt_param(self, param, val):
        """
        Set a COPT parameter.

        Parameters
        ----------
        param: str
            The COPT parameter to set. Options include any COPT parameter.
            Please see the COPT documentation for options.
        val: any
            The value to set the parameter to. See COPT documentation for possible values.
        """
        self._solver_model.setParam(param, val)

    def get_copt_param_info(self, param):
        """
        Get information about a COPT parameter.

        Parameters
        ----------
        param: str
            The COPT parameter to get info for. See COPT documenation for possible options.

        Returns
        -------
        A 5-tuple containing the parameter name, current value, default value,
        minimum value and maximum value.
        """
        return self._solver_model.getParamInfo(param)

    def reset(self):
        self._solver_model.reset()
