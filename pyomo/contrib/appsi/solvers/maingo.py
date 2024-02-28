from collections import namedtuple
import logging
import math
import sys
from typing import Optional, List, Dict

from pyomo.contrib.appsi.base import (
    PersistentSolver,
    Results,
    TerminationCondition,
    MIPSolverConfig,
    PersistentBase,
    PersistentSolutionLoader,
)
from pyomo.contrib.appsi.cmodel import cmodel, cmodel_available
from pyomo.common.collections import ComponentMap
from pyomo.common.config import ConfigValue, NonNegativeInt
from pyomo.common.dependencies import attempt_import
from pyomo.common.errors import PyomoException
from pyomo.common.log import LogStream
from pyomo.common.tee import capture_output, TeeStream
from pyomo.common.timing import HierarchicalTimer
from pyomo.core.base import SymbolMap, NumericLabeler, TextLabeler
from pyomo.core.base.constraint import _GeneralConstraintData
from pyomo.core.base.expression import ScalarExpression
from pyomo.core.base.param import _ParamData
from pyomo.core.base.sos import _SOSConstraintData
from pyomo.core.base.var import Var, _GeneralVarData
import pyomo.core.expr.expr_common as common
import pyomo.core.expr as EXPR
from pyomo.core.expr.numvalue import (
    value,
    is_constant,
    is_fixed,
    native_numeric_types,
    native_types,
    nonpyomo_leaf_types,
)
from pyomo.core.kernel.objective import minimize, maximize
from pyomo.core.staleflag import StaleFlagManager
from pyomo.repn.util import valid_expr_ctypes_minlp

_plusMinusOne = {-1, 1}

MaingoVar = namedtuple("MaingoVar", "type name lb ub init")

logger = logging.getLogger(__name__)


def _import_maingopy():
    try:
        import maingopy
    except ImportError:
        MAiNGO._available = MAiNGO.Availability.NotFound
        raise
    return maingopy


maingopy, maingopy_available = attempt_import("maingopy", importer=_import_maingopy)


class MAiNGOConfig(MIPSolverConfig):
    def __init__(
        self,
        description=None,
        doc=None,
        implicit=False,
        implicit_domain=None,
        visibility=0,
    ):
        super(MAiNGOConfig, self).__init__(
            description=description,
            doc=doc,
            implicit=implicit,
            implicit_domain=implicit_domain,
            visibility=visibility,
        )

        self.declare("logfile", ConfigValue(domain=str))
        self.declare("solver_output_logger", ConfigValue())
        self.declare("log_level", ConfigValue(domain=NonNegativeInt))

        self.logfile = ""
        self.solver_output_logger = logger
        self.log_level = logging.INFO


class MAiNGOSolutionLoader(PersistentSolutionLoader):
    def load_vars(self, vars_to_load=None):
        self._assert_solution_still_valid()
        self._solver.load_vars(vars_to_load=vars_to_load)

    def get_primals(self, vars_to_load=None):
        self._assert_solution_still_valid()
        return self._solver.get_primals(vars_to_load=vars_to_load)


class MAiNGOResults(Results):
    def __init__(self, solver):
        super(MAiNGOResults, self).__init__()
        self.wallclock_time = None
        self.cpu_time = None
        self.solution_loader = MAiNGOSolutionLoader(solver=solver)


class SolverModel(maingopy.MAiNGOmodel):
    def __init__(self, var_list, objective, con_list, idmap):
        maingopy.MAiNGOmodel.__init__(self)
        self._var_list = var_list
        self._con_list = con_list
        self._objective = objective
        self._idmap = idmap

    def build_maingo_objective(self, obj, visitor):
        maingo_obj = visitor.dfs_postorder_stack(obj.expr)
        if obj.sense == maximize:
            maingo_obj *= -1
        return maingo_obj

    def build_maingo_constraints(self, cons, visitor):
        eqs = []
        ineqs = []
        for con in cons:
            if con.equality:
                eqs += [visitor.dfs_postorder_stack(con.body - con.lower)]
            elif con.has_ub() and con.has_lb():
                ineqs += [visitor.dfs_postorder_stack(con.body - con.upper)]
                ineqs += [visitor.dfs_postorder_stack(con.lower - con.body)]
            elif con.has_ub():
                ineqs += [visitor.dfs_postorder_stack(con.body - con.upper)]
            elif con.has_ub():
                ineqs += [visitor.dfs_postorder_stack(con.lower - con.body)]
            else:
                raise ValueError(
                    "Constraint does not have a lower "
                    "or an upper bound: {0} \n".format(con)
                )
        return eqs, ineqs

    def get_variables(self):
        return [
            maingopy.OptimizationVariable(
                maingopy.Bounds(var.lb, var.ub), var.type, var.name
            )
            for var in self._var_list
        ]

    def get_initial_point(self):
        return [var.init if not var.init is None else (var.lb  + var.ub)/2.0 for var in self._var_list]

    def evaluate(self, maingo_vars):
        visitor = ToMAiNGOVisitor(maingo_vars, self._idmap)
        result = maingopy.EvaluationContainer()
        result.objective = self.build_maingo_objective(self._objective, visitor)
        eqs, ineqs = self.build_maingo_constraints(self._con_list, visitor)
        result.eq = eqs
        result.ineq = ineqs
        return result


LEFT_TO_RIGHT = common.OperatorAssociativity.LEFT_TO_RIGHT
RIGHT_TO_LEFT = common.OperatorAssociativity.RIGHT_TO_LEFT


class ToMAiNGOVisitor(EXPR.ExpressionValueVisitor):
    def __init__(self, variables, idmap):
        super(ToMAiNGOVisitor, self).__init__()
        self.variables = variables
        self.idmap = idmap
        self._pyomo_func_to_maingo_func = {
            "log": maingopy.log,
            "log10": ToMAiNGOVisitor.maingo_log10,
            "sin": maingopy.sin,
            "cos": maingopy.cos,
            "tan": maingopy.tan,
            "cosh": maingopy.cosh,
            "sinh": maingopy.sinh,
            "tanh": maingopy.tanh,
            "asin": maingopy.asin,
            "acos": maingopy.acos,
            "atan": maingopy.atan,
            "exp": maingopy.exp,
            "sqrt": maingopy.sqrt,
            "asinh": ToMAiNGOVisitor.maingo_asinh,
            "acosh": ToMAiNGOVisitor.maingo_acosh,
            "atanh": ToMAiNGOVisitor.maingo_atanh,
        }

    @classmethod
    def maingo_log10(cls, x):
        return maingopy.log(x) / math.log(10)

    @classmethod
    def maingo_asinh(cls, x):
        return maingopy.log(x + maingopy.sqrt(maingopy.pow(x,2) + 1))

    @classmethod
    def maingo_acosh(cls, x):
        return maingopy.log(x + maingopy.sqrt(maingopy.pow(x,2) - 1))

    @classmethod
    def maingo_atanh(cls, x):
        return 0.5 * maingopy.log(x+1) - 0.5 * maingopy.log(1-x)

    def visit(self, node, values):
        """Visit nodes that have been expanded"""
        for i, val in enumerate(values):
            arg = node._args_[i]

            if arg is None:
                values[i] = "Undefined"
            elif arg.__class__ in native_numeric_types:
                pass
            elif arg.__class__ in nonpyomo_leaf_types:
                values[i] = val
            else:
                parens = False
                if arg.is_expression_type() and node.PRECEDENCE is not None:
                    if arg.PRECEDENCE is None:
                        pass
                    elif node.PRECEDENCE < arg.PRECEDENCE:
                        parens = True
                    elif node.PRECEDENCE == arg.PRECEDENCE:
                        if i == 0:
                            parens = node.ASSOCIATIVITY != LEFT_TO_RIGHT
                        elif i == len(node._args_) - 1:
                            parens = node.ASSOCIATIVITY != RIGHT_TO_LEFT
                        else:
                            parens = True
                if parens:
                    values[i] = val

        if node.__class__ in EXPR.NPV_expression_types:
            return value(node)

        if node.__class__ in {EXPR.ProductExpression, EXPR.MonomialTermExpression}:
            return values[0] * values[1]

        if node.__class__ in {EXPR.SumExpression}:
            return sum(values)

        if node.__class__ in {EXPR.PowExpression}:
            return maingopy.pow(values[0], values[1])

        if node.__class__ in {EXPR.DivisionExpression}:
            return values[0] / values[1]

        if node.__class__ in {EXPR.NegationExpression}:
            return -values[0]

        if node.__class__ in {EXPR.AbsExpression}:
            return maingopy.abs(values[0])

        if node.__class__ in {EXPR.UnaryFunctionExpression}:
            pyomo_func = node.getname()
            maingo_func = self._pyomo_func_to_maingo_func[pyomo_func]
            return maingo_func(values[0])

        if node.__class__ in {ScalarExpression}:
            return values[0]

        raise ValueError(f"Unknown function expression encountered: {node.getname()}")

    def visiting_potential_leaf(self, node):
        """
        Visiting a potential leaf.

        Return True if the node is not expanded.
        """
        if node.__class__ in native_types:
            return True, node

        if node.is_expression_type():
            if node.__class__ is EXPR.MonomialTermExpression:
                return True, self._monomial_to_maingo(node)
            if node.__class__ is EXPR.LinearExpression:
                return True, self._linear_to_maingo(node)
            return False, None

        if node.is_component_type():
            if node.ctype not in valid_expr_ctypes_minlp:
                # Make sure all components in active constraints
                # are basic ctypes we know how to deal with.
                raise RuntimeError(
                    "Unallowable component '%s' of type %s found in an active "
                    "constraint or objective.\nMAiNGO cannot export "
                    "expressions with this component type."
                    % (node.name, node.ctype.__name__)
                )

        if node.is_fixed():
            return True, node()
        else:
            assert node.is_variable_type()
            maingo_var_id = self.idmap[id(node)]
            maingo_var = self.variables[maingo_var_id]
            return True, maingo_var

    def _monomial_to_maingo(self, node):
        const, var = node.args
        maingo_var_id = self.idmap[id(var)]
        maingo_var = self.variables[maingo_var_id]
        if const.__class__ not in native_types:
            const = value(const)
        if var.is_fixed():
            return const * var.value
        if not const:
            return 0
        if const in _plusMinusOne:
            if const < 0:
                return -maingo_var
            else:
                return maingo_var
        return const * maingo_var

    def _linear_to_maingo(self, node):
        values = [
            self._monomial_to_maingo(arg)
            if (
                arg.__class__ is EXPR.MonomialTermExpression
                and not arg.arg(1).is_fixed()
            )
            else value(arg)
            for arg in node.args
        ]
        return sum(values)


class MAiNGO(PersistentBase, PersistentSolver):
    """
    Interface to MAiNGO
    """

    _available = None

    def __init__(self, only_child_vars=False):
        super(MAiNGO, self).__init__(only_child_vars=only_child_vars)
        self._config = MAiNGOConfig()
        self._solver_options = dict()
        self._solver_model = None
        self._mymaingo = None
        self._symbol_map = SymbolMap()
        self._labeler = None
        self._maingo_vars = []
        self._objective = None
        self._cons = []
        self._pyomo_var_to_solver_var_id_map = dict()
        self._last_results_object: Optional[MAiNGOResults] = None

    def available(self):
        if not maingopy_available:
            return self.Availability.NotFound
        self._available = True
        return self._available

    def version(self):
        # Check if Python >= 3.8
        if sys.version_info.major >= 3 and sys.version_info.minor >= 8:
            from importlib.metadata import version
            version = version('maingopy')
        else:
            import pkg_resources
            version = pkg_resources.get_distribution('maingopy').version
            
        return tuple(int(k) for k in version.split('.'))

    @property
    def config(self) -> MAiNGOConfig:
        return self._config

    @config.setter
    def config(self, val: MAiNGOConfig):
        self._config = val

    @property
    def maingo_options(self):
        """
        A dictionary mapping solver options to values for those options. These
        are solver specific.

        Returns
        -------
        dict
            A dictionary mapping solver options to values for those options
        """
        return self._solver_options

    @maingo_options.setter
    def maingo_options(self, val: Dict):
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

        with TeeStream(*ostreams) as t:
            with capture_output(output=t.STDOUT, capture_fd=False):
                config = self.config
                options = self.maingo_options

                self._mymaingo = maingopy.MAiNGO(self._solver_model)

                self._mymaingo.set_option("loggingDestination", 2)
                self._mymaingo.set_log_file_name(config.logfile)

                if config.time_limit is not None:
                    self._mymaingo.set_option("maxTime", config.time_limit)
                if config.mip_gap is not None:
                    self._mymaingo.set_option("epsilonR", config.mip_gap)
                for key, option in options.items():
                    self._mymaingo.set_option(key, option)

                timer.start("MAiNGO solve")
                self._mymaingo.solve()
                timer.stop("MAiNGO solve")

        return self._postsolve(timer)

    def solve(self, model, timer: HierarchicalTimer = None):
        StaleFlagManager.mark_all_as_stale()

        if self._last_results_object is not None:
            self._last_results_object.solution_loader.invalidate()
        if timer is None:
            timer = HierarchicalTimer()
        timer.start("set_instance")
        self.set_instance(model)
        timer.stop("set_instance")
        res = self._solve(timer)
        self._last_results_object = res
        if self.config.report_timing:
            logger.info("\n" + str(timer))
        return res

    def _process_domain_and_bounds(self, var):
        _v, _lb, _ub, _fixed, _domain_interval, _value = self._vars[id(var)]
        lb, ub, step = _domain_interval

        if _fixed:
            lb = _value
            ub = _value
        else:
            if lb is None and _lb is None:
                logger.warning("No lower bound for variable " + var.getname() + " set. Using -1e10 instead. Please consider setting a valid lower bound.")
            if ub is None and _ub is None:
                logger.warning("No upper bound for variable " + var.getname() + " set. Using +1e10 instead. Please consider setting a valid upper bound.")
                
            if _lb is None:
                _lb = -1e10
            if _ub is None:
                _ub = 1e10
            if lb is None:
                lb = -1e10
            if ub is None:
                ub = 1e10

            lb = max(value(_lb), lb)
            ub = min(value(_ub), ub)

        if step == 0:
            vtype = maingopy.VT_CONTINUOUS
        elif step == 1:
            if lb == 0 and ub == 1:
                vtype = maingopy.VT_BINARY
            else:
                vtype = maingopy.VT_INTEGER
        else:
            raise ValueError(
                f"Unrecognized domain step: {step} (should be either 0 or 1)"
            )


        return lb, ub, vtype

    def _add_variables(self, variables: List[_GeneralVarData]):
        for ndx, var in enumerate(variables):
            varname = self._symbol_map.getSymbol(var, self._labeler)
            lb, ub, vtype = self._process_domain_and_bounds(var)
            self._maingo_vars.append(
                MaingoVar(name=varname, type=vtype, lb=lb, ub=ub, init=var.value)
            )
            self._pyomo_var_to_solver_var_id_map[id(var)] = len(self._maingo_vars) - 1

    def _add_params(self, params: List[_ParamData]):
        pass

    def _reinit(self):
        saved_config = self.config
        saved_options = self.maingo_options
        saved_update_config = self.update_config
        self.__init__(only_child_vars=self._only_child_vars)
        self.config = saved_config
        self.maingo_options = saved_options
        self.update_config = saved_update_config

    def set_instance(self, model):
        if self._last_results_object is not None:
            self._last_results_object.solution_loader.invalidate()
        if not self.available():
            c = self.__class__
            raise PyomoException(
                f"Solver {c.__module__}.{c.__qualname__} is not available "
                f"({self.available()})."
            )
        self._reinit()
        self._model = model
        if self.use_extensions and cmodel_available:
            self._expr_types = cmodel.PyomoExprTypes()

        if self.config.symbolic_solver_labels:
            self._labeler = TextLabeler()
        else:
            self._labeler = NumericLabeler("x")

        self.add_block(model)
        self._solver_model = SolverModel(
            var_list=self._maingo_vars,
            con_list=self._cons,
            objective=self._objective,
            idmap=self._pyomo_var_to_solver_var_id_map,
        )

    def _add_constraints(self, cons: List[_GeneralConstraintData]):
        self._cons = cons

    def _add_sos_constraints(self, cons: List[_SOSConstraintData]):
        if len(cons) >= 1:
            raise NotImplementedError("MAiNGO does not currently support SOS constraints.")
        pass

    def _remove_constraints(self, cons: List[_GeneralConstraintData]):
        pass

    def _remove_sos_constraints(self, cons: List[_SOSConstraintData]):
        if len(cons) >= 1:
            raise NotImplementedError("MAiNGO does not currently support SOS constraints.")
        pass

    def _remove_variables(self, variables: List[_GeneralVarData]):
        pass

    def _remove_params(self, params: List[_ParamData]):
        pass

    def _update_variables(self, variables: List[_GeneralVarData]):
        pass

    def update_params(self):
        pass

    def _set_objective(self, obj):
        if obj is None:
            raise NotImplementedError(
                "MAiNGO needs a objective. Please set a dummy objective."
            )
        else:
            if not obj.sense in {minimize, maximize}:
                raise ValueError(
                    "Objective sense is not recognized: {0}".format(obj.sense)
                )
            self._objective = obj

    def _postsolve(self, timer: HierarchicalTimer):
        config = self.config

        mprob = self._mymaingo
        status = mprob.get_status()
        results = MAiNGOResults(solver=self)
        results.wallclock_time = mprob.get_wallclock_solution_time()
        results.cpu_time = mprob.get_cpu_solution_time()

        if status in {maingopy.GLOBALLY_OPTIMAL, maingopy.FEASIBLE_POINT}:
            results.termination_condition = TerminationCondition.optimal
            if status == maingopy.FEASIBLE_POINT:
                logger.warning("MAiNGO did only find a feasible solution but did not prove its global optimality.")
        elif status == maingopy.INFEASIBLE:
            results.termination_condition = TerminationCondition.infeasible
        else:
            results.termination_condition = TerminationCondition.unknown

        results.best_feasible_objective = None
        results.best_objective_bound = None
        if self._objective is not None:
            try:
                if self._objective.sense == maximize:
                    results.best_feasible_objective = -mprob.get_objective_value()
                else:
                    results.best_feasible_objective = mprob.get_objective_value()
            except:
                results.best_feasible_objective = None
            try:
                if self._objective.sense == maximize:
                    results.best_objective_bound = -mprob.get_final_LBD()
                else:
                    results.best_objective_bound = mprob.get_final_LBD()
            except:
                if self._objective.sense == maximize:
                    results.best_objective_bound = math.inf
                else:
                    results.best_objective_bound = -math.inf

            if results.best_feasible_objective is not None and not math.isfinite(
                results.best_feasible_objective
            ):
                results.best_feasible_objective = None

        timer.start("load solution")
        if config.load_solution:
            if not results.best_feasible_objective is None:
                if results.termination_condition != TerminationCondition.optimal:
                    logger.warning(
                        "Loading a feasible but suboptimal solution. "
                        "Please set load_solution=False and check "
                        "results.termination_condition and "
                        "results.found_feasible_solution() before loading a solution."
                    )
                self.load_vars()
            else:
                raise RuntimeError(
                    "A feasible solution was not found, so no solution can be loaded."
                    "Please set opt.config.load_solution=False and check "
                    "results.termination_condition and "
                    "results.best_feasible_objective before loading a solution."
                )
        timer.stop("load solution")

        return results

    def load_vars(self, vars_to_load=None):
        for v, val in self.get_primals(vars_to_load=vars_to_load).items():
            v.set_value(val, skip_validation=True)
        StaleFlagManager.mark_all_as_stale(delayed=True)

    def get_primals(self, vars_to_load=None):
        if not self._mymaingo.get_status() in {
            maingopy.GLOBALLY_OPTIMAL,
            maingopy.FEASIBLE_POINT,
        }:
            raise RuntimeError(
                "Solver does not currently have a valid solution."
                "Please check the termination condition."
            )

        var_id_map = self._pyomo_var_to_solver_var_id_map
        ref_vars = self._referenced_variables
        if vars_to_load is None:
            vars_to_load = var_id_map.keys()
        else:
            vars_to_load = [id(v) for v in vars_to_load]

        maingo_var_ids_to_load = [
            var_id_map[pyomo_var_id] for pyomo_var_id in vars_to_load
        ]

        solution_point = self._mymaingo.get_solution_point()
        vals = [solution_point[var_id] for var_id in maingo_var_ids_to_load]

        res = ComponentMap()
        for var_id, val in zip(vars_to_load, vals):
            using_cons, using_sos, using_obj = ref_vars[var_id]
            if using_cons or using_sos or (using_obj is not None):
                res[self._vars[var_id][0]] = val
        return res

    def get_reduced_costs(self, vars_to_load=None):
        raise ValueError("MAiNGO does not support returning Reduced Costs")

    def get_duals(self, cons_to_load=None):
        raise ValueError("MAiNGO does not support returning Duals")
