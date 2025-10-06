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

from __future__ import annotations
import datetime
import io
import logging
import math
from typing import Tuple, List, Optional, Sequence, Mapping, Dict

from pyomo.common.collections import ComponentMap
from pyomo.core.expr.numvalue import is_constant
from pyomo.common.numeric_types import native_numeric_types
from pyomo.common.errors import InfeasibleConstraintException, ApplicationError
from pyomo.common.timing import HierarchicalTimer
from pyomo.core.base.block import BlockData
from pyomo.core.base.var import VarData, ScalarVar
from pyomo.core.base.param import ParamData, ScalarParam
from pyomo.core.base.constraint import Constraint, ConstraintData
from pyomo.core.base.objective import ObjectiveData
from pyomo.core.base.sos import SOSConstraint, SOSConstraintData
from pyomo.core.kernel.objective import minimize, maximize
from pyomo.core.expr.numeric_expr import (
    NegationExpression,
    PowExpression,
    ProductExpression,
    MonomialTermExpression,
    DivisionExpression,
    SumExpression,
    LinearExpression,
    UnaryFunctionExpression,
    NPV_NegationExpression,
    NPV_PowExpression,
    NPV_ProductExpression,
    NPV_DivisionExpression,
    NPV_SumExpression,
    NPV_UnaryFunctionExpression,
)
from pyomo.core.expr.numvalue import NumericConstant
from pyomo.gdp.disjunct import AutoLinkedBinaryVar
from pyomo.core.base.expression import ExpressionData, ScalarExpression
from pyomo.core.expr.relational_expr import (
    EqualityExpression,
    InequalityExpression,
    RangedExpression,
)
from pyomo.core.staleflag import StaleFlagManager
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor
from pyomo.common.dependencies import attempt_import
from pyomo.contrib.solver.common.base import (
    SolverBase,
    Availability,
    PersistentSolverBase,
)
from pyomo.contrib.solver.common.config import BranchAndBoundConfig
from pyomo.contrib.solver.common.util import (
    NoFeasibleSolutionError,
    NoOptimalSolutionError,
    NoSolutionError,
)
from pyomo.contrib.solver.common.util import get_objective
from pyomo.contrib.solver.common.solution_loader import NoSolutionSolutionLoader
from pyomo.contrib.solver.common.results import (
    Results,
    SolutionStatus,
    TerminationCondition,
)
from pyomo.contrib.solver.common.solution_loader import (
    SolutionLoaderBase,
    load_import_suffixes,
)
from pyomo.common.config import ConfigValue
from pyomo.common.tee import capture_output, TeeStream
from pyomo.core.base.units_container import _PyomoUnit
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
from pyomo.contrib.observer.model_observer import (
    Observer,
    ModelChangeDetector,
    AutoUpdateConfig,
)


logger = logging.getLogger(__name__)


scip, scip_available = attempt_import('pyscipopt')


class ScipConfig(BranchAndBoundConfig):
    def __init__(
        self,
        description=None,
        doc=None,
        implicit=False,
        implicit_domain=None,
        visibility=0,
    ):
        BranchAndBoundConfig.__init__(
            self,
            description=description,
            doc=doc,
            implicit=implicit,
            implicit_domain=implicit_domain,
            visibility=visibility,
        )
        self.use_mipstart: bool = self.declare(
            'use_mipstart',
            ConfigValue(
                default=False,
                domain=bool,
                description="If True, the current values of the integer variables "
                "will be passed to Scip.",
            ),
        )


def _handle_var(node, data, opt, visitor):
    if id(node) not in opt._pyomo_var_to_solver_var_map:
        scip_var = opt._add_var(node)
    else:
        scip_var = opt._pyomo_var_to_solver_var_map[id(node)]
    return scip_var


def _handle_param(node, data, opt, visitor):
    # for the persistent interface, we create scip variables in place
    # of parameters. However, this makes things complicated for range
    # constraints because scip does not allow variables in the
    # lower and upper parts of range constraints
    if visitor.in_range:
        return node.value
    if not opt.is_persistent():
        return node.value
    if not node.mutable:
        return node.value
    if id(node) not in opt._pyomo_param_to_solver_param_map:
        scip_param = opt._add_param(node)
    else:
        scip_param = opt._pyomo_param_to_solver_param_map[id(node)]
    return scip_param


def _handle_constant(node, data, opt, visitor):
    return node.value


def _handle_float(node, data, opt, visitor):
    return float(node)


def _handle_negation(node, data, opt, visitor):
    return -data[0]


def _handle_pow(node, data, opt, visitor):
    x, y = data  # x ** y = exp(log(x**y)) = exp(y*log(x))
    if is_constant(node.args[1]):
        return x**y
    else:
        xlb, xub = compute_bounds_on_expr(node.args[0])
        if xlb > 0:
            return scip.exp(y * scip.log(x))
        else:
            return x**y  # scip will probably raise an error here


def _handle_product(node, data, opt, visitor):
    assert len(data) == 2
    return data[0] * data[1]


def _handle_division(node, data, opt, visitor):
    return data[0] / data[1]


def _handle_sum(node, data, opt, visitor):
    return sum(data)


def _handle_exp(node, data, opt, visitor):
    return scip.exp(data[0])


def _handle_log(node, data, opt, visitor):
    return scip.log(data[0])


def _handle_log10(node, data, opt, visitor):
    return scip.log(data[0]) / math.log(10)


def _handle_sin(node, data, opt, visitor):
    return scip.sin(data[0])


def _handle_cos(node, data, opt, visitor):
    return scip.cos(data[0])


def _handle_sqrt(node, data, opt, visitor):
    return scip.sqrt(data[0])


def _handle_abs(node, data, opt, visitor):
    return abs(data[0])


def _handle_tan(node, data, opt, visitor):
    return scip.sin(data[0]) / scip.cos(data[0])


def _handle_tanh(node, data, opt, visitor):
    x = data[0]
    _exp = scip.exp
    return (_exp(x) - _exp(-x)) / (_exp(x) + _exp(-x))


_unary_map = {
    'exp': _handle_exp,
    'log': _handle_log,
    'sin': _handle_sin,
    'cos': _handle_cos,
    'sqrt': _handle_sqrt,
    'abs': _handle_abs,
    'tan': _handle_tan,
    'log10': _handle_log10,
    'tanh': _handle_tanh,
}


def _handle_unary(node, data, opt, visitor):
    if node.getname() in _unary_map:
        return _unary_map[node.getname()](node, data, opt, visitor)
    else:
        raise NotImplementedError(f'unable to handle unary expression: {str(node)}')


def _handle_equality(node, data, opt, visitor):
    return data[0] == data[1]


def _handle_ranged(node, data, opt, visitor):
    # note that the lower and upper parts of the
    # range constraint cannot have variables
    return data[0] <= (data[1] <= data[2])


def _handle_inequality(node, data, opt, visitor):
    return data[0] <= data[1]


def _handle_named_expression(node, data, opt, visitor):
    return data[0]


def _handle_unit(node, data, opt, visitor):
    return node.value


_operator_map = {
    NegationExpression: _handle_negation,
    PowExpression: _handle_pow,
    ProductExpression: _handle_product,
    MonomialTermExpression: _handle_product,
    DivisionExpression: _handle_division,
    SumExpression: _handle_sum,
    LinearExpression: _handle_sum,
    UnaryFunctionExpression: _handle_unary,
    NPV_NegationExpression: _handle_negation,
    NPV_PowExpression: _handle_pow,
    NPV_ProductExpression: _handle_product,
    NPV_DivisionExpression: _handle_division,
    NPV_SumExpression: _handle_sum,
    NPV_UnaryFunctionExpression: _handle_unary,
    EqualityExpression: _handle_equality,
    RangedExpression: _handle_ranged,
    InequalityExpression: _handle_inequality,
    ScalarExpression: _handle_named_expression,
    ExpressionData: _handle_named_expression,
    VarData: _handle_var,
    ScalarVar: _handle_var,
    ParamData: _handle_param,
    ScalarParam: _handle_param,
    float: _handle_float,
    int: _handle_float,
    AutoLinkedBinaryVar: _handle_var,
    _PyomoUnit: _handle_unit,
    NumericConstant: _handle_constant,
}


class _PyomoToScipVisitor(StreamBasedExpressionVisitor):
    def __init__(self, solver, **kwds):
        super().__init__(**kwds)
        self.solver = solver
        self.in_range = False

    def initializeWalker(self, expr):
        self.in_range = False
        return True, None

    def exitNode(self, node, data):
        nt = type(node)
        if nt in _operator_map:
            return _operator_map[nt](node, data, self.solver, self)
        elif nt in native_numeric_types:
            _operator_map[nt] = _handle_float
            return _handle_float(node, data, self.solver, self)
        else:
            raise NotImplementedError(f'unrecognized expression type: {nt}')

    def enterNode(self, node):
        if type(node) is RangedExpression:
            self.in_range = True
        return None, []


logger = logging.getLogger("pyomo.solvers")


class ScipDirectSolutionLoader(SolutionLoaderBase):
    def __init__(
        self, solver_model, var_id_map, var_map, con_map, pyomo_model, opt
    ) -> None:
        super().__init__()
        self._solver_model = solver_model
        self._vars = var_id_map
        self._var_map = var_map
        self._con_map = con_map
        self._pyomo_model = pyomo_model
        # make sure the scip model does not get freed until the solution loader is garbage collected
        self._opt = opt

    def get_number_of_solutions(self) -> int:
        return self._solver_model.getNSols()

    def get_solution_ids(self) -> List:
        return list(range(self.get_number_of_solutions()))

    def load_vars(
        self, vars_to_load: Optional[Sequence[VarData]] = None, solution_id=None
    ) -> None:
        for v, val in self.get_vars(
            vars_to_load=vars_to_load, solution_id=solution_id
        ).items():
            v.value = val

    def get_vars(
        self, vars_to_load: Optional[Sequence[VarData]] = None, solution_id=None
    ) -> Mapping[VarData, float]:
        if self.get_number_of_solutions() == 0:
            raise NoSolutionError()
        if vars_to_load is None:
            vars_to_load = list(self._vars.values())
        if solution_id is None:
            solution_id = 0
        sol = self._solver_model.getSols()[solution_id]
        res = ComponentMap()
        for v in vars_to_load:
            sv = self._var_map[id(v)]
            res[v] = sol[sv]
        return res

    def get_reduced_costs(
        self, vars_to_load: Optional[Sequence[VarData]] = None, solution_id=None
    ) -> Mapping[VarData, float]:
        return NotImplemented

    def get_duals(
        self, cons_to_load: Optional[Sequence[ConstraintData]] = None, solution_id=None
    ) -> Dict[ConstraintData, float]:
        return NotImplemented

    def load_import_suffixes(self, solution_id=None):
        load_import_suffixes(self._pyomo_model, self, solution_id=solution_id)


class ScipPersistentSolutionLoader(ScipDirectSolutionLoader):
    def __init__(
        self, solver_model, var_id_map, var_map, con_map, pyomo_model, opt
    ) -> None:
        super().__init__(solver_model, var_id_map, var_map, con_map, pyomo_model, opt)
        self._valid = True

    def invalidate(self):
        self._valid = False

    def _assert_solution_still_valid(self):
        if not self._valid:
            raise RuntimeError('The results in the solver are no longer valid.')

    def load_vars(
        self, vars_to_load: Sequence[VarData] | None = None, solution_id=None
    ) -> None:
        self._assert_solution_still_valid()
        return super().load_vars(vars_to_load, solution_id)

    def get_vars(
        self, vars_to_load: Sequence[VarData] | None = None, solution_id=None
    ) -> Mapping[VarData, float]:
        self._assert_solution_still_valid()
        return super().get_vars(vars_to_load, solution_id)

    def get_duals(
        self, cons_to_load: Sequence[ConstraintData] | None = None, solution_id=None
    ) -> Dict[ConstraintData, float]:
        self._assert_solution_still_valid()
        return super().get_duals(cons_to_load)

    def get_reduced_costs(
        self, vars_to_load: Sequence[VarData] | None = None, solution_id=None
    ) -> Mapping[VarData, float]:
        self._assert_solution_still_valid()
        return super().get_reduced_costs(vars_to_load)

    def get_number_of_solutions(self) -> int:
        self._assert_solution_still_valid()
        return super().get_number_of_solutions()

    def get_solution_ids(self) -> List:
        self._assert_solution_still_valid()
        return super().get_solution_ids()

    def load_import_suffixes(self, solution_id=None):
        self._assert_solution_still_valid()
        super().load_import_suffixes(solution_id)


class ScipDirect(SolverBase):

    _available = None
    _tc_map = None
    _minimum_version = (5, 5, 0)  # this is probably conservative

    CONFIG = ScipConfig()

    def __init__(self, **kwds):
        super().__init__(**kwds)
        self._solver_model = None
        self._vars = {}  # var id to var
        self._params = {}  # param id to param
        self._pyomo_var_to_solver_var_map = {}  # var id to scip var
        self._pyomo_con_to_solver_con_map = {}
        self._pyomo_param_to_solver_param_map = (
            {}
        )  # param id to scip var with equal bounds
        self._pyomo_sos_to_solver_sos_map = {}
        self._expr_visitor = _PyomoToScipVisitor(self)
        self._objective = None  # pyomo objective
        self._obj_var = (
            None  # a scip variable because the objective cannot be nonlinear
        )
        self._obj_con = None  # a scip constraint (obj_var >= obj_expr)

    def _clear(self):
        self._solver_model = None
        self._vars = {}
        self._params = {}
        self._pyomo_var_to_solver_var_map = {}
        self._pyomo_con_to_solver_con_map = {}
        self._pyomo_param_to_solver_param_map = {}
        self._pyomo_sos_to_solver_sos_map = {}
        self._objective = None
        self._obj_var = None
        self._obj_con = None

    def available(self) -> Availability:
        if self._available is not None:
            return self._available

        if not scip_available:
            ScipDirect._available = Availability.NotFound
        elif self.version() < self._minimum_version:
            ScipDirect._available = Availability.BadVersion
        else:
            ScipDirect._available = Availability.FullLicense

        return self._available

    def version(self) -> Tuple:
        return tuple(int(i) for i in scip.__version__.split('.'))

    def solve(self, model: BlockData, **kwds) -> Results:
        start_timestamp = datetime.datetime.now(datetime.timezone.utc)
        orig_config = self.config
        if not self.available():
            raise ApplicationError(f'{self.name} is not available: {self.available()}')
        try:
            config = self.config(value=kwds, preserve_implicit=True)

            # hack to work around legacy solver wrapper __setattr__
            # otherwise, this would just be self.config = config
            object.__setattr__(self, 'config', config)

            StaleFlagManager.mark_all_as_stale()

            if config.timer is None:
                config.timer = HierarchicalTimer()
            timer = config.timer

            ostreams = [io.StringIO()] + config.tee

            scip_model, solution_loader, has_obj = self._create_solver_model(model)

            scip_model.hideOutput(quiet=False)
            if config.threads is not None:
                scip_model.setParam('lp/threads', config.threads)
            if config.time_limit is not None:
                scip_model.setParam('limits/time', config.time_limit)
            if config.rel_gap is not None:
                scip_model.setParam('limits/gap', config.rel_gap)
            if config.abs_gap is not None:
                scip_model.setParam('limits/absgap', config.abs_gap)

            if config.use_mipstart:
                self._mipstart()

            for key, option in config.solver_options.items():
                scip_model.setParam(key, option)

            timer.start('optimize')
            with capture_output(TeeStream(*ostreams), capture_fd=True):
                # scip_model.writeProblem(filename='foo.lp')
                scip_model.optimize()
            timer.stop('optimize')

            results = self._postsolve(scip_model, solution_loader, has_obj)
        except InfeasibleConstraintException:
            results = self._get_infeasible_results()
        finally:
            # hack to work around legacy solver wrapper __setattr__
            # otherwise, this would just be self.config = orig_config
            object.__setattr__(self, 'config', orig_config)

        results.solver_log = ostreams[0].getvalue()
        end_timestamp = datetime.datetime.now(datetime.timezone.utc)
        results.timing_info.start_timestamp = start_timestamp
        results.timing_info.wall_time = (
            end_timestamp - start_timestamp
        ).total_seconds()
        results.timing_info.timer = timer
        return results

    def _get_tc_map(self):
        if ScipDirect._tc_map is None:
            tc = TerminationCondition
            ScipDirect._tc_map = {
                "unknown": tc.unknown,
                "userinterrupt": tc.interrupted,
                "nodelimit": tc.iterationLimit,
                "totalnodelimit": tc.iterationLimit,
                "stallnodelimit": tc.iterationLimit,
                "timelimit": tc.maxTimeLimit,
                "memlimit": tc.unknown,
                "gaplimit": tc.convergenceCriteriaSatisfied,  # TODO: check this
                "primallimit": tc.objectiveLimit,
                "duallimit": tc.objectiveLimit,
                "sollimit": tc.unknown,
                "bestsollimit": tc.unknown,
                "restartlimit": tc.unknown,
                "optimal": tc.convergenceCriteriaSatisfied,
                "infeasible": tc.provenInfeasible,
                "unbounded": tc.unbounded,
                "inforunbd": tc.infeasibleOrUnbounded,
                "terminate": tc.unknown,
            }
        return ScipDirect._tc_map

    def _get_infeasible_results(self):
        res = Results()
        res.solution_loader = NoSolutionSolutionLoader()
        res.solution_status = SolutionStatus.noSolution
        res.termination_condition = TerminationCondition.provenInfeasible
        res.incumbent_objective = None
        res.objective_bound = None
        res.iteration_count = None
        res.timing_info.scip_time = None
        res.solver_config = self.config
        res.solver_name = self.name
        res.solver_version = self.version()
        if self.config.raise_exception_on_nonoptimal_result:
            raise NoOptimalSolutionError()
        if self.config.load_solutions:
            raise NoFeasibleSolutionError()
        return res

    def _scip_lb_ub_from_var(self, var):
        if var.is_fixed():
            val = var.value
            return val, val

        lb, ub = var.bounds

        if lb is None:
            lb = -self._solver_model.infinity()
        if ub is None:
            ub = self._solver_model.infinity()

        return lb, ub

    def _add_var(self, var):
        vtype = self._scip_vtype_from_var(var)
        lb, ub = self._scip_lb_ub_from_var(var)

        scip_var = self._solver_model.addVar(lb=lb, ub=ub, vtype=vtype)

        self._vars[id(var)] = var
        self._pyomo_var_to_solver_var_map[id(var)] = scip_var
        return scip_var

    def _add_param(self, p):
        vtype = "C"
        lb = ub = p.value
        scip_var = self._solver_model.addVar(lb=lb, ub=ub, vtype=vtype)
        self._params[id(p)] = p
        self._pyomo_param_to_solver_param_map[id(p)] = scip_var
        return scip_var

    def __del__(self):
        """Frees SCIP resources used by this solver instance."""
        if self._solver_model is not None:
            self._solver_model.freeProb()
            self._solver_model = None

    def _add_constraints(self, cons: List[ConstraintData]):
        for con in cons:
            self._add_constraint(con)

    def _add_sos_constraints(self, cons: List[SOSConstraintData]):
        for on in cons:
            self._add_sos_constraint(con)

    def _create_solver_model(self, model):
        timer = self.config.timer
        timer.start('create scip model')
        self._clear()
        self._solver_model = scip.Model()
        timer.start('collect constraints')
        cons = list(
            model.component_data_objects(Constraint, descend_into=True, active=True)
        )
        timer.stop('collect constraints')
        timer.start('translate constraints')
        self._add_constraints(cons)
        timer.stop('translate constraints')
        timer.start('sos')
        sos = list(
            model.component_data_objects(SOSConstraint, descend_into=True, active=True)
        )
        self._add_sos_constraints(sos)
        timer.stop('sos')
        timer.start('get objective')
        obj = get_objective(model)
        timer.stop('get objective')
        timer.start('translate objective')
        self._set_objective(obj)
        timer.stop('translate objective')
        has_obj = obj is not None
        solution_loader = ScipDirectSolutionLoader(
            solver_model=self._solver_model,
            var_id_map=self._vars,
            var_map=self._pyomo_var_to_solver_var_map,
            con_map=self._pyomo_con_to_solver_con_map,
            pyomo_model=model,
            opt=self,
        )
        timer.stop('create scip model')
        return self._solver_model, solution_loader, has_obj

    def _add_constraint(self, con):
        scip_expr = self._expr_visitor.walk_expression(con.expr)
        scip_con = self._solver_model.addCons(scip_expr)
        self._pyomo_con_to_solver_con_map[con] = scip_con

    def _add_sos_constraint(self, con):
        level = con.level
        if level not in [1, 2]:
            raise ValueError(
                f"{self.name} does not support SOS level {level} constraints"
            )

        scip_vars = []
        weights = []

        for v, w in con.get_items():
            vid = id(v)
            if vid not in self._pyomo_var_to_solver_var_map:
                self._add_var(v)
            scip_vars.append(self._pyomo_var_to_solver_var_map[vid])
            weights.append(w)

        if level == 1:
            scip_cons = self._solver_model.addConsSOS1(scip_vars, weights=weights)
        else:
            scip_cons = self._solver_model.addConsSOS2(scip_vars, weights=weights)
        self._pyomo_con_to_solver_con_map[con] = scip_cons

    def _scip_vtype_from_var(self, var):
        """
        This function takes a pyomo variable and returns the appropriate SCIP variable type

        Parameters
        ----------
        var: pyomo.core.base.var.Var
            The pyomo variable that we want to retrieve the SCIP vtype of

        Returns
        -------
        vtype: str
            B for Binary, I for Integer, or C for Continuous
        """
        if var.is_binary():
            vtype = "B"
        elif var.is_integer():
            vtype = "I"
        elif var.is_continuous():
            vtype = "C"
        else:
            raise ValueError(f"Variable domain type is not recognized for {var.domain}")
        return vtype

    def _set_objective(self, obj):
        if self._obj_var is None:
            self._obj_var = self._solver_model.addVar(
                lb=-self._solver_model.infinity(),
                ub=self._solver_model.infinity(),
                vtype="C",
            )

        if self._obj_con is not None:
            self._solver_model.delCons(self._obj_con)

        if obj is None:
            scip_expr = 0
            sense = "minimize"
        else:
            scip_expr = self._expr_visitor.walk_expression(obj.expr)
            if obj.sense == minimize:
                sense = "minimize"
            elif obj.sense == maximize:
                sense = "maximize"
            else:
                raise ValueError(f"Objective sense is not recognized: {obj.sense}")

        if sense == "minimize":
            self._obj_con = self._solver_model.addCons(self._obj_var >= scip_expr)
        else:
            self._obj_con = self._solver_model.addCons(self._obj_var <= scip_expr)

        self._solver_model.setObjective(self._obj_var, sense=sense)
        self._objective = obj

    def _postsolve(
        self, scip_model, solution_loader: ScipDirectSolutionLoader, has_obj
    ):

        results = Results()
        results.solution_loader = solution_loader
        results.timing_info.scip_time = scip_model.getSolvingTime()
        results.termination_condition = self._get_tc_map().get(
            scip_model.getStatus(), TerminationCondition.unknown
        )

        if solution_loader.get_number_of_solutions() > 0:
            if (
                results.termination_condition
                == TerminationCondition.convergenceCriteriaSatisfied
            ):
                results.solution_status = SolutionStatus.optimal
            else:
                results.solution_status = SolutionStatus.feasible
        else:
            results.solution_status = SolutionStatus.noSolution

        if (
            results.termination_condition
            != TerminationCondition.convergenceCriteriaSatisfied
            and self.config.raise_exception_on_nonoptimal_result
        ):
            raise NoOptimalSolutionError()

        if has_obj:
            try:
                if (
                    scip_model.getNSols() > 0
                    and scip_model.getObjVal() < scip_model.infinity()
                ):
                    results.incumbent_objective = scip_model.getObjVal()
                else:
                    results.incumbent_objective = None
            except:
                results.incumbent_objective = None
            try:
                results.objective_bound = scip_model.getDualbound()
                if results.objective_bound <= -scip_model.infinity():
                    results.objective_bound = -math.inf
                if results.objective_bound >= scip_model.infinity():
                    results.objective_bound = math.inf
            except:
                if self._objective.sense == minimize:
                    results.objective_bound = -math.inf
                else:
                    results.objective_bound = math.inf
        else:
            results.incumbent_objective = None
            results.objective_bound = None

        self.config.timer.start('load solution')
        if self.config.load_solutions:
            if solution_loader.get_number_of_solutions() > 0:
                solution_loader.load_solution()
            else:
                raise NoFeasibleSolutionError()
        self.config.timer.stop('load solution')

        results.iteration_count = scip_model.getNNodes()
        results.solver_config = self.config
        results.solver_name = self.name
        results.solver_version = self.version()

        return results

    def _mipstart(self):
        # TODO: it is also possible to specify continuous variables, but
        #       I think we should have a differnt option for that
        sol = self._solver_model.createPartialSol()
        for vid, scip_var in self._pyomo_var_to_solver_var_map.items():
            pyomo_var = self._vars[vid]
            if pyomo_var.is_integer():
                sol[scip_var] = pyomo_var.value
        self._solver_model.addSol(sol)


class ScipPersistentConfig(ScipConfig):
    def __init__(
        self,
        description=None,
        doc=None,
        implicit=False,
        implicit_domain=None,
        visibility=0,
    ):
        ScipConfig.__init__(
            self,
            description=description,
            doc=doc,
            implicit=implicit,
            implicit_domain=implicit_domain,
            visibility=visibility,
        )
        self.auto_updates: bool = self.declare('auto_updates', AutoUpdateConfig())


class _ScipObserver(Observer):
    def __init__(self, opt: ScipPersistent) -> None:
        self.opt = opt

    def add_variables(self, variables: List[VarData]):
        self.opt._add_variables(variables)

    def add_parameters(self, params: List[ParamData]):
        self.opt._add_parameters(params)

    def add_constraints(self, cons: List[ConstraintData]):
        self.opt._add_constraints(cons)

    def add_sos_constraints(self, cons: List[SOSConstraintData]):
        self.opt._add_sos_constraints(cons)

    def add_objectives(self, objs: List[ObjectiveData]):
        self.opt._add_objectives(objs)

    def remove_objectives(self, objs: List[ObjectiveData]):
        self.opt._remove_objectives(objs)

    def remove_constraints(self, cons: List[ConstraintData]):
        self.opt._remove_constraints(cons)

    def remove_sos_constraints(self, cons: List[SOSConstraintData]):
        self.opt._remove_sos_constraints(cons)

    def remove_variables(self, variables: List[VarData]):
        self.opt._remove_variables(variables)

    def remove_parameters(self, params: List[ParamData]):
        self.opt._remove_parameters(params)

    def update_variables(self, variables: List[VarData]):
        self.opt._update_variables(variables)

    def update_parameters(self, params: List[ParamData]):
        self.opt._update_parameters(params)


class ScipPersistent(ScipDirect, PersistentSolverBase):
    _minimum_version = (5, 5, 0)  # this is probably conservative
    CONFIG = ScipPersistentConfig()

    def __init__(self, **kwds):
        super().__init__(**kwds)
        self._pyomo_model = None
        self._observer = None
        self._change_detector = None
        self._last_results_object: Optional[Results] = None
        self._needs_reopt = False
        self._range_constraints = set()

    def _clear(self):
        super()._clear()
        self._pyomo_model = None
        self._objective = None
        self._observer = None
        self._change_detector = None
        self._needs_reopt = False

    def _check_reopt(self):
        if self._needs_reopt:
            # self._solver_model.freeReoptSolve()  # when is it safe to use this one???
            self._solver_model.freeTransform()
            self._needs_reopt = False

    def _create_solver_model(self, pyomo_model):
        if pyomo_model is self._pyomo_model:
            self.update()
        else:
            self.set_instance(pyomo_model=pyomo_model)

        solution_loader = ScipPersistentSolutionLoader(
            solver_model=self._solver_model,
            var_id_map=self._vars,
            var_map=self._pyomo_var_to_solver_var_map,
            con_map=self._pyomo_con_to_solver_con_map,
            pyomo_model=pyomo_model,
            opt=self,
        )

        has_obj = self._objective is not None
        return self._solver_model, solution_loader, has_obj

    def solve(self, model, **kwds) -> Results:
        res = super().solve(model, **kwds)
        self._needs_reopt = True
        return res

    def update(self):
        if self.config.timer is None:
            timer = HierarchicalTimer()
        else:
            timer = self.config.timer
        if self._pyomo_model is None:
            raise RuntimeError('must call set_instance or solve before update')
        timer.start('update')
        self._change_detector.update(timer=timer)
        timer.stop('update')

    def set_instance(self, pyomo_model):
        if self.config.timer is None:
            timer = HierarchicalTimer()
        else:
            timer = self.config.timer
        self._clear()
        self._pyomo_model = pyomo_model
        self._solver_model = scip.Model()
        self._observer = _ScipObserver(self)
        timer.start('set_instance')
        self._change_detector = ModelChangeDetector(
            model=self._pyomo_model,
            observers=[self._observer],
            **dict(self.config.auto_updates),
        )
        self._change_detector.config = self.config.auto_updates
        timer.stop('set_instance')

    def _invalidate_last_results(self):
        if self._last_results_object is not None:
            self._last_results_object.solution_loader.invalidate()

    def _add_variables(self, variables: List[VarData]):
        self._check_reopt()
        self._invalidate_last_results()
        for v in variables:
            self._add_var(v)

    def _add_parameters(self, params: List[ParamData]):
        self._check_reopt()
        self._invalidate_last_results()
        for p in params:
            self._add_param(p)

    def _add_constraints(self, cons: List[ConstraintData]):
        self._check_reopt()
        self._invalidate_last_results()
        for con in cons:
            if type(con.expr) is RangedExpression:
                self._range_constraints.add(con)
        super()._add_constraints(cons)

    def _add_sos_constraints(self, cons: List[SOSConstraintData]):
        self._check_reopt()
        self._invalidate_last_results()
        return super()._add_sos_constraints(cons)

    def _add_objectives(self, objs: List[ObjectiveData]):
        self._check_reopt()
        if len(objs) > 1:
            raise NotImplementedError(
                'the persistent interface to gurobi currently '
                f'only supports single-objective problems; got {len(objs)}: '
                f'{[str(i) for i in objs]}'
            )

        if len(objs) == 0:
            return

        obj = objs[0]

        if self._objective is not None:
            raise NotImplementedError(
                'the persistent interface to gurobi currently '
                'only supports single-objective problems; tried to add '
                f'an objective ({str(obj)}), but there is already an '
                f'active objective ({str(self._objective)})'
            )

        self._invalidate_last_results()
        self._set_objective(obj)

    def _remove_objectives(self, objs: List[ObjectiveData]):
        self._check_reopt()
        for obj in objs:
            if obj is not self._objective:
                raise RuntimeError(
                    'tried to remove an objective that has not been added: '
                    f'{str(obj)}'
                )
            else:
                self._invalidate_last_results()
                self._set_objective(None)

    def _remove_constraints(self, cons: List[ConstraintData]):
        self._check_reopt()
        self._invalidate_last_results()
        for con in cons:
            scip_con = self._pyomo_con_to_solver_con_map.pop(con)
            self._solver_model.delCons(scip_con)
            self._range_constraints.discard(con)

    def _remove_sos_constraints(self, cons: List[SOSConstraintData]):
        self._check_reopt()
        self._invalidate_last_results()
        for con in cons:
            scip_con = self._pyomo_con_to_solver_con_map.pop(con)
            self._solver_model.delCons(scip_con)

    def _remove_variables(self, variables: List[VarData]):
        self._check_reopt()
        self._invalidate_last_results()
        for v in variables:
            vid = id(v)
            scip_var = self._pyomo_var_to_solver_var_map.pop(vid)
            self._solver_model.delVar(scip_var)
            self._vars.pop(vid)

    def _remove_parameters(self, params: List[ParamData]):
        self._check_reopt()
        self._invalidate_last_results()
        for p in params:
            pid = id(p)
            scip_var = self._pyomo_param_to_solver_param_map.pop(pid)
            self._solver_model.delVar(scip_var)
            self._params.pop(pid)

    def _update_variables(self, variables: List[VarData]):
        self._check_reopt()
        self._invalidate_last_results()
        for v in variables:
            vid = id(v)
            scip_var = self._pyomo_var_to_solver_var_map[vid]
            vtype = self._scip_vtype_from_var(v)
            lb, ub = self._scip_lb_ub_from_var(v)
            self._solver_model.chgVarLb(scip_var, lb)
            self._solver_model.chgVarUb(scip_var, ub)
            self._solver_model.chgVarType(scip_var, vtype)

    def _update_parameters(self, params: List[ParamData]):
        self._check_reopt()
        self._invalidate_last_results()
        for p in params:
            pid = id(p)
            scip_var = self._pyomo_param_to_solver_param_map[pid]
            lb = ub = p.value
            self._solver_model.chgVarLb(scip_var, lb)
            self._solver_model.chgVarUb(scip_var, ub)
            impacted_vars = self._change_detector.get_variables_impacted_by_param(p)
            if impacted_vars:
                self._update_variables(impacted_vars)
            impacted_cons = self._change_detector.get_constraints_impacted_by_param(p)
            for con in impacted_cons:
                if con in self._range_constraints:
                    self._remove_constraints([con])
                    self._add_constraints([con])

    def add_variables(self, variables):
        if self._change_detector is None:
            raise RuntimeError('call set_instance first')
        self._change_detector.add_variables(variables)

    def add_constraints(self, cons):
        if self._change_detector is None:
            raise RuntimeError('call set_instance first')
        self._change_detector.add_constraints(cons)

    def add_sos_constraints(self, cons):
        if self._change_detector is None:
            raise RuntimeError('call set_instance first')
        self._change_detector.add_sos_constraints(cons)

    def set_objective(self, obj: ObjectiveData):
        if self._change_detector is None:
            raise RuntimeError('call set_instance first')
        self._change_detector.add_objectives([obj])

    def remove_constraints(self, cons):
        if self._change_detector is None:
            raise RuntimeError('call set_instance first')
        self._change_detector.remove_constraints(cons)

    def remove_sos_constraints(self, cons):
        if self._change_detector is None:
            raise RuntimeError('call set_instance first')
        self._change_detector.remove_sos_constraints(cons)

    def remove_variables(self, variables):
        if self._change_detector is None:
            raise RuntimeError('call set_instance first')
        self._change_detector.remove_variables(variables)

    def update_variables(self, variables):
        if self._change_detector is None:
            raise RuntimeError('call set_instance first')
        self._change_detector.update_variables(variables)

    def update_parameters(self, params):
        if self._change_detector is None:
            raise RuntimeError('call set_instance first')
        self._change_detector.update_parameters(params)
