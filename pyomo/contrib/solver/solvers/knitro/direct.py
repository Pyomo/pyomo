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

from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence
from typing import Any, List, Optional, Tuple

from pyomo.common.collections.component_map import ComponentMap
from pyomo.common.errors import ApplicationError
from pyomo.common.flags import NOTSET
from pyomo.common.numeric_types import value
from pyomo.common.tee import TeeStream, capture_output
from pyomo.common.timing import HierarchicalTimer
from pyomo.contrib.solver.common.base import Availability, SolverBase
from pyomo.contrib.solver.common.results import (
    Results,
    SolutionStatus,
    TerminationCondition,
)
from pyomo.contrib.solver.common.solution_loader import SolutionLoaderBase
from pyomo.contrib.solver.common.util import (
    IncompatibleModelError,
    NoDualsError,
    NoOptimalSolutionError,
    NoSolutionError,
    collect_vars_and_named_exprs,
)
from pyomo.core.base.block import BlockData
from pyomo.core.base.constraint import Constraint, ConstraintData
from pyomo.core.base.objective import Objective, ObjectiveData
from pyomo.core.base.var import VarData
from pyomo.core.plugins.transform.util import partial
from pyomo.core.staleflag import StaleFlagManager
from pyomo.repn.standard_repn import StandardRepn, generate_standard_repn

from .api import knitro, KNITRO_AVAILABLE, KNITRO_VERSION
from .config import KnitroConfig


def get_active_objectives(block: BlockData) -> List[ObjectiveData]:
    generator = block.component_data_objects(
        Objective, descend_into=True, active=True, sort=True
    )
    return list(generator)


def get_active_constraints(block: BlockData) -> List[ConstraintData]:
    generator = block.component_data_objects(
        Constraint, descend_into=True, active=True, sort=True
    )
    return list(generator)


def get_solution_status(status: int) -> SolutionStatus:
    if (
        status == knitro.KN_RC_OPTIMAL
        or status == knitro.KN_RC_OPTIMAL_OR_SATISFACTORY
        or status == knitro.KN_RC_NEAR_OPT
    ):
        return SolutionStatus.optimal
    elif status == knitro.KN_RC_FEAS_NO_IMPROVE:
        return SolutionStatus.feasible
    elif (
        status == knitro.KN_RC_INFEASIBLE
        or status == knitro.KN_RC_INFEAS_CON_BOUNDS
        or status == knitro.KN_RC_INFEAS_VAR_BOUNDS
        or status == knitro.KN_RC_INFEAS_NO_IMPROVE
    ):
        return SolutionStatus.infeasible
    else:
        return SolutionStatus.noSolution


def get_termination_condition(status: int) -> TerminationCondition:
    if (
        status == knitro.KN_RC_OPTIMAL
        or status == knitro.KN_RC_OPTIMAL_OR_SATISFACTORY
        or status == knitro.KN_RC_NEAR_OPT
    ):
        return TerminationCondition.convergenceCriteriaSatisfied
    elif status == knitro.KN_RC_INFEAS_NO_IMPROVE:
        return TerminationCondition.locallyInfeasible
    elif status == knitro.KN_RC_INFEASIBLE:
        return TerminationCondition.provenInfeasible
    elif status == knitro.KN_RC_UNBOUNDED_OR_INFEAS or status == knitro.KN_RC_UNBOUNDED:
        return TerminationCondition.infeasibleOrUnbounded
    elif (
        status == knitro.KN_RC_ITER_LIMIT_FEAS
        or status == knitro.KN_RC_ITER_LIMIT_INFEAS
    ):
        return TerminationCondition.iterationLimit
    elif (
        status == knitro.KN_RC_TIME_LIMIT_FEAS
        or status == knitro.KN_RC_TIME_LIMIT_INFEAS
    ):
        return TerminationCondition.maxTimeLimit
    elif status == knitro.KN_RC_USER_TERMINATION:
        return TerminationCondition.interrupted
    else:
        return TerminationCondition.unknown


class ModelRepresentation:
    """An intermediate representation of a Pyomo model.

    This class aggregates the objectives, constraints, and all referenced variables.
    """

    objs: List[ObjectiveData]
    cons: List[ConstraintData]
    variables: List[VarData]

    def __init__(self, objs: Iterable[ObjectiveData], cons: Iterable[ConstraintData]):
        self.objs = list(objs)
        self.cons = list(cons)

        # Collect all referenced variables using a dictionary to ensure uniqueness.
        var_map = {}
        for obj in self.objs:
            _, variables, _, _ = collect_vars_and_named_exprs(obj.expr)
            for v in variables:
                var_map[id(v)] = v
        for con in self.cons:
            _, variables, _, _ = collect_vars_and_named_exprs(con.body)
            for v in variables:
                var_map[id(v)] = v
        self.variables = list(var_map.values())


def build_model_representation(block: BlockData) -> ModelRepresentation:
    """Builds an intermediate representation from a Pyomo model block."""
    objs = get_active_objectives(block)
    cons = get_active_constraints(block)
    return ModelRepresentation(objs=objs, cons=cons)


class NLExpression:
    """Holds the data required to evaluate a non-linear expression."""

    body: Optional[Any]
    variables: List[VarData]

    def __init__(self, expr: Optional[Any], variables: Iterable[VarData]):
        self.body = expr
        self.variables = list(variables)

    def create_evaluator(self, vmap: Mapping[int, int]):
        def _fn(x: List[float]) -> float:
            # Set the values of the Pyomo variables from the solver's vector `x`
            for var in self.variables:
                i = vmap[id(var)]
                var.set_value(x[i])
            return value(self.body)

        return _fn


class KnitroLicenseManager:
    """Manages the global KNITRO license context."""

    _lmc = None

    @staticmethod
    def initialize():
        if KnitroLicenseManager._lmc is None:
            KnitroLicenseManager._lmc = knitro.KN_checkout_license()
        return KnitroLicenseManager._lmc

    @staticmethod
    def release():
        if KnitroLicenseManager._lmc is not None:
            knitro.KN_release_license(KnitroLicenseManager._lmc)
            KnitroLicenseManager._lmc = None

    @staticmethod
    def create_new_context():
        lmc = KnitroLicenseManager.initialize()
        return knitro.KN_new_lm(lmc)

    @staticmethod
    def version() -> Tuple[int, int, int]:
        return tuple(int(x) for x in KNITRO_VERSION.split("."))

    @staticmethod
    def available() -> Availability:
        if not KNITRO_AVAILABLE:
            return Availability.NotFound
        try:
            stream = io.StringIO()
            with capture_output(TeeStream(stream), capture_fd=1):
                kc = KnitroLicenseManager.create_new_context()
                knitro.KN_free(kc)
            # TODO: parse the stream to check the license type.
            return Availability.FullLicense
        except Exception:
            return Availability.BadLicense


class KnitroProblemContext:
    """
    A wrapper around the KNITRO API for a single optimization problem.

    This class manages the lifecycle of a KNITRO problem instance (`kc`),
    including building the problem by adding variables and constraints,
    setting options, solving, and freeing the context.
    """

    var_map: MutableMapping[int, int]
    con_map: MutableMapping[int, int]
    obj_nl_expr: Optional[NLExpression]
    con_nl_expr_map: MutableMapping[int, NLExpression]

    def __init__(self):
        self._kc = KnitroLicenseManager.create_new_context()
        self.var_map = {}
        self.con_map = {}
        self.obj_nl_expr = None
        self.con_nl_expr_map = {}

    def __del__(self):
        self.close()

    def _execute(self, api_fn, *args, **kwargs):
        if self._kc is None:
            raise RuntimeError("KNITRO context has been freed and cannot be used.")
        return api_fn(self._kc, *args, **kwargs)

    def close(self):
        if self._kc is not None:
            self._execute(knitro.KN_free)
            self._kc = None

    def add_vars(self, variables: Iterable[VarData]):
        n_vars = len(variables)
        idx_vars = self._execute(knitro.KN_add_vars, n_vars)
        if idx_vars is None:
            return

        for i, var in zip(idx_vars, variables):
            self.var_map[id(var)] = i

        var_types, fxbnds, lobnds, upbnds = {}, {}, {}, {}
        for var in variables:
            i = self.var_map[id(var)]
            if var.is_binary():
                var_types[i] = knitro.KN_VARTYPE_BINARY
            elif var.is_integer():
                var_types[i] = knitro.KN_VARTYPE_INTEGER
            elif not var.is_continuous():
                msg = f"Unknown variable type for variable {var.name}."
                raise ValueError(msg)

            if var.fixed:
                fxbnds[i] = value(var.value)
            else:
                if var.has_lb():
                    lobnds[i] = value(var.lb)
                if var.has_ub():
                    upbnds[i] = value(var.ub)

        self._execute(knitro.KN_set_var_types, var_types.keys(), var_types.values())
        self._execute(knitro.KN_set_var_fxbnds, fxbnds.keys(), fxbnds.values())
        self._execute(knitro.KN_set_var_lobnds, lobnds.keys(), lobnds.values())
        self._execute(knitro.KN_set_var_upbnds, upbnds.keys(), upbnds.values())

    def _add_expr_structs_from_repn(
        self,
        repn: StandardRepn,
        add_const_fn: Callable[[float], None],
        add_lin_fn: Callable[[Iterable[int], Iterable[float]], None],
        add_quad_fn: Callable[[Iterable[int], Iterable[int], Iterable[float]], None],
    ):
        if repn.constant is not None:
            add_const_fn(repn.constant)
        if repn.linear_vars:
            idx_lin_vars = [self.var_map.get(id(v)) for v in repn.linear_vars]
            add_lin_fn(idx_lin_vars, list(repn.linear_coefs))
        if repn.quadratic_vars:
            quad_vars1, quad_vars2 = zip(*repn.quadratic_vars)
            idx_quad_vars1 = [self.var_map.get(id(v)) for v in quad_vars1]
            idx_quad_vars2 = [self.var_map.get(id(v)) for v in quad_vars2]
            add_quad_fn(idx_quad_vars1, idx_quad_vars2, list(repn.quadratic_coefs))

    def add_cons(self, cons: Iterable[ConstraintData]):
        n_cons = len(cons)
        idx_cons = self._execute(knitro.KN_add_cons, n_cons)
        if idx_cons is None:
            return

        for i, con in zip(idx_cons, cons):
            self.con_map[id(con)] = i

        eqbnds, lobnds, upbnds = {}, {}, {}
        for con in cons:
            i = self.con_map[id(con)]
            if con.equality:
                eqbnds[i] = value(con.lower)
            else:
                if con.has_lb():
                    lobnds[i] = value(con.lb)
                if con.has_ub():
                    upbnds[i] = value(con.ub)

        self._execute(knitro.KN_set_con_eqbnds, eqbnds.keys(), eqbnds.values())
        self._execute(knitro.KN_set_con_lobnds, lobnds.keys(), lobnds.values())
        self._execute(knitro.KN_set_con_upbnds, upbnds.keys(), upbnds.values())

        for con in cons:
            i = self.con_map[id(con)]
            repn = generate_standard_repn(con.body)
            self._add_expr_structs_from_repn(
                repn,
                add_const_fn=partial(self._execute, knitro.KN_add_con_constants, i),
                add_lin_fn=partial(self._execute, knitro.KN_add_con_linear_struct, i),
                add_quad_fn=partial(
                    self._execute, knitro.KN_add_con_quadratic_struct, i
                ),
            )
            if repn.nonlinear_expr is not None:
                self.con_nl_expr_map[i] = NLExpression(
                    repn.nonlinear_expr, repn.nonlinear_vars
                )

    def set_obj(self, obj: ObjectiveData):
        obj_goal = (
            knitro.KN_OBJGOAL_MINIMIZE
            if obj.is_minimizing()
            else knitro.KN_OBJGOAL_MAXIMIZE
        )
        self._execute(knitro.KN_set_obj_goal, obj_goal)
        repn = generate_standard_repn(obj.expr)
        self._add_expr_structs_from_repn(
            repn,
            add_const_fn=partial(self._execute, knitro.KN_add_obj_constant),
            add_lin_fn=partial(self._execute, knitro.KN_add_obj_linear_struct),
            add_quad_fn=partial(self._execute, knitro.KN_add_obj_quadratic_struct),
        )
        if repn.nonlinear_expr is not None:
            self.obj_nl_expr = NLExpression(repn.nonlinear_expr, repn.nonlinear_vars)

    def _build_callback(self):
        if self.obj_nl_expr is None and not self.con_nl_expr_map:
            return None

        obj_eval = (
            self.obj_nl_expr.create_evaluator(self.var_map)
            if self.obj_nl_expr is not None
            else None
        )
        con_eval_map = {
            i: nl_expr.create_evaluator(self.var_map)
            for i, nl_expr in self.con_nl_expr_map.items()
        }

        def _callback(_, cb, req, res, data=None):
            if req.type != knitro.KN_RC_EVALFC:
                # This callback only handles function evaluations, not derivatives.
                return -1
            x = req.x
            if obj_eval is not None:
                res.obj = obj_eval(x)
            for i, con_eval in enumerate(con_eval_map.values()):
                res.c[i] = con_eval(x)
            return 0  # Return 0 for success

        return _callback

    def _register_callback(self):
        callback_fn = self._build_callback()
        if callback_fn is not None:
            eval_obj = self.obj_nl_expr is not None
            idx_cons = list(self.con_nl_expr_map.keys())
            self._execute(knitro.KN_add_eval_callback, eval_obj, idx_cons, callback_fn)

    def solve(self) -> int:
        self._register_callback()
        return self._execute(knitro.KN_solve)

    def get_num_iters(self) -> int:
        return self._execute(knitro.KN_get_number_iters)

    def get_solve_time(self) -> float:
        return self._execute(knitro.KN_get_solve_time_real)

    def get_primals(self, variables: Iterable[VarData]) -> Optional[List[float]]:
        idx_vars = [self.var_map.get(id(var)) for var in variables]
        return self._execute(knitro.KN_get_var_primal_values, idx_vars)

    def get_duals(self, cons: Iterable[ConstraintData]) -> Optional[List[float]]:
        idx_cons = [self.con_map.get(id(con)) for con in cons]
        return self._execute(knitro.KN_get_con_dual_values, idx_cons)

    def set_options(self, **options):
        for param, val in options.items():
            param_id = self._execute(knitro.KN_get_param_id, param)
            param_type = self._execute(knitro.KN_get_param_type, param_id)
            if param_type == knitro.KN_PARAMTYPE_INTEGER:
                setter_fn = knitro.KN_set_int_param
            elif param_type == knitro.KN_PARAMTYPE_FLOAT:
                setter_fn = knitro.KN_set_double_param
            else:
                setter_fn = knitro.KN_set_char_param
            self._execute(setter_fn, param_id, val)

    def set_outlev(self, level: int = knitro.KN_OUTLEV_ALL):
        self.set_options(outlev=level)

    def set_time_limit(self, time_limit: float):
        self.set_options(maxtime_cpu=time_limit)

    def set_num_threads(self, num_threads: int):
        self.set_options(numthreads=num_threads)


class KnitroDirectSolutionLoader(SolutionLoaderBase):
    def __init__(self, problem: KnitroProblemContext, model_repn: ModelRepresentation):
        super().__init__()
        self._problem = problem
        self._model_repn = model_repn

    def get_number_of_solutions(self) -> int:
        _, _, x, _ = self._problem._execute(knitro.KN_get_solution)
        return 1 if x is not None else 0

    def get_vars(
        self, vars_to_load: Optional[Sequence[VarData]] = None
    ) -> Mapping[VarData, float]:
        if vars_to_load is None:
            vars_to_load = self._model_repn.variables

        x = self._problem.get_primals(vars_to_load)
        if x is None:
            return NoSolutionError()
        return ComponentMap([(var, x[i]) for i, var in enumerate(vars_to_load)])

    # TODO: remove this when the solution loader is fixed.
    def get_primals(self, vars_to_load=None):
        return self.get_vars(vars_to_load)

    def get_duals(
        self, cons_to_load: Optional[Sequence[ConstraintData]] = None
    ) -> Mapping[ConstraintData, float]:
        if cons_to_load is None:
            cons_to_load = self._model_repn.cons

        y = self._problem.get_duals(cons_to_load)
        if y is None:
            return NoDualsError()
        return ComponentMap([(con, y[i]) for i, con in enumerate(cons_to_load)])


class KnitroDirectSolver(SolverBase):
    NAME = "KNITRO"
    CONFIG = KnitroConfig()
    config: KnitroConfig

    def __init__(self, **kwds):
        super().__init__(**kwds)
        self._available_cache = NOTSET

    def available(self) -> Availability:
        if self._available_cache is NOTSET:
            self._available_cache = KnitroLicenseManager.available()
        return self._available_cache

    def version(self):
        return KnitroLicenseManager.version()

    def _build_config(self, **kwds) -> KnitroConfig:
        return self.config(value=kwds, preserve_implicit=True)

    def _validate_model(self, model_repn: ModelRepresentation):
        if len(model_repn.objs) > 1:
            raise IncompatibleModelError(
                f"{self.NAME} does not support multiple objectives."
            )

    def solve(self, model: BlockData, **kwds) -> Results:
        config = self._build_config(**kwds)
        timer = config.timer or HierarchicalTimer()

        avail = self.available()
        if not avail:
            raise ApplicationError(f"Solver {self.NAME} is not available: {avail}.")

        StaleFlagManager.mark_all_as_stale()
        timer.start("build_model_representation")
        model_repn = build_model_representation(model)
        timer.stop("build_model_representation")

        self._validate_model(model_repn)

        stream = io.StringIO()
        ostreams = [stream] + config.tee
        with capture_output(TeeStream(*ostreams), capture_fd=False):
            problem = KnitroProblemContext()

            timer.start("add_vars")
            problem.add_vars(model_repn.variables)
            timer.stop("add_vars")

            timer.start("add_cons")
            problem.add_cons(model_repn.cons)
            timer.stop("add_cons")

            if model_repn.objs:
                timer.start("set_objective")
                problem.set_obj(model_repn.objs[0])
                timer.stop("set_objective")

            problem.set_outlev()
            if config.threads is not None:
                problem.set_num_threads(config.threads)
            if config.time_limit is not None:
                problem.set_time_limit(config.time_limit)

            timer.start("load_options")
            problem.set_options(**config.solver_options)
            timer.stop("load_options")

            timer.start("solve")
            status = problem.solve()
            timer.stop("solve")

        results = Results()
        results.solver_config = config
        results.solver_name = self.NAME
        results.solver_version = self.version()
        results.solver_log = stream.getvalue()
        results.iteration_count = problem.get_num_iters()
        results.solution_status = get_solution_status(status)
        results.termination_condition = get_termination_condition(status)
        if (
            config.raise_exception_on_nonoptimal_result
            and results.termination_condition
            != TerminationCondition.convergenceCriteriaSatisfied
        ):
            raise NoOptimalSolutionError()

        results.solution_loader = KnitroDirectSolutionLoader(problem, model_repn)
        if config.load_solutions:
            timer.start("load_solutions")
            results.solution_loader.load_vars()
            timer.stop("load_solutions")

        results.timing_info.solve_time = problem.get_solve_time()
        return results
