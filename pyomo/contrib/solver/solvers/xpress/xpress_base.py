# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""
Shared infrastructure for the Xpress connector: expression walker, solution loaders,
and the constraint/variable build helpers used by both XpressDirect and XpressPersistent.

Architecture overview
---------------------
The connector converts a Pyomo ConcreteModel into an xp.problem by walking every
constraint body with XpressExpressionWalker, which turns each Pyomo expression tree
into an equivalent Xpress expression object (xp.Sum, xp.constraint, etc.).

This per-constraint walk approach handles LP, MIP, QP, QCP, and NLP in a single code
path without special-casing. The linear walk fast path, PauseGC, and single-pass
to_bounded_expression keep per-constraint overhead low enough that the flexibility
comes at no practical cost.

Mutable parameter tracking (XpressPersistent)
----------------------------------------------
XpressPersistent uses generate_standard_repn (in xpress_persistent.py) to identify
mutable coefficients at set_instance time.  The walker here is a pure expression
builder: it produces xp expressions from Pyomo expression trees but does not track
which params affect which matrix entries.  All mutable tracking logic lives in
xpress_persistent.py.
"""

import datetime
import io
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Optional, Sequence, cast

from pyomo.common.collections import ComponentMap
from pyomo.common.dependencies import attempt_import
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.common.tee import capture_output, TeeStream
from pyomo.common.timing import HierarchicalTimer
from pyomo.common.enums import ObjectiveSense
from pyomo.core.staleflag import StaleFlagManager
from pyomo.core.base.block import BlockData
from pyomo.core.base.var import VarData
from pyomo.core.base.constraint import ConstraintData
from pyomo.core.base.external import PythonCallbackFunction
from pyomo.core.base.sos import SOSConstraintData
from pyomo.core.expr.numeric_expr import (
    AbsExpression,
    DivisionExpression,
    ExpressionBase,
    Expr_ifExpression,
    ExternalFunctionExpression,
    LinearExpression,
    MaxExpression,
    MinExpression,
    MonomialTermExpression,
    NegationExpression,
    ProductExpression,
    PowExpression,
    SumExpression,
    UnaryFunctionExpression,
)
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor
from pyomo.core.base import Expression as NamedExpressionComponent
from pyomo.core.expr.numvalue import value
from pyomo.common.gc_manager import PauseGC
from pyomo.repn.util import BeforeChildDispatcher

from pyomo.contrib.solver.common.base import SolverBase, Availability
from pyomo.contrib.solver.common.config import BranchAndBoundConfig
from pyomo.contrib.solver.common.results import (
    Results,
    SolutionStatus,
    TerminationCondition,
    get_infeasible_results,
)
from pyomo.contrib.solver.common.solution_loader import SolutionLoader
from pyomo.contrib.solver.common.util import (
    IncompatibleModelError,
    NoDualsError,
    NoReducedCostsError,
    NoSolutionError,
    NoFeasibleSolutionError,
    NoOptimalSolutionError,
)

# ---- Pure-Python module-level constants (no xp dependency) ------------------

_BOUND_TYPE_CODES = ['L', 'U']

_VAR_TYPE_CODES: dict[tuple[bool, bool], int] = {
    (True, True): 66,  # 'B' -- binary (implies integer)
    (False, True): 73,  # 'I' -- integer
    (False, False): 67,  # 'C' -- continuous
}

# ---- xp-dependent maps (empty until _init_xp_maps() is called) --------------
# Mutable dicts so that `from .xpress_base import _OBJ_SENSE_MAP` in sibling
# modules receives the same object and sees updates via .update().

_VAR_XP_TYPE_MAP: dict = {}
_OBJ_SENSE_MAP: dict = {}
_SOL_STATUS_MAP: dict = {}
_STOP_TYPE_MAP: dict = {}
_XP_FUNCTION_MAP: dict = {}
_CON_TYPE_MAP: dict = {}  # (is_range, is_equality, has_ub) -> xp constraint type


class _ExitHandlerMap(dict):
    """Dict with MRO fallback: unknown subclasses resolve via their base class.

    Pyomo has many concrete expression subclasses (e.g. ScalarExpression) that
    are not registered explicitly.  Walking the Python MRO finds the registered
    base class handler (e.g. NamedExpressionComponent for all named expressions)
    and caches the result so subsequent lookups are direct dict hits.
    Only the minimal set of base classes needs to be registered.
    """

    def __missing__(self, key):
        for cls in key.__mro__:
            if cls in self:
                self[key] = self[cls]  # cache for subsequent lookups
                return self[cls]
        raise IncompatibleModelError(
            f"Expression type '{type(key).__name__}' is not supported by Xpress."
        )


_EXIT_HANDLERS: _ExitHandlerMap = _ExitHandlerMap()


def _exit_unary(visitor: 'XpressExpressionWalker', node, arg) -> Any:
    fn = _XP_FUNCTION_MAP.get(node.getname())
    if fn is None:
        raise IncompatibleModelError(
            f"Unsupported function '{node.getname()}' in expression. "
            "Xpress does not support this function natively."
        )
    return fn(arg)


def _exit_named_expression(visitor: 'XpressExpressionWalker', node, arg) -> Any:
    visitor.subexpression_cache[id(node)] = arg
    return arg


def _exit_external_function(visitor: 'XpressExpressionWalker', node, *data) -> Any:
    """Handle ExternalFunctionExpression nodes via xp.user.

    `data` is (xp_arg1, ..., xp_argN, fcn_id_int).  The last element is the
    evaluated _PythonCallbackFunctionID (a plain Python int, not an xp expression).
    Only PythonCallbackFunction is supported; AMPLExternalFunction has no
    callable Python callbacks.
    """
    pyo_fcn = node._fcn
    if not isinstance(pyo_fcn, PythonCallbackFunction):
        raise IncompatibleModelError(
            f"ExternalFunction of type '{type(pyo_fcn).__name__}' uses AMPL callbacks "
            "which are not supported by the Xpress connector. "
            "Use PythonCallbackFunction instead."
        )
    xp_args = data[:-1]  # strip fcn_id (last element is always a plain int)
    has_grad = pyo_fcn._grad is not None or pyo_fcn._fgh is not None

    if pyo_fcn._fgh is not None:
        _fgh = pyo_fcn._fgh

        def xp_cb(*vals):
            f, g, _ = _fgh(list(vals), 1, None)
            return (f, *g) if g is not None else f

    elif pyo_fcn._grad is not None:
        _fcn, _grad = pyo_fcn._fcn, pyo_fcn._grad

        def xp_cb(*vals):
            args = list(vals)
            return (_fcn(*args), *_grad(args, None))

    else:
        _fcn = pyo_fcn._fcn

        def xp_cb(*vals):
            return _fcn(*vals)

    # Xpress identifies user functions by their __name__; each distinct Pyomo
    # ExternalFunction needs a unique name to avoid a "duplicate user function" error.
    xp_cb.__name__ = f'xp_user_{id(pyo_fcn)}'
    return xp.user(xp_cb, *xp_args, derivatives="always" if has_grad else "never")


def _init_xpress(xp, xpress_available):
    """Populate all Pyomo<->Xpress entity maps. No-op if already done or if
    xpress did not import successfully."""
    if not xpress_available:
        return

    _VAR_XP_TYPE_MAP.update(
        {
            (True, True): xp.binary,
            (False, True): xp.integer,
            (False, False): xp.continuous,
        }
    )
    _OBJ_SENSE_MAP.update(
        {
            ObjectiveSense.minimize: xp.ObjSense.MINIMIZE,
            ObjectiveSense.maximize: xp.ObjSense.MAXIMIZE,
        }
    )
    TC = TerminationCondition
    SS = SolutionStatus
    _SOL_STATUS_MAP.update(
        {
            xp.SolStatus.OPTIMAL: (TC.convergenceCriteriaSatisfied, SS.optimal),
            # SLP (xp.user) reports FEASIBLE on completion: local optimum found.
            xp.SolStatus.FEASIBLE: (TC.convergenceCriteriaSatisfied, SS.feasible),
            xp.SolStatus.INFEASIBLE: (TC.provenInfeasible, SS.infeasible),
            xp.SolStatus.UNBOUNDED: (TC.unbounded, SS.unknown),
        }
    )
    _STOP_TYPE_MAP.update(
        {
            xp.StopType.TIMELIMIT: TC.maxTimeLimit,
            xp.StopType.NODELIMIT: TC.iterationLimit,
            xp.StopType.ITERLIMIT: TC.iterationLimit,
            xp.StopType.WORKLIMIT: TC.iterationLimit,
            xp.StopType.MIPGAP: TC.convergenceCriteriaSatisfied,
            xp.StopType.CTRLC: TC.interrupted,
            xp.StopType.USER: TC.interrupted,
            xp.StopType.SOLLIMIT: TC.objectiveLimit,
            xp.StopType.GENERICERROR: TC.error,
            xp.StopType.MEMORYERROR: TC.error,
            xp.StopType.NUMERICALERROR: TC.error,
        }
    )

    # (is_range, is_equality, has_ub) -> xp constraint type
    _CON_TYPE_MAP.update(
        {
            (True, False, True): xp.rng,
            (False, True, True): xp.eq,
            (False, True, False): xp.eq,
            (False, False, True): xp.leq,
            (False, False, False): xp.geq,
        }
    )

    # 1:1 mapping from pyomo nodes to Xpress operators
    _EXIT_HANDLERS.update(
        {
            NegationExpression: lambda v, n, a: -a,
            SumExpression: lambda v, n, *a: xp.Sum(list(a)),
            ProductExpression: lambda v, n, a, b: a * b,
            MonomialTermExpression: lambda v, n, a, b: a * b,
            DivisionExpression: lambda v, n, a, b: a / b,
            PowExpression: lambda v, n, a, b: a**b,
            MaxExpression: lambda v, n, *a: xp.max(list(a)),
            MinExpression: lambda v, n, *a: xp.min(list(a)),
            AbsExpression: lambda v, n, a: xp.abs(a),
            UnaryFunctionExpression: _exit_unary,
            NamedExpressionComponent: _exit_named_expression,
            ExternalFunctionExpression: _exit_external_function,
        }
    )

    # Pyomo defines many unary operators with a single class + a name attribute.
    # We need a 2 levels dispatch with this second dict.
    _XP_FUNCTION_MAP.update(
        {
            'sin': xp.sin,
            'cos': xp.cos,
            'tan': xp.tan,
            'asin': xp.asin,
            'acos': xp.acos,
            'atan': xp.atan,
            'exp': xp.exp,
            'log': xp.log,
            'log10': xp.log10,
            'sqrt': xp.sqrt,
            'abs': xp.abs,
            # Decomposed via xpress primitives -- same call signature as direct fns
            'sinh': lambda e: 0.5 * (xp.exp(e) - xp.exp(-e)),
            'cosh': lambda e: 0.5 * (xp.exp(e) + xp.exp(-e)),
            'tanh': lambda e: (xp.exp(e) - xp.exp(-e)) / (xp.exp(e) + xp.exp(-e)),
            'asinh': lambda e: xp.log(e + xp.sqrt(e * e + 1)),
            'acosh': lambda e: xp.log(e + xp.sqrt(e * e - 1)),
            'atanh': lambda e: 0.5 * xp.log((1 + e) / (1 - e)),
            # TODO: we could support ceil/floor adding 1 auxiliary int var and constraint
        }
    )


xp, xpress_available = attempt_import('xpress', callback=_init_xpress)


def _register_pool_collector(prob, pool_limit: int) -> 'list[list[float]]':
    """Register an IntsolCallback to collect MIP solutions during solve.

    Returns the list that will be populated by the callback.  Solution 0
    (the incumbent) is always available via prob.getSolution(); pool[k] holds
    the (k+1)-th solution kept in the rolling window.

    pool_limit > 0: keep a rolling window of the last N solutions found.
        When the window is full the oldest entry is evicted on each new solution,
        so the pool always contains the N most recently found feasible solutions.

    Zero overhead for LP/QP: the callback never fires for continuous problems.
    """
    pool: list[list[float]] = []

    def _intsol_cb(cbprob, cbdata):
        if len(pool) == pool_limit:
            pool.pop(0)
        pool.append(cbprob.getCallbackSolution())

    # We set SERIALIZEPREINTSOL to ensure determinism in how solutions are found
    prob.controls.serializepreintsol = 1
    prob.addIntsolCallback(_intsol_cb)
    return pool


@dataclass
class EntityMaps:
    """Stable handle maps from Pyomo entities to Xpress entity objects.

    Xpress entity handles (xp.var, xp.constraint, xp.sos) remain valid after
    other entities are deleted, unlike integer indices, which Xpress renumbers
    after every deletion.  Storing handles here eliminates all index rebuilds.

    vars: keyed by id(VarData), not VarData itself, because VarData objects are
      not hashable in all Pyomo versions and id() is stable for the lifetime of
      the solve session.

    cons: each constraint maps to a single xp.constraint handle. Range constraints
      are stored as Xpress 'R'-type rows, one row per constraint.

    sos: values are single xp.sos handles (SOS sets are never split).
    """

    vars: dict[int, Any]  # id(VarData) -> xp.var
    cons: dict[ConstraintData, Any]  # ConstraintData -> xp.constraint
    sos: dict[SOSConstraintData, Any]  # SOSConstraintData -> xp.sos


class XpressSolutionLoaderBase(SolutionLoader):
    """Base solution loader shared by direct and persistent Xpress solvers."""

    def __init__(
        self,
        xp_prob,
        pyomo_model: BlockData,
        variables: list[VarData],
        maps: EntityMaps,
        pool_solutions: Optional[list] = None,
    ) -> None:
        super().__init__()
        self._xp_prob = xp_prob
        self._pyomo_model = pyomo_model
        self._vars = variables
        self._maps = maps
        # list[list[float]]: pool[k] = solution (k+1) in B&B discovery order.
        # Populated by _register_pool_collector during prob.optimize().
        self._pool: list = pool_solutions if pool_solutions is not None else []
        # Active solution id: 0 = incumbent (default). k > 0 = pool[k-1].
        # _set_solution_id(None) is treated the same as 0 for framework compat.
        self._active_id: int = 0

    def _query_vars(
        self, variables: Optional[Sequence[VarData]], fn, exc_type: type[Exception]
    ) -> Iterator[tuple[VarData, float]]:
        """Query a variable-valued attribute from the Xpress problem, returning
        (VarData, value) pairs. Raises exc_type on xp.ModelError."""
        try:
            if variables is None:
                return zip(self._vars, fn())
            xp_vars = [self._maps.vars[id(var)] for var in variables]
            return zip(variables, fn(xp_vars))
        except xp.ModelError as e:
            raise exc_type() from e

    def get_number_of_solutions(self) -> int:
        # LP/QP: pool stays empty, so returns 1 (the incumbent).
        # MIP with pool_solutions > 0: incumbent + len(pool) collected solutions.
        return 1 + len(self._pool)

    def get_solution_ids(self) -> list:
        return list(range(1 + len(self._pool)))

    def _set_solution_id(self, solution_id: Optional[int]) -> Optional[int]:
        prev = self._active_id
        self._active_id = solution_id
        return prev

    def _get_solution_vals(
        self, vars_to_load: Optional[Sequence[VarData]]
    ) -> Iterator[tuple[VarData, float]]:
        """Yield (VarData, float) pairs for the currently active solution.

        Active solution 0 or None -> incumbent via prob.getSolution().
        Active solution k > 0     -> pool[k-1] (B&B discovery order).

        For pool entries, values are indexed by xp.var.index (Xpress column
        index), which matches the column registration order in self._vars.
        """
        sid = self._active_id
        if not sid:
            prob = self._xp_prob
            return self._query_vars(vars_to_load, prob.getSolution, NoSolutionError)
        k = sid - 1
        if k >= len(self._pool):
            raise NoSolutionError(
                f'Solution {sid} not available: pool contains {len(self._pool)} solutions.'
            )
        sol = self._pool[k]
        return self._query_vars(
            vars_to_load,
            lambda vs=None: sol if vs is None else [sol[v.index] for v in vs],
            NoSolutionError,
        )

    def load_vars(self, vars_to_load: Sequence[VarData] | None = None) -> None:
        for var, val in self._get_solution_vals(vars_to_load):
            var.set_value(val, skip_validation=True)
        StaleFlagManager.mark_all_as_stale(delayed=True)

    def get_vars(
        self, vars_to_load: Sequence[VarData] | None = None
    ) -> Mapping[VarData, float]:
        return ComponentMap(self._get_solution_vals(vars_to_load))

    def get_reduced_costs(
        self, vars_to_load: Sequence[VarData] | None = None
    ) -> Mapping[VarData, float]:
        if self._active_id != 0:
            raise NoReducedCostsError(
                'Reduced costs available only for incumbent (solution_id=0).'
            )
        prob = self._xp_prob
        pairs = self._query_vars(vars_to_load, prob.getRedCosts, NoReducedCostsError)
        return ComponentMap(pairs)

    def get_duals(
        self, cons_to_load: Sequence[ConstraintData] | None = None
    ) -> dict[ConstraintData, float]:
        if self._active_id != 0:
            raise NoDualsError('Duals available only for incumbent (solution_id=0).')
        if cons_to_load is None:
            # Note: Xpress cons order might differ from maps.cons dict order
            #       thus, prob.getDuals() with no args is not viable here
            cons_to_load = list(self._maps.cons.keys())
            xp_cons = list(self._maps.cons.values())
        else:
            xp_cons = [self._maps.cons[c] for c in cons_to_load]
        try:
            vals = self._xp_prob.getDuals(xp_cons)
        except xp.ModelError as e:
            raise NoDualsError() from e
        return {con: float(vals[i]) for i, con in enumerate(cons_to_load)}


class XpressSolverMixin(SolverBase):
    """Shared logic for XpressDirect and XpressPersistent."""

    _available = None
    _version = None
    _xpress_available = xpress_available

    def available(self) -> Availability:
        if self._available is None:
            if not self._xpress_available:
                type(self)._available = Availability.NotFound
            else:
                try:
                    xp.problem()
                    type(self)._available = Availability.FullLicense
                except Exception:
                    type(self)._available = Availability.BadLicense
        assert self._available is not None
        return self._available

    def version(self) -> tuple:
        if not xpress_available:
            return tuple()
        if XpressSolverMixin._version is None:
            XpressSolverMixin._version = tuple(
                getattr(xp, 'getVersionNumbers', xp.getversionnumbers)()
            )
        return XpressSolverMixin._version

    @staticmethod
    def _var_bounds(var: VarData) -> tuple:
        if var.fixed:
            if var.value is None:
                raise ValueError(f"Variable '{var.name}' is fixed but has no value.")
            val = value(var.value)
            return val, val
        vlb, vub = var.bounds
        inf = xp.infinity
        return (-inf if vlb is None else value(vlb), inf if vub is None else value(vub))

    @staticmethod
    def _set_var_types(prob, pyo_vars: list[VarData], xp_vars) -> None:
        """Bulk-set column types (C/I/B) for pyo_vars."""
        ctypes = [_VAR_TYPE_CODES[v.is_binary(), v.is_integer()] for v in pyo_vars]
        prob.chgColType(xp_vars, ctypes)

    def _set_var_bounds(self, prob, pyo_vars: list[VarData], xp_vars) -> None:
        """Bulk-set variable bounds for pyo_vars, respecting fixed-var pinning."""
        n = len(pyo_vars)
        cbounds = [b for var in pyo_vars for b in self._var_bounds(var)]
        cols = [v for v in xp_vars for _ in range(2)]
        prob.chgBounds(cols, _BOUND_TYPE_CODES * n, cbounds)

    def _add_vars_impl(self, prob, pyo_vars: list[VarData], symbolic_labels: bool):
        """Add columns, set types and bounds. Returns the xp.var array.

        Uses addVariables(n, name='') to get sequential C1/C2/... auto-names
        that never repeat across incremental calls. Without name='', Xpress
        generates x(0)/x(1)/... which conflict on a second batch.
        """
        n = len(pyo_vars)
        if n == 0:
            return []
        ncol = prob.attributes.cols
        xp_vars = prob.addVariables(len(pyo_vars), name='')
        self._set_var_types(prob, pyo_vars, xp_vars)
        self._set_var_bounds(prob, pyo_vars, xp_vars)
        if symbolic_labels:
            names = [v.name for v in pyo_vars]
            prob.addNames(xp.Namespaces.COLUMN, names, ncol, ncol + n - 1)
        return xp_vars

    @staticmethod
    def _add_cons_impl(
        prob,
        pyo_cons: list[ConstraintData],
        walker: 'XpressExpressionWalker',
        symbolic_labels: bool,
    ) -> list:
        """Walk pyo_cons and build xp.constraint objects in a single tight loop.

        Performance choices:
          - PauseGC wraps the loop: each walk creates many short-lived objects;
            deferring GC eliminates unpredictable mid-loop pauses.
          - to_bounded_expression called once per constraint to obtain (lb, body, ub)
            together, avoiding three separate property accesses.
          - _before_linear fast path: LinearExpression bodies (the common case for LP/MIP)
            bypass the full StreamBasedExpressionVisitor dispatch, saving
            initializeWalker + enterNode overhead.
        """
        if len(pyo_cons) == 0:
            return []
        _walk_expr = walker.walk_expression
        xp_cons = []
        with PauseGC():
            for c in pyo_cons:
                lb, body, ub = c.to_bounded_expression()
                if type(body) is LinearExpression:
                    result = _before_linear(walker, body)[1]
                else:
                    result = _walk_expr(body)
                name = c.name if symbolic_labels else None
                vlb, vub = value(lb), value(ub)
                xp_cons.append(xp.constraint(body=result, lb=vlb, ub=vub, name=name))
        prob.addConstraint(xp_cons)
        return xp_cons

    @staticmethod
    def _add_sos_impl(
        prob, pyo_sos: list, var_map: dict[int, Any], symbolic_labels: bool = False
    ):
        """Add SOS sets. Returns the xp.sos handle array."""
        n = len(pyo_sos)
        if n == 0:
            return []
        settype: list[int] = []
        setstart: list[int] = [0]
        setind: list[int] = []
        refval: list[float] = []
        for con in pyo_sos:
            setind.extend(var_map[id(var)].index for var in con.variables)
            refval.extend(float(w) for _, w in con.get_items())
            settype.append(ord('1' if con.level == 1 else '2'))
            setstart.append(len(setind))
        nsos = prob.attributes.sets
        prob.addSets(settype, setstart, setind, refval)
        if symbolic_labels:
            names = [con.name for con in pyo_sos]
            prob.addNames(xp.Namespaces.SET, names, nsos, nsos + n - 1)
        return prob.getSOS(first=nsos, last=nsos + n - 1)

    def _warmstart(self, prob, vars: list[VarData], entind: list[int]) -> None:
        ws_vals: list[float] = []
        ws_cols: list[int] = []
        for j in entind:
            var = vars[j]
            if var.value is not None:
                ws_vals.append(var.value)
                ws_cols.append(j)
        if ws_vals:
            prob.addMipSol(ws_vals, ws_cols)

    def _apply_solver_controls(self, prob, config: BranchAndBoundConfig) -> None:
        if config.time_limit is not None:
            prob.controls.timelimit = float(config.time_limit)
        if config.threads is not None:
            prob.controls.threads = config.threads
        if config.rel_gap is not None:
            prob.controls.miprelstop = config.rel_gap
        if config.abs_gap is not None:
            prob.controls.mipabsstop = config.abs_gap
        for key, val in config.solver_options.items():
            setattr(prob.controls, key, val)

    def _create_xpress_model(
        self, _m: BlockData, _c: BranchAndBoundConfig, _t: HierarchicalTimer
    ) -> tuple:
        raise NotImplementedError

    def solve(self, model: BlockData, **kwds) -> Results:
        start_timestamp = datetime.datetime.now(datetime.timezone.utc)
        tick = time.perf_counter()

        config = cast(
            BranchAndBoundConfig, self.config(value=kwds, preserve_implicit=True)
        )
        if config.timer is None:
            config.timer = HierarchicalTimer()
        timer = config.timer

        StaleFlagManager.mark_all_as_stale()
        log_stream = io.StringIO()
        ostreams = [log_stream] + config.tee

        orig_cwd = None
        if config.working_dir is not None:
            orig_cwd = os.getcwd()
            os.chdir(str(config.working_dir))

        try:
            with capture_output(TeeStream(*ostreams), capture_fd=False):
                prob, solution_loader, has_obj = self._create_xpress_model(
                    model, config, timer
                )
                self._apply_solver_controls(prob, config)
                timer.start('optimize')
                prob.optimize()
                timer.stop('optimize')
                res = self._populate_results(prob, solution_loader, has_obj, config)

        except InfeasibleConstraintException as err:
            res = get_infeasible_results(
                model=model,
                solver=self,
                config=config,
                err_msg=(
                    'The problem was proven infeasible during compilation:\n' f'\t{err}'
                ),
            )
        finally:
            if orig_cwd is not None:
                os.chdir(orig_cwd)

        res.solver_log = log_stream.getvalue()
        tock = time.perf_counter()
        res.timing_info.start_timestamp = start_timestamp
        res.timing_info.wall_time = tock - tick
        res.timing_info.timer = timer
        return res

    def _populate_results(
        self,
        prob,
        solution_loader: XpressSolutionLoaderBase,
        has_obj: bool,
        config: BranchAndBoundConfig,
    ) -> Results:
        sv = prob.attributes.solvestatus
        ss = prob.attributes.solstatus
        st = prob.attributes.stopstatus

        TC = TerminationCondition
        SS = SolutionStatus

        if sv == xp.SolveStatus.COMPLETED:
            tc, sol_status = _SOL_STATUS_MAP.get(ss, (TC.unknown, SS.noSolution))
        elif sv == xp.SolveStatus.STOPPED:
            sol_status = SS.feasible if ss == xp.SolStatus.FEASIBLE else SS.noSolution
            tc = _STOP_TYPE_MAP.get(st, TC.unknown)
        elif sv == xp.SolveStatus.FAILED:
            tc = TC.error
            sol_status = SS.noSolution
        else:  # UNSTARTED
            tc = TC.unknown
            sol_status = SS.noSolution

        results = Results()
        results.termination_condition = tc
        results.solution_status = sol_status
        results.solution_loader = solution_loader
        results.solver_name = self.name
        results.solver_version = self.version()
        results.solver_config = config

        has_solution = sol_status in (SS.optimal, SS.feasible)

        if has_obj and has_solution:
            try:
                obj_val = float(prob.attributes.objval)
                results.incumbent_objective = (
                    None if not math.isfinite(obj_val) else obj_val
                )
            except (xp.ModelError, AttributeError):
                results.incumbent_objective = None
            try:
                results.objective_bound = float(prob.attributes.bestbound)
            except (xp.ModelError, AttributeError):
                results.objective_bound = (
                    -math.inf if sol_status == SS.optimal else math.inf
                )
        else:
            results.incumbent_objective = None
            results.objective_bound = None

        results.timing_info.xpress_time = prob.attributes.time
        results.extra_info.simplex_iterations = prob.attributes.simplexiter
        results.extra_info.barrier_iterations = prob.attributes.bariter
        results.extra_info.node_count = prob.attributes.nodes
        results.extra_info.mip_solutions_found = prob.attributes.mipsols

        if (
            tc != TC.convergenceCriteriaSatisfied
            and config.raise_exception_on_nonoptimal_result
        ):
            raise NoOptimalSolutionError()

        if config.load_solutions:
            if has_solution:
                solution_loader.load_solution()
            else:
                raise NoFeasibleSolutionError()

        return results


# ---- Expression Walker ------------------------------------------------------


def _register_variable(visitor: 'XpressExpressionWalker', pyo_var: VarData):
    """Return the xp.var for pyo_var, registering it in prob/var_map on first use."""
    xp_var = visitor.var_map.get(id(pyo_var))
    if xp_var is not None:
        return xp_var
    lb, ub = XpressSolverMixin._var_bounds(pyo_var)
    vtype = _VAR_XP_TYPE_MAP[pyo_var.is_binary(), pyo_var.is_integer()]
    vname = pyo_var.name if visitor.use_names else None
    xp_var = visitor.prob.addVariable(lb=lb, ub=ub, vartype=vtype, name=vname)
    visitor.var_map[id(pyo_var)] = xp_var
    if visitor.registered_vars is not None:
        visitor.registered_vars.append(pyo_var)
    return xp_var


def _before_monomial(visitor: 'XpressExpressionWalker', child: MonomialTermExpression):
    coef, var = child.args
    coef_val = value(coef)
    xp_var = _register_variable(visitor, var)
    return False, coef_val * xp_var


def _before_linear(visitor: 'XpressExpressionWalker', child: LinearExpression):
    # child.args bypasses LinearExpression._build_cache.
    # xp.Sum(list) is a single C call; Python + accumulation is O(n) calls.
    terms = []
    for arg in child.args:
        if isinstance(arg, VarData):
            terms.append(_register_variable(visitor, arg))
        elif type(arg) is MonomialTermExpression:
            coef, var = arg.args
            terms.append(value(coef) * _register_variable(visitor, var))
        else:
            terms.append(value(arg))
    return False, xp.Sum(terms)


def _before_incompatible(_v, child: ExpressionBase):
    raise IncompatibleModelError(
        f"Expression '{child}' of type '{type(child).__name__}' "
        "is not supported by the Xpress solver."
    )


class XpressBeforeChildDispatcher(BeforeChildDispatcher):
    __slots__ = ()

    def __init__(self) -> None:
        super().__init__()
        self[MonomialTermExpression] = _before_monomial
        self[LinearExpression] = _before_linear
        self[Expr_ifExpression] = _before_incompatible

    @staticmethod
    def _before_var(visitor: 'XpressExpressionWalker', child):
        return False, _register_variable(visitor, child)

    @staticmethod
    def _before_named_expression(visitor: 'XpressExpressionWalker', child) -> tuple:
        cached = visitor.subexpression_cache.get(id(child), None)
        return cached is None, cached

    @staticmethod
    def _before_leaf(visitor, child):
        return False, value(child)

    _before_npv = _before_leaf
    _before_param = _before_leaf
    _before_native_numeric = _before_leaf
    _before_native_logical = _before_leaf
    _before_complex = _before_incompatible
    _before_string = _before_incompatible
    _before_invalid = _before_incompatible


class XpressExpressionWalker(StreamBasedExpressionVisitor):
    """Pyomo expression tree -> Xpress expression objects.

    Pure expression builder: no ExprType tracking, no mutable param tracking.
    Each node type maps directly to its Xpress operation in _EXIT_HANDLERS.
    Mutable tracking is handled by generate_standard_repn in xpress_persistent.py.

    _before_linear provides a fast path for LinearExpression bodies (the dominant
    case in LP/MIP models) via direct call from _add_cons_impl.
    """

    before_child_dispatcher = XpressBeforeChildDispatcher()

    def __init__(
        self,
        var_map: dict[int, Any],
        prob,
        use_names: bool = False,
        registered_vars: 'Optional[list[VarData]]' = None,
    ) -> None:
        super().__init__()
        self.var_map = var_map
        self.prob = prob
        self.use_names = use_names
        self.registered_vars = registered_vars
        self.subexpression_cache: dict = {}

    def initializeWalker(self, expr) -> tuple:
        return self.beforeChild(None, expr, 0)

    def beforeChild(self, _n, child, _c: int) -> tuple:
        return self.before_child_dispatcher[type(child)](self, child)

    def exitNode(self, node: ExpressionBase, data: list) -> Any:
        return _EXIT_HANDLERS[type(node)](self, node, *data)
