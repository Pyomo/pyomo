# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

from typing import Any, Collection, Mapping, Optional, Sequence

from pyomo.common.timing import HierarchicalTimer
from pyomo.core.base.block import BlockData
from pyomo.core.base.constraint import Constraint, ConstraintData
from pyomo.core.base.objective import ObjectiveData
from pyomo.core.base.param import ParamData
from pyomo.core.base.sos import SOSConstraint, SOSConstraintData
from pyomo.core.base.var import VarData
from pyomo.core.expr.numvalue import value
from pyomo.common.collections import ComponentMap
from pyomo.repn import generate_standard_repn

from pyomo.contrib.solver.common.base import PersistentSolverBase
from pyomo.common.config import ConfigValue, NonNegativeInt
from pyomo.contrib.solver.common.config import BranchAndBoundConfig
from pyomo.contrib.solver.common.util import (
    IncompatibleModelError,
    NoSolutionError,
    get_objective,
)
from pyomo.contrib.observer.model_observer import (
    AutoUpdateConfig,
    ModelChangeDetector,
    Observer,
    Reason,
)

from .xpress_base import (
    XpressSolverMixin,
    XpressSolutionLoaderBase,
    EntityMaps,
    XpressExpressionWalker,
    _OBJ_SENSE_MAP,
    _CON_TYPE_MAP,
    _register_pool_collector,
    xp,
)


def _is_constant(expr) -> bool:
    """True if expr has no mutable params (safe to treat as a fixed coefficient)."""
    return getattr(expr, 'is_constant', lambda: True)()


def _collect_fixed(vars: list) -> ComponentMap:
    """Collect and unfix all fixed variables; return ComponentMap for re-fixing."""
    fixed = ComponentMap()
    for var in vars:
        if var.is_fixed():
            fixed[var] = var.value
            var.unfix()
    return fixed


def _refix(fixed: ComponentMap) -> None:
    """Re-fix variables that were temporarily unfixed before repn computation."""
    for var, val in fixed.items():
        var.fix(val)


# ---------------------------------------------------------------------------
# Mutable parameter helpers
# ---------------------------------------------------------------------------


class _UpdateBatch:
    """Accumulates constraint updates for a single flush.

    LP/QP affine updates are applied first (no structural changes, indices stable).
    NL rebuilds follow as a single batched delConstraint + addConstraint pair.
    The new xp.constraint objects are built during collect() so flush() needs
    only prob, maps, and mutable_helpers.
    """

    def __init__(self):
        # parallel lists for chgMCoef(rows, cols, vals)
        self.coef_rows = []
        self.coef_cols = []
        self.coef_vals = []
        # parallel lists for chgRHS(rows, vals)
        self.rhs_rows = []
        self.rhs_vals = []
        # parallel lists for chgRHSRange(rows, vals)
        self.rng_rows = []
        self.rng_vals = []
        # list of (row, col1, col2, val) tuples for chgQRowCoeff
        self.quad_updates = []
        # NL row replacement: old handles for delConstraint, new for addConstraint
        self.nl_old_cons = []
        self.nl_new_cons = []

    def flush(self, prob) -> None:
        if self.coef_rows:
            prob.chgMCoef(self.coef_rows, self.coef_cols, self.coef_vals)
        if self.rhs_rows:
            prob.chgRHS(self.rhs_rows, self.rhs_vals)
        if self.rng_rows:
            prob.chgRHSRange(self.rng_rows, self.rng_vals)
        for row, c1, c2, val in self.quad_updates:
            prob.chgQRowCoeff(row, c1, c2, val)
        if self.nl_old_cons:
            prob.delConstraint(self.nl_old_cons)
            prob.addConstraint(self.nl_new_cons)


class _MutableConstraint:
    """Handles mutable-param updates for a constraint row.

    LP/QP path (nl_expr is None): targeted chgMCoef/chgRHS/chgQRowCoeff updates.
    NL path (nl_expr set): full row rebuild -- re-walks nl_expr only; reconstructs
    the rest from a precomputed stable xp expression and parallel coef lists.

    generate_standard_repn guarantees unique variable entries so parallel lists
    suffice -- no dict accumulation needed.

    _rhs_expr: bound - body_constant (Pyomo expr). For NL, always set (even when
               non-mutable) since it is needed for the rebuild's xp.constraint call.
    _rng_expr: ub - lb for range constraints (Pyomo expr). Same rule as _rhs_expr.
    _type:     xp constraint type (leq/geq/eq/rng), stored for NL rebuild only.
    _stable_xp: precomputed xp expression for the non-mutable lin+quad body terms.
                None for LP/QP (not needed for targeted updates).
    """

    __slots__ = (
        '_con',
        '_xp_con',
        '_lin_vars',
        '_lin_coefs',
        '_quad_v1s',
        '_quad_v2s',
        '_quad_coefs',
        '_rhs_expr',
        '_rng_expr',
        '_type',
        '_stable_xp',
        '_nl_expr',
    )

    def __init__(
        self,
        con,
        xp_con,
        lin_vars: list,
        lin_coefs: list,
        quad_v1s: list,
        quad_v2s: list,
        quad_coefs: list,
        rhs_expr: Any,
        rng_expr: Any,
        con_type,
        stable_xp: Any,
        nl_expr: Any,
    ) -> None:
        # ConstraintData; needed to update maps.cons after NL rebuild
        self._con = con
        # xp.constraint handle; updated in-place after each NL rebuild
        self._xp_con = xp_con
        # mutable linear xp.var handles (parallel with _lin_coefs)
        self._lin_vars = lin_vars
        # mutable linear Pyomo coef expressions
        self._lin_coefs = lin_coefs
        # mutable quad var handles, parallel triplet (_quad_v1s, _quad_v2s, _quad_coefs)
        self._quad_v1s = quad_v1s
        self._quad_v2s = quad_v2s
        self._quad_coefs = quad_coefs
        self._rhs_expr = rhs_expr
        self._rng_expr = rng_expr
        self._type = con_type
        self._stable_xp = stable_xp
        # repn.nonlinear_expr; None for LP/QP, set for NL constraints
        self._nl_expr = nl_expr

    def collect(self, batch: _UpdateBatch, walker, use_names: bool, maps) -> None:
        if self._nl_expr is not None:
            batch.nl_old_cons.append(self._xp_con)
            self._xp_con = self._make_xp_con(walker, use_names)
            maps.cons[self._con] = self._xp_con
            batch.nl_new_cons.append(self._xp_con)
            return
        row = self._xp_con
        if self._lin_vars:
            batch.coef_rows.extend([row] * len(self._lin_vars))
            batch.coef_cols.extend(self._lin_vars)
            batch.coef_vals.extend(value(c) for c in self._lin_coefs)
        if self._rhs_expr is not None:
            batch.rhs_rows.append(row)
            batch.rhs_vals.append(value(self._rhs_expr))
        if self._rng_expr is not None:
            batch.rng_rows.append(row)
            batch.rng_vals.append(value(self._rng_expr))
        if self._quad_v1s:
            batch.quad_updates.extend(
                (row, v1, v2, value(c))
                for v1, v2, c in zip(self._quad_v1s, self._quad_v2s, self._quad_coefs)
            )

    def _make_xp_con(self, walker, use_names: bool) -> Any:
        nl_expr = (
            walker.walk_expression(self._nl_expr) if self._nl_expr is not None else None
        )
        return xp.constraint(
            body=_assemble_xp_expr(
                stable_xp=self._stable_xp,
                lin_vars=self._lin_vars,
                lin_coefs=self._lin_coefs,
                quad_v1s=self._quad_v1s,
                quad_v2s=self._quad_v2s,
                quad_coefs=self._quad_coefs,
                constant=None,
                nl_expr=nl_expr,
            ),
            rhs=value(self._rhs_expr),
            rhsrange=value(self._rng_expr),
            type=self._type,
            name=self._con.name if use_names else None,
        )


def _assemble_xp_expr(
    stable_xp,
    lin_vars: list,
    lin_coefs: list,
    quad_v1s: list,
    quad_v2s: list,
    quad_coefs: list,
    constant,
    nl_expr,
) -> Any:
    """Assemble a single Xpress expression from precomputed parts.

    Coefs may be Pyomo expressions or pre-evaluated floats; value() is called on
    each one. All vars must already be xp.var handles.
    stable_xp is a precomputed xp expression (or None if absent).
    constant is a Pyomo expression, float, or None; value() is called if not None.
    nl_expr is a walked xp expression (or None).

    Uses xp.Sum([...]) so stable_xp is never aliased or mutated.
    """
    parts = []
    if stable_xp is not None:
        parts.append(stable_xp)
    parts.extend(value(c) * v for c, v in zip(lin_coefs, lin_vars))
    parts.extend(
        value(c) * v1 * v2 for c, v1, v2 in zip(quad_coefs, quad_v1s, quad_v2s)
    )
    if constant is not None:
        parts.append(value(constant))
    if nl_expr is not None:
        parts.append(nl_expr)
    return xp.Sum(parts) if len(parts) > 0 else 0.0


class _MutableObjective:
    """Handles mutable-param updates for the objective function.

    Mirrors _MutableConstraint: both LP/QP and NL paths share parallel coef lists.
    NL additionally carries _stable_xp and _nl_expr (re-walked on every update).

    LP/QP: chgObj(keys+[-1], vals+[const]) + chgMQObj (Hessian-scaled at update time).
    NL:    setObjective(stable_xp + mutable_terms + const + nl_part).
    """

    __slots__ = (
        '_lin_vars',
        '_lin_coefs',
        '_quad_v1s',
        '_quad_v2s',
        '_quad_coefs',
        '_constant',
        '_stable_xp',
        '_nl_expr',
    )

    def __init__(
        self,
        lin_vars: list,
        lin_coefs: list,
        quad_v1s: list,
        quad_v2s: list,
        quad_coefs: list,
        constant,
        stable_xp: Any,
        nl_expr: Any,
    ) -> None:
        # mutable linear xp.var handles and Pyomo coef expressions (parallel)
        self._lin_vars = lin_vars
        self._lin_coefs = lin_coefs
        # mutable quad var handles and coefs, parallel triplet
        self._quad_v1s = quad_v1s
        self._quad_v2s = quad_v2s
        self._quad_coefs = quad_coefs
        # LP/QP: mutable Pyomo expr or None; NL: repn.constant always (may be 0)
        self._constant = constant
        # non-mutable lin+quad body as a precomputed xp expr; None for LP/QP
        self._stable_xp = stable_xp
        # repn.nonlinear_expr; None for LP/QP objectives
        self._nl_expr = nl_expr

    def update(self, prob, walker: 'XpressExpressionWalker') -> None:
        if self._nl_expr is None:
            if self._constant is not None:
                prob.chgObj([-1], [-value(self._constant)])
            if len(self._lin_vars) > 0:
                vals = [value(c) for c in self._lin_coefs]
                prob.chgObj(self._lin_vars, vals)
            if len(self._quad_v1s) > 0:
                # Xpress stores the QP objective as (1/2)*x'Qx, so chgMQObj
                # expects Hessian-scaled values. Off-diagonal entries appear twice
                # in the symmetric Hessian (which compensates), diagonal entries
                # must be doubled to match the user-visible coefficient.
                prob.chgMQObj(
                    self._quad_v1s,
                    self._quad_v2s,
                    [
                        value(c) * (2 if v1 is v2 else 1)
                        for v1, v2, c in zip(
                            self._quad_v1s, self._quad_v2s, self._quad_coefs
                        )
                    ],
                )
        else:
            prob.setObjective(
                _assemble_xp_expr(
                    stable_xp=self._stable_xp,
                    lin_vars=self._lin_vars,
                    lin_coefs=self._lin_coefs,
                    quad_v1s=self._quad_v1s,
                    quad_v2s=self._quad_v2s,
                    quad_coefs=self._quad_coefs,
                    constant=self._constant,
                    nl_expr=walker.walk_expression(self._nl_expr),
                )
            )


class XpressPersistentSolutionLoader(XpressSolutionLoaderBase):
    """Solution loader for XpressPersistent -- invalidated before each re-solve."""

    def __init__(
        self,
        prob,
        pyomo_model: BlockData,
        variables: list[VarData],
        maps: EntityMaps,
        pool_solutions: Optional[list] = None,
    ) -> None:
        super().__init__(prob, pyomo_model, variables, maps, pool_solutions)
        self._valid = True

    def invalidate(self) -> None:
        self._valid = False

    def _assert_valid(self) -> None:
        if not self._valid:
            raise NoSolutionError(
                'The results from the previous solve are no longer valid because '
                'the model has been modified.'
            )

    def load_vars(self, vars_to_load: Sequence[VarData] | None = None) -> None:
        self._assert_valid()
        return super().load_vars(vars_to_load)

    def get_vars(
        self, vars_to_load: Sequence[VarData] | None = None
    ) -> Mapping[VarData, float]:
        self._assert_valid()
        return super().get_vars(vars_to_load)

    def get_duals(
        self, cons_to_load: Sequence[ConstraintData] | None = None
    ) -> dict[ConstraintData, float]:
        self._assert_valid()
        return super().get_duals(cons_to_load)

    def get_reduced_costs(
        self, vars_to_load: Sequence[VarData] | None = None
    ) -> Mapping[VarData, float]:
        self._assert_valid()
        return super().get_reduced_costs(vars_to_load)


class XpressPersistentConfig(BranchAndBoundConfig):
    """Configuration for XpressPersistent."""

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
        self.auto_updates = self.declare('auto_updates', AutoUpdateConfig())
        self.warmstart: bool = self.declare(
            'warmstart',
            ConfigValue(
                default=True,
                domain=bool,
                description='Pass current integer variable values as a MIP warm start.',
            ),
        )
        self.pool_solutions: int = self.declare(
            'pool_solutions',
            ConfigValue(
                default=0,
                domain=NonNegativeInt,
                description=(
                    'MIP solution pool size (0 = disabled). '
                    'N > 0: keep a rolling window of the last N solutions found.'
                ),
            ),
        )


class XpressPersistent(XpressSolverMixin, PersistentSolverBase, Observer):
    """Persistent Xpress solver: reuses the xp.problem() across re-solves."""

    CONFIG = XpressPersistentConfig()

    def __init__(self, **kwds):
        super().__init__(**kwds)
        # active xp.problem(); None until set_instance
        self._xp_prob = None
        # model currently loaded; identity-checked on re-solve
        self._pyomo_model: Optional[BlockData] = None
        # ordered VarData list; used for batch unfix before repn calls
        self._vars: Optional[list[VarData]] = None
        # Pyomo->Xpress handle maps (vars, cons, sos)
        self._maps: Optional[EntityMaps] = None
        # active Pyomo objective; None if model has no objective
        self._objective: Optional[ObjectiveData] = None
        # invalidated on every model change; prevents stale solution reads
        self._last_solution_loader: Optional[XpressPersistentSolutionLoader] = None
        # watches model for changes and notifies this observer
        self._change_detector: Optional[ModelChangeDetector] = None
        # per-constraint update helpers (LP/QP targeted or NL rebuild)
        self._mutable_helpers: dict[ConstraintData, _MutableConstraint] = {}
        # None when objective has no mutable params
        self._mutable_objective: Optional[_MutableObjective] = None
        # drives symbolic names in Xpress (symbolic_solver_labels config)
        self._use_names: bool = False
        # lazily initialised on first NL walk; reused across all walks
        self._walker: XpressExpressionWalker | None = None

    def _clear(self):
        self._xp_prob = None
        self._pyomo_model = None
        self._vars = None
        self._maps = None
        self._objective = None
        self._last_solution_loader = None
        self._change_detector = None
        self._mutable_helpers = {}
        self._mutable_objective = None
        self._use_names = False
        self._walker = None

    def _invalidate_last_results(self):
        if self._last_solution_loader is not None:
            self._last_solution_loader.invalidate()
            self._last_solution_loader = None

    def _create_xpress_model(
        self, model: BlockData, config: BranchAndBoundConfig, timer: HierarchicalTimer
    ) -> tuple[Any, XpressPersistentSolutionLoader, bool]:
        self._invalidate_last_results()

        if model is self._pyomo_model:
            timer.start('update')
            self.update(timer=timer, auto_updates=config.auto_updates)
            timer.stop('update')
        else:
            timer.start('set_instance')
            self.set_instance(
                model,
                _t=timer,
                use_names=config.symbolic_solver_labels,
                auto_updates=config.auto_updates,
            )
            timer.stop('set_instance')

        assert self._vars is not None
        assert self._maps is not None
        has_obj = self._objective is not None
        xp_prob = self._xp_prob
        vars = self._vars

        pool: list = []
        if config.pool_solutions > 0:
            pool = _register_pool_collector(xp_prob, config.pool_solutions)

        self._last_solution_loader = XpressPersistentSolutionLoader(
            xp_prob, model, vars, self._maps, pool
        )

        if config.warmstart and (
            xp_prob.attributes.mipents > 0 or xp_prob.attributes.sets > 0
        ):
            entind = [i for i, var in enumerate(vars) if not var.is_continuous()]
            self._warmstart(self._xp_prob, vars, entind)

        return xp_prob, self._last_solution_loader, has_obj

    def set_instance(self, model, _t=None, use_names=False, auto_updates=None):
        self._clear()
        self._pyomo_model = model
        self._use_names = use_names
        self._xp_prob = xp.problem()
        self._maps = EntityMaps(vars={}, cons={}, sos={})
        self._vars = []

        detector_kwds = {} if auto_updates is None else auto_updates
        self._change_detector = ModelChangeDetector(
            model=model, observers=[self], **detector_kwds
        )

    def update(self, timer=None, auto_updates=None):
        if self._pyomo_model is None:
            raise RuntimeError('set_instance must be called before update')
        assert self._change_detector is not None
        detector_kwds = {} if auto_updates is None else auto_updates
        self._change_detector.update(timer=timer, **detector_kwds)

    def _add_variables(self, pyo_vars: list[VarData]) -> None:
        assert self._maps is not None
        assert self._vars is not None
        xp_vars = self._add_vars_impl(self._xp_prob, pyo_vars, self._use_names)
        self._maps.vars.update((id(pv), xv) for pv, xv in zip(pyo_vars, xp_vars))
        self._vars.extend(pyo_vars)

    def _remove_variables(self, pyo_vars: list[VarData]) -> None:
        assert self._maps is not None
        assert self._vars is not None
        self._xp_prob.delVariable([self._maps.vars.pop(id(var)) for var in pyo_vars])
        removed = {id(var) for var in pyo_vars}
        self._vars = [var for var in self._vars if id(var) not in removed]

    def _update_variables(self, variables: Mapping[VarData, Reason]) -> None:
        assert self._maps is not None
        self._invalidate_last_results()
        new_vars = []
        del_vars = []
        mod_vars = []
        for var, reason in variables.items():
            if reason & Reason.added:
                new_vars.append(var)
            elif reason & Reason.removed:
                del_vars.append(var)
            else:
                mod_vars.append(var)
        if len(new_vars) > 0:
            self._add_variables(new_vars)
        if len(del_vars) > 0:
            self._remove_variables(del_vars)
        if len(mod_vars) > 0:
            xp_vars = [self._maps.vars[id(var)] for var in mod_vars]
            self._set_var_bounds(self._xp_prob, mod_vars, xp_vars)
            self._set_var_types(self._xp_prob, mod_vars, xp_vars)

    # -----------------------------------------------------------------------
    # repn-based xp expression builder
    # -----------------------------------------------------------------------

    def _build_xp_from_repn(self, repn) -> Any:
        """Build an Xpress expression from a StandardRepn (linear + quadratic parts)."""
        var_map = self._maps.vars
        nl_expr = None
        if repn.nonlinear_expr is not None:
            if self._walker is None:
                self._walker = XpressExpressionWalker(
                    var_map, self._xp_prob, use_names=self._use_names
                )
            nl_expr = self._walker.walk_expression(repn.nonlinear_expr)

        return _assemble_xp_expr(
            stable_xp=None,
            lin_vars=[var_map[id(v)] for v in repn.linear_vars],
            lin_coefs=repn.linear_coefs,
            quad_v1s=[var_map[id(v1)] for v1, _ in repn.quadratic_vars],
            quad_v2s=[var_map[id(v2)] for _, v2 in repn.quadratic_vars],
            quad_coefs=repn.quadratic_coefs,
            constant=repn.constant,
            nl_expr=nl_expr,
        )

    # -----------------------------------------------------------------------
    # Mutable helper registration
    # -----------------------------------------------------------------------

    def _register_mutable_constraint(
        self, con: ConstraintData, xp_con, repn, lb, ub
    ) -> None:
        has_ub = ub is not None
        is_range = has_ub and lb is not None and not con.equality
        is_nl = repn.nonlinear_expr is not None

        rhs_ref = ub if has_ub else lb
        const_mutable = not _is_constant(repn.constant)
        mut_rhs = not _is_constant(rhs_ref)

        # NL always needs rhs/rng for rebuild; LP/QP only if mutable.
        if rhs_ref is None:
            rhs_expr = None
        elif is_nl or const_mutable or mut_rhs:
            rhs_expr = rhs_ref - repn.constant
            if not is_nl and _is_constant(rhs_expr):
                rhs_expr = None
        else:
            rhs_expr = None
        rng_expr = (
            (ub - lb)
            if is_range and (is_nl or not _is_constant(ub) or not _is_constant(lb))
            else None
        )

        if is_nl:
            stable_xp = 0.0
            con_type = _CON_TYPE_MAP[is_range, con.equality, has_ub]
        else:
            stable_xp = con_type = None

        lin_vars, lin_coefs = [], []
        var_map = self._maps.vars
        for coef, var in zip(repn.linear_coefs, repn.linear_vars):
            if not _is_constant(coef):
                lin_vars.append(var_map[id(var)])
                lin_coefs.append(coef)
            elif is_nl:
                stable_xp += value(coef) * var_map[id(var)]

        quad_v1s, quad_v2s, quad_coefs = [], [], []
        for coef, (v1, v2) in zip(repn.quadratic_coefs, repn.quadratic_vars):
            if not _is_constant(coef):
                quad_v1s.append(var_map[id(v1)])
                quad_v2s.append(var_map[id(v2)])
                quad_coefs.append(coef)
            elif is_nl:
                stable_xp += value(coef) * var_map[id(v1)] * var_map[id(v2)]

        if (
            is_nl
            or len(lin_vars) > 0
            or len(quad_v1s) > 0
            or rhs_expr is not None
            or rng_expr is not None
        ):
            self._mutable_helpers[con] = _MutableConstraint(
                con=con,
                xp_con=xp_con,
                lin_vars=lin_vars,
                lin_coefs=lin_coefs,
                quad_v1s=quad_v1s,
                quad_v2s=quad_v2s,
                quad_coefs=quad_coefs,
                rhs_expr=rhs_expr,
                rng_expr=rng_expr,
                con_type=con_type,
                stable_xp=stable_xp,
                nl_expr=repn.nonlinear_expr,
            )

    # -----------------------------------------------------------------------
    # Constraint management
    # -----------------------------------------------------------------------

    def _add_constraints(self, pyo_cons: list[ConstraintData]) -> None:
        assert self._maps is not None
        if len(pyo_cons) == 0:
            return
        xp_cons = []
        fixed_vars = _collect_fixed(self._vars)
        try:
            for con in pyo_cons:
                lb, body, ub = con.to_bounded_expression()
                repn = generate_standard_repn(body, compute_values=False)
                name = con.name if self._use_names else None
                xp_expr = self._build_xp_from_repn(repn)

                vlb, vub = value(lb), value(ub)
                xp_con = xp.constraint(body=xp_expr, lb=vlb, ub=vub, name=name)
                xp_cons.append(xp_con)
                self._maps.cons[con] = xp_con
                self._register_mutable_constraint(con, xp_con, repn, lb, ub)
        finally:
            _refix(fixed_vars)
        self._xp_prob.addConstraint(xp_cons)

    def _remove_constraints(self, cons: list[ConstraintData]) -> None:
        assert self._maps is not None
        self._xp_prob.delConstraint([self._maps.cons.pop(con) for con in cons])
        for con in cons:
            self._mutable_helpers.pop(con, None)

    def _add_sos_constraints(self, cons: list[SOSConstraintData]) -> None:
        assert self._maps is not None
        xp_sos = self._add_sos_impl(
            self._xp_prob, cons, self._maps.vars, self._use_names
        )
        self._maps.sos.update(zip(cons, xp_sos))

    def _remove_sos_constraints(self, cons: list[SOSConstraintData]) -> None:
        assert self._maps is not None
        xp_sos = [self._maps.sos.pop(con) for con in cons]
        self._xp_prob.delSOS(xp_sos)

    # -----------------------------------------------------------------------
    # Objective management
    # -----------------------------------------------------------------------

    def _clear_objective(self):
        self._xp_prob.delObj(0)

    def _set_objective(self, obj: ObjectiveData | None) -> None:
        assert self._maps is not None
        self._objective = obj
        self._mutable_objective = None
        self._clear_objective()
        if obj is None:
            return

        fixed_vars = _collect_fixed(self._vars)
        repn = generate_standard_repn(obj.expr, compute_values=False)
        _refix(fixed_vars)

        sense = _OBJ_SENSE_MAP[obj.sense]
        self._xp_prob.setObjective(self._build_xp_from_repn(repn), sense=sense)

        is_nl = repn.nonlinear_expr is not None
        lin_vars, lin_coefs, stable_lin = [], [], []
        for coef, var in zip(repn.linear_coefs, repn.linear_vars):
            xp_var = self._maps.vars[id(var)]
            if not _is_constant(coef):
                lin_vars.append(xp_var)
                lin_coefs.append(coef)
            elif is_nl:
                stable_lin.append(value(coef) * xp_var)

        quad_v1s, quad_v2s, quad_coefs, stable_quad = [], [], [], []
        for coef, (v1, v2) in zip(repn.quadratic_coefs, repn.quadratic_vars):
            xp_v1 = self._maps.vars[id(v1)]
            xp_v2 = self._maps.vars[id(v2)]
            if not _is_constant(coef):
                quad_v1s.append(xp_v1)
                quad_v2s.append(xp_v2)
                quad_coefs.append(coef)
            elif is_nl:
                stable_quad.append(value(coef) * xp_v1 * xp_v2)

        stable_xp = (
            xp.Sum(stable_lin + stable_quad)
            if is_nl and (stable_lin or stable_quad)
            else 0.0
        )
        constant = repn.constant if is_nl or not _is_constant(repn.constant) else None

        if lin_vars or quad_v1s or constant is not None or is_nl:
            self._mutable_objective = _MutableObjective(
                lin_vars=lin_vars,
                lin_coefs=lin_coefs,
                quad_v1s=quad_v1s,
                quad_v2s=quad_v2s,
                quad_coefs=quad_coefs,
                constant=constant,
                stable_xp=stable_xp if is_nl else None,
                nl_expr=repn.nonlinear_expr,
            )

    def _update_constraints(self, cons: Mapping[ConstraintData, Reason]) -> None:
        self._invalidate_last_results()
        old_cons = [c for c, r in cons.items() if r & (Reason.removed | Reason.expr)]
        new_cons = [c for c, r in cons.items() if r & (Reason.added | Reason.expr)]
        if len(old_cons) > 0:
            self._remove_constraints(old_cons)
        if len(new_cons) > 0:
            self._add_constraints(new_cons)

    def _update_sos_constraints(self, cons: Mapping[SOSConstraintData, Reason]) -> None:
        self._invalidate_last_results()
        old_sos = [
            s for s, r in cons.items() if r & (Reason.removed | Reason.sos_items)
        ]
        new_sos = [s for s, r in cons.items() if r & (Reason.added | Reason.sos_items)]
        if len(old_sos) > 0:
            self._remove_sos_constraints(old_sos)
        if len(new_sos) > 0:
            self._add_sos_constraints(new_sos)

    def _update_objectives(self, objs: Mapping[ObjectiveData, Reason]) -> None:
        assert self._pyomo_model is not None
        self._invalidate_last_results()
        any_added = False
        for obj, reason in objs.items():
            if reason & (Reason.added | Reason.expr):
                self._set_objective(obj)
                any_added = True
            elif reason & Reason.removed:
                if obj is self._objective:
                    self._set_objective(None)
            elif reason & Reason.sense:
                self._xp_prob.chgObjSense(_OBJ_SENSE_MAP[obj.sense])
        if any_added:
            try:
                get_objective(self._pyomo_model)
            except ValueError as e:
                raise IncompatibleModelError(
                    'Xpress supports at most one active objective. '
                    'Deactivate extras with obj.deactivate().'
                ) from e

    def _update_parameters(self, params: Mapping[ParamData, Reason]) -> None:
        if self._change_detector is None:
            return
        assert self._maps is not None
        self._invalidate_last_results()
        cd = self._change_detector
        affected_cons = set()
        affected_vars = {}
        affected_obj = False
        for p, reason in params.items():
            if not (reason & Reason.value):  # type: ignore[operator]
                continue
            affected_cons.update(cd.get_constraints_impacted_by_param(p))
            if not affected_obj and len(cd.get_objectives_impacted_by_param(p)) > 0:
                affected_obj = True
            for var in cd.get_variables_impacted_by_param(p):
                affected_vars[id(var)] = var
        if len(affected_vars) > 0:
            av = list(affected_vars.values())
            self._set_var_bounds(
                self._xp_prob, av, [self._maps.vars[id(var)] for var in av]
            )
        prob = self._xp_prob
        if len(affected_cons) > 0:
            batch = _UpdateBatch()
            walker, use_names, maps = self._walker, self._use_names, self._maps
            for con in affected_cons:
                if con in self._mutable_helpers:
                    self._mutable_helpers[con].collect(batch, walker, use_names, maps)
            batch.flush(prob)
        if affected_obj and self._objective is not None:
            if self._mutable_objective is not None:
                self._mutable_objective.update(prob, self._walker)
            else:
                self._set_objective(self._objective)

    def add_variables(self, variables: list[VarData]) -> None:
        assert self._maps is not None
        self._add_variables(variables)

    def remove_variables(self, variables: list[VarData]) -> None:
        assert self._maps is not None
        self._remove_variables(variables)

    def update_variables(self, variables: list[VarData]) -> None:
        assert self._change_detector is not None
        self._change_detector.update_variables(variables)

    def add_constraints(self, cons: list[ConstraintData]) -> None:
        assert self._change_detector is not None
        self._change_detector.add_constraints(cons)

    def remove_constraints(self, cons: list[ConstraintData]) -> None:
        assert self._change_detector is not None
        self._change_detector.remove_constraints(cons)

    def add_sos_constraints(self, cons: list[SOSConstraintData]) -> None:
        assert self._change_detector is not None
        self._change_detector.add_sos_constraints(cons)

    def remove_sos_constraints(self, cons: list[SOSConstraintData]) -> None:
        assert self._change_detector is not None
        self._change_detector.remove_sos_constraints(cons)

    def set_objective(self, obj: ObjectiveData) -> None:
        assert self._change_detector is not None
        self._change_detector.add_objectives([obj])

    def update_parameters(self, params: Optional[Collection[ParamData]] = None) -> None:
        assert self._change_detector is not None
        self._change_detector.update_parameters(params)

    def add_block(self, block: BlockData) -> None:
        assert self._change_detector is not None
        new_cons = list(
            block.component_data_objects(Constraint, descend_into=True, active=True)
        )
        new_sos = list(
            block.component_data_objects(SOSConstraint, descend_into=True, active=True)
        )
        if len(new_cons) > 0:
            self._change_detector.add_constraints(new_cons)
        if len(new_sos) > 0:
            self._change_detector.add_sos_constraints(new_sos)

    def remove_block(self, block: BlockData) -> None:
        assert self._change_detector is not None
        old_cons = list(
            block.component_data_objects(Constraint, descend_into=True, active=True)
        )
        old_sos = list(
            block.component_data_objects(SOSConstraint, descend_into=True, active=True)
        )
        if len(old_cons) > 0:
            self._change_detector.remove_constraints(old_cons)
        if len(old_sos) > 0:
            self._change_detector.remove_sos_constraints(old_sos)

    def has_instance(self) -> bool:
        """Return True if set_instance has been called and a model is loaded."""
        return self._pyomo_model is not None

    def get_xpress_problem(self) -> Any:
        """Return the underlying xp.problem object for direct Xpress API access."""
        assert self._xp_prob is not None
        return self._xp_prob

    def get_xpress_control(self, *args) -> Any:
        assert self._xp_prob is not None
        return self._xp_prob.getControl(*args)

    def set_xpress_control(self, *args) -> None:
        assert self._xp_prob is not None
        self._xp_prob.setControl(*args)

    def get_xpress_attribute(self, *args) -> Any:
        assert self._xp_prob is not None
        return self._xp_prob.getAttrib(*args)

    def get_xpress_var(self, var: VarData) -> Any:
        """Return the xp.var handle for a Pyomo variable."""
        assert self._maps is not None
        return self._maps.vars[id(var)]

    def get_xpress_constraint(self, con: ConstraintData) -> Any:
        """Return the xp.constraint handle for a Pyomo constraint."""
        assert self._maps is not None
        return self._maps.cons[con]

    def get_xpress_sos(self, con: SOSConstraintData) -> Any:
        """Return the xp.sos handle for a Pyomo SOS constraint."""
        assert self._maps is not None
        return self._maps.sos[con]

    def release(self) -> None:
        """Drop the xp.problem and release all solver resources."""
        self._clear()

    def reset(self) -> None:
        """Clear all model data from the xp.problem, then drop it."""
        if self._xp_prob is not None:
            self._xp_prob.reset()
        self._clear()

    def write(self, filename: str, flags: str = '') -> None:
        """Write the loaded Xpress problem to a file."""
        self._xp_prob.writeProb(filename, flags)

    def write_iis(self, filename: str) -> str:
        """Compute the IIS and write it to filename in LP format.

        Must be called after an infeasible solve.  Raises if no model is
        loaded or if Xpress cannot compute an IIS.

        Returns the filename written.
        """
        assert self._xp_prob is not None
        self._xp_prob.firstIIS(1)
        self._xp_prob.writeIIS(1, 0, filename, 'l')
        return filename

    def get_iis(self) -> dict:
        """Compute the IIS and return the conflicting Pyomo objects.

        Must be called after an infeasible solve.  Raises if no model is
        loaded or if Xpress cannot compute an IIS.

        Returns a dict with keys:
            'constraints': list[ConstraintData] -- Pyomo constraints in the IIS
            'variables':   list[VarData]         -- Pyomo variables whose bounds
                                                    contribute to the IIS
        """
        assert self._xp_prob is not None and self._maps is not None
        self._xp_prob.firstIIS(1)
        rowind, colind, *_ = self._xp_prob.getIISData(1)
        col_to_var = {i: pv for i, pv in enumerate(self._vars)}
        row_to_con = {xc.index: pc for pc, xc in self._maps.cons.items()}
        return {
            'constraints': [row_to_con[r] for r in rowind if r in row_to_con],
            'variables': [col_to_var[c] for c in colind if c in col_to_var],
        }
