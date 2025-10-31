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

from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence
from types import MappingProxyType
from typing import Any, Optional, TypeVar

from pyomo.common.enums import ObjectiveSense
from pyomo.common.errors import DeveloperError
from pyomo.common.numeric_types import value
from pyomo.contrib.solver.solvers.knitro.api import knitro
from pyomo.contrib.solver.solvers.knitro.callback import build_callback_handler
from pyomo.contrib.solver.solvers.knitro.package import Package
from pyomo.contrib.solver.solvers.knitro.typing import (
    BoundType,
    Callback,
    ItemData,
    ItemType,
    StructureType,
    ValueType,
)
from pyomo.contrib.solver.solvers.knitro.utils import NonlinearExpressionData
from pyomo.core.base.constraint import ConstraintData
from pyomo.core.base.objective import ObjectiveData
from pyomo.core.base.var import VarData
from pyomo.repn.standard_repn import generate_standard_repn


def parse_bounds(
    items: Iterable[ItemType], idx_map: Mapping[int, int]
) -> Mapping[BoundType, MutableMapping[int, float]]:
    bounds_map = {bnd_type: {} for bnd_type in BoundType}
    for item in items:
        i = idx_map[id(item)]
        if isinstance(item, VarData):
            if item.fixed:
                bounds_map[BoundType.EQ][i] = value(item.value)
                continue
            lb, ub = item.bounds
            if lb is not None:
                bounds_map[BoundType.LO][i] = lb
            if ub is not None:
                bounds_map[BoundType.UP][i] = ub
        elif isinstance(item, ConstraintData):
            lb, _, ub = item.to_bounded_expression(evaluate_bounds=True)
            if item.equality:
                bounds_map[BoundType.EQ][i] = lb
                continue
            if lb is not None:
                bounds_map[BoundType.LO][i] = lb
            if ub is not None:
                bounds_map[BoundType.UP][i] = ub
    return bounds_map


def parse_types(
    items: Iterable[ItemType], idx_map: Mapping[int, int]
) -> Mapping[int, int]:
    types_map = {}
    for item in items:
        i = idx_map[id(item)]
        if isinstance(item, VarData):
            if item.is_binary():
                types_map[i] = knitro.KN_VARTYPE_BINARY
            elif item.is_integer():
                types_map[i] = knitro.KN_VARTYPE_INTEGER
            elif item.is_continuous():
                types_map[i] = knitro.KN_VARTYPE_CONTINUOUS
            else:
                msg = f"Variable {item.name} has unsupported type."
                raise ValueError(msg)
    return types_map


def api_set_param(param_type: int) -> Callable[..., None]:
    if param_type == knitro.KN_PARAMTYPE_INTEGER:
        return knitro.KN_set_int_param
    elif param_type == knitro.KN_PARAMTYPE_FLOAT:
        return knitro.KN_set_double_param
    elif param_type == knitro.KN_PARAMTYPE_STRING:
        return knitro.KN_set_char_param
    raise DeveloperError(f"Unsupported KNITRO parameter type: {param_type}")


def api_get_values(
    item_type: type[ItemType], value_type: ValueType
) -> Callable[..., Optional[list[float]]]:
    if item_type is VarData:
        if value_type == ValueType.PRIMAL:
            return knitro.KN_get_var_primal_values
        elif value_type == ValueType.DUAL:
            return knitro.KN_get_var_dual_values
    elif item_type is ConstraintData:
        if value_type == ValueType.DUAL:
            return knitro.KN_get_con_dual_values
        elif value_type == ValueType.PRIMAL:
            return knitro.KN_get_con_values
    raise DeveloperError(
        f"Unsupported KNITRO item type or value type: {item_type}, {value_type}"
    )


def api_add_items(item_type: type[ItemType]) -> Callable[..., Optional[list[int]]]:
    if item_type is VarData:
        return knitro.KN_add_vars
    elif item_type is ConstraintData:
        return knitro.KN_add_cons
    raise DeveloperError(f"Unsupported KNITRO item type: {item_type}")


def api_set_bnds(
    item_type: type[ItemType], bound_type: BoundType
) -> Callable[..., None]:
    if item_type is VarData:
        if bound_type == BoundType.EQ:
            return knitro.KN_set_var_fxbnds
        elif bound_type == BoundType.LO:
            return knitro.KN_set_var_lobnds
        elif bound_type == BoundType.UP:
            return knitro.KN_set_var_upbnds
    elif item_type is ConstraintData:
        if bound_type == BoundType.EQ:
            return knitro.KN_set_con_eqbnds
        elif bound_type == BoundType.LO:
            return knitro.KN_set_con_lobnds
        elif bound_type == BoundType.UP:
            return knitro.KN_set_con_upbnds
    raise DeveloperError(
        f"Unsupported KNITRO item type or bound type: {item_type}, {bound_type}"
    )


def api_set_types(item_type: type[ItemType]) -> Callable[..., None]:
    if item_type is VarData:
        return knitro.KN_set_var_types
    raise DeveloperError(f"Unsupported KNITRO item type: {item_type}")


def api_add_struct(is_obj: bool, structure_type: StructureType) -> Callable[..., None]:
    if is_obj:
        if structure_type == StructureType.CONSTANT:
            return knitro.KN_add_obj_constant
        elif structure_type == StructureType.LINEAR:
            return knitro.KN_add_obj_linear_struct
        elif structure_type == StructureType.QUADRATIC:
            return knitro.KN_add_obj_quadratic_struct
    else:
        if structure_type == StructureType.CONSTANT:
            return knitro.KN_add_con_constants
        elif structure_type == StructureType.LINEAR:
            return knitro.KN_add_con_linear_struct
        elif structure_type == StructureType.QUADRATIC:
            return knitro.KN_add_con_quadratic_struct
    raise DeveloperError(
        f"Unsupported KNITRO structure type: is_obj={is_obj}, structure_type={structure_type}"
    )


class Engine:
    """A wrapper around the KNITRO API for a single optimization problem."""

    has_objective: bool
    maps: Mapping[type[ItemData], MutableMapping[int, int]]
    nonlinear_map: MutableMapping[Optional[int], NonlinearExpressionData]
    nonlinear_diff_order: int

    _kc: Optional[Any]
    _status: Optional[int]

    def __init__(self, *, nonlinear_diff_order: int = 2) -> None:
        self.has_objective = False
        # Maps:
        # VarData -> {id(var): idx in KNITRO}
        # ConstraintData -> {id(con): idx in KNITRO}
        self.maps = MappingProxyType({VarData: {}, ConstraintData: {}})
        # Nonlinear map:
        # None -> objective nonlinear expression
        # idx_con -> constraint nonlinear expression
        self.nonlinear_map = {}
        self.nonlinear_diff_order = nonlinear_diff_order
        self._kc = None
        self._status = None

    def __enter__(self) -> "Engine":
        self.renew()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def renew(self) -> None:
        self.close()
        self._kc = Package.create_context()
        self.has_objective = False
        for item_type in self.maps:
            self.maps[item_type].clear()
        self.nonlinear_map.clear()
        # TODO: remove this when the tolerance tests are fixed in test_solvers
        tol = 1e-8
        self.set_options(ftol=tol, opttol=tol, xtol=tol)

    def close(self) -> None:
        if self._kc is not None:
            self.execute(knitro.KN_free)
            self._kc = None

    T = TypeVar("T")

    def execute(self, api_func: Callable[..., T], *args, **kwargs) -> T:
        if self._kc is None:
            msg = "KNITRO context has not been initialized or has been freed."
            raise RuntimeError(msg)
        return api_func(self._kc, *args, **kwargs)

    def add_vars(self, variables: Sequence[VarData]) -> None:
        self.add_items(VarData, variables)
        self.set_types(VarData, variables)
        self.set_bounds(VarData, variables)

    def add_cons(self, cons: Sequence[ConstraintData]) -> None:
        self.add_items(ConstraintData, cons)
        self.set_bounds(ConstraintData, cons)
        self.set_con_structures(cons)

    def set_obj(self, obj: ObjectiveData) -> None:
        self.has_objective = True
        self.set_obj_goal(obj.sense)
        self.set_obj_structures(obj)

    def set_options(self, **options) -> None:
        for param, val in options.items():
            self.set_option(param, val)

    def set_outlev(self, level: Optional[int] = None) -> None:
        if level is None:
            level = knitro.KN_OUTLEV_ALL
        self.set_options(outlev=level)

    def set_time_limit(self, time_limit: float) -> None:
        self.set_options(maxtime_cpu=time_limit)

    def set_num_threads(self, nthreads: int) -> None:
        self.set_options(threads=nthreads)

    def solve(self) -> int:
        self.register_callbacks()
        self._status = self.execute(knitro.KN_solve)
        return self._status

    def get_idx_vars(self, variables: Iterable[VarData]) -> list[int]:
        return self.get_idxs(VarData, variables)

    def get_status(self) -> int:
        if self._status is None:
            msg = "Solver has not been run. No status is available!"
            raise RuntimeError(msg)
        return self._status

    def get_num_iters(self) -> int:
        return self.execute(knitro.KN_get_number_iters)

    def get_num_solutions(self) -> int:
        _, _, x, _ = self.execute(knitro.KN_get_solution)
        return 1 if x is not None else 0

    def get_solve_time(self) -> float:
        return self.execute(knitro.KN_get_solve_time_real)

    def get_obj_value(self) -> Optional[float]:
        if not self.has_objective:
            return None
        if self._status not in {
            knitro.KN_RC_OPTIMAL,
            knitro.KN_RC_OPTIMAL_OR_SATISFACTORY,
            knitro.KN_RC_NEAR_OPT,
            knitro.KN_RC_ITER_LIMIT_FEAS,
            knitro.KN_RC_FEAS_NO_IMPROVE,
            knitro.KN_RC_TIME_LIMIT_FEAS,
        }:
            return None
        return self.execute(knitro.KN_get_obj_value)

    def get_idxs(
        self, item_type: type[ItemType], items: Iterable[ItemType]
    ) -> list[int]:
        idx_map = self.maps[item_type]
        return [idx_map[id(item)] for item in items]

    def get_values(
        self,
        item_type: type[ItemType],
        value_type: ValueType,
        items: Iterable[ItemType],
    ) -> Optional[list[float]]:
        func = api_get_values(item_type, value_type)
        idxs = self.get_idxs(item_type, items)
        return self.execute(func, idxs)

    def set_option(self, param: str, val) -> None:
        param_id = self.execute(knitro.KN_get_param_id, param)
        param_type = self.execute(knitro.KN_get_param_type, param_id)
        func = api_set_param(param_type)
        self.execute(func, param_id, val)

    def set_obj_goal(self, sense: ObjectiveSense) -> None:
        obj_goal = (
            knitro.KN_OBJGOAL_MINIMIZE
            if sense == ObjectiveSense.minimize
            else knitro.KN_OBJGOAL_MAXIMIZE
        )
        self.execute(knitro.KN_set_obj_goal, obj_goal)

    def add_items(self, item_type: type[ItemType], items: Sequence[ItemType]) -> None:
        func = api_add_items(item_type)
        idxs = self.execute(func, len(items))
        if idxs is not None:
            self.maps[item_type].update(zip(map(id, items), idxs))

    def set_bounds(self, item_type: type[ItemType], items: Iterable[ItemType]) -> None:
        bounds_map = parse_bounds(items, self.maps[item_type])
        for bound_type, bounds in bounds_map.items():
            if not bounds:
                continue

            func = api_set_bnds(item_type, bound_type)
            self.execute(func, bounds.keys(), bounds.values())

    def set_types(self, item_type: type[ItemType], items: Iterable[ItemType]) -> None:
        types_map = parse_types(items, self.maps[item_type])
        if types_map:
            func = api_set_types(item_type)
            self.execute(func, types_map.keys(), types_map.values())

    def set_con_structures(self, cons: Iterable[ConstraintData]) -> None:
        for con in cons:
            i = self.maps[ConstraintData][id(con)]
            self.add_structures(i, con.body)

    def set_obj_structures(self, obj: ObjectiveData) -> None:
        self.add_structures(None, obj.expr)

    def add_structures(self, i: Optional[int], expr) -> None:
        repn = generate_standard_repn(expr)
        if repn is None:
            return

        is_obj = i is None
        base_args = () if is_obj else (i,)
        structure_type_seq: list[StructureType] = []
        args_seq: list[tuple[Any, ...]] = []

        if repn.constant is not None:
            structure_type_seq += [StructureType.CONSTANT]
            args_seq += [(repn.constant,)]

        if repn.linear_vars:
            idx_lin_vars = self.get_idx_vars(repn.linear_vars)
            lin_coefs = list(repn.linear_coefs)
            structure_type_seq += [StructureType.LINEAR]
            args_seq += [(idx_lin_vars, lin_coefs)]

        if repn.quadratic_vars:
            quad_vars1, quad_vars2 = zip(*repn.quadratic_vars)
            idx_quad_vars1 = self.get_idx_vars(quad_vars1)
            idx_quad_vars2 = self.get_idx_vars(quad_vars2)
            quad_coefs = list(repn.quadratic_coefs)
            structure_type_seq += [StructureType.QUADRATIC]
            args_seq += [(idx_quad_vars1, idx_quad_vars2, quad_coefs)]

        for structure_type, args in zip(structure_type_seq, args_seq):
            func = api_add_struct(is_obj, structure_type)
            self.execute(func, *base_args, *args)

        if repn.nonlinear_expr is not None:
            self.nonlinear_map[i] = NonlinearExpressionData(
                repn.nonlinear_expr,
                repn.nonlinear_vars,
                var_map=self.maps[VarData],
                diff_order=self.nonlinear_diff_order,
            )

    def add_callback(
        self, i: Optional[int], expr: NonlinearExpressionData, callback: Callback
    ) -> None:
        is_obj = i is None
        idx_cons = [i] if not is_obj else None
        cb = self.execute(knitro.KN_add_eval_callback, is_obj, idx_cons, callback.func)

        if expr.diff_order >= 1:
            idx_vars = self.get_idx_vars(expr.grad_vars)
            obj_grad_idx_vars = idx_vars if is_obj else None
            jac_idx_cons = [i] * len(idx_vars) if not is_obj else None
            jac_idx_vars = idx_vars if not is_obj else None
            self.execute(
                knitro.KN_set_cb_grad,
                cb,
                obj_grad_idx_vars,
                jac_idx_cons,
                jac_idx_vars,
                callback.grad,
            )

        if expr.diff_order >= 2:
            hess_vars1, hess_vars2 = zip(*expr.hess_vars)
            hess_idx_vars1 = self.get_idx_vars(hess_vars1)
            hess_idx_vars2 = self.get_idx_vars(hess_vars2)
            self.execute(
                knitro.KN_set_cb_hess, cb, hess_idx_vars1, hess_idx_vars2, callback.hess
            )

    def register_callbacks(self) -> None:
        for i, expr in self.nonlinear_map.items():
            self.register_callback(i, expr)

    def register_callback(
        self, i: Optional[int], expr: NonlinearExpressionData
    ) -> None:
        callback = build_callback_handler(expr, idx=i).expand()
        self.add_callback(i, expr, callback)
