from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence
from typing import Any, Optional, Protocol, TypeVar, Union

from pyomo.common.enums import ObjectiveSense
from pyomo.common.numeric_types import value
from pyomo.core.base.constraint import ConstraintData
from pyomo.core.base.objective import ObjectiveData
from pyomo.core.base.var import VarData
from pyomo.repn.standard_repn import generate_standard_repn

from .api import knitro
from .package import Package
from .typing import (
    Atom,
    BoundType,
    Callback,
    Request,
    Result,
    StructureType,
    UnreachableError,
    ValueType,
)
from .utils import NonlinearExpressionData


class _Callback(Protocol):
    _atom: Atom

    def func(self, req: Request, res: Result) -> int: ...
    def grad(self, req: Request, res: Result) -> int: ...
    def hess(self, req: Request, res: Result) -> int: ...

    def expand(self) -> Callback:
        procs = (self.func, self.grad, self.hess)
        return Callback(*map(self._expand, procs))

    @staticmethod
    def _expand(proc: Callable[[Request, Result], int]):
        def _expanded(
            kc: Any, cb: Any, req: Request, res: Result, user_data: Any = None
        ) -> int:
            return proc(req, res)

        return _expanded


class _ObjectiveCallback(_Callback):
    def __init__(self, atom: Atom) -> None:
        self._atom = atom

    def func(self, req: Request, res: Result) -> int:
        res.obj = self._atom.func(req.x)
        return 0

    def grad(self, req: Request, res: Result) -> int:
        res.objGrad[:] = self._atom.grad(req.x)
        return 0

    def hess(self, req: Request, res: Result) -> int:
        res.hess[:] = self._atom.hess(req.x, req.sigma)
        return 0


class _ConstraintCallback(_Callback):
    i: int

    def __init__(self, i: int, atom: Atom) -> None:
        self.i = i
        self._atom = atom

    def func(self, req: Request, res: Result) -> int:
        res.c[:] = [self._atom.func(req.x)]
        return 0

    def grad(self, req: Request, res: Result) -> int:
        res.jac[:] = self._atom.grad(req.x)
        return 0

    def hess(self, req: Request, res: Result) -> int:
        res.hess[:] = self._atom.hess(req.x, req.lambda_[self.i])
        return 0


def parse_bounds(
    items: Union[Iterable[VarData], Iterable[ConstraintData]],
    idx_map: Mapping[int, int],
) -> Mapping[BoundType, MutableMapping[int, float]]:
    bounds_map = {bnd_type: {} for bnd_type in BoundType}
    for item in items:
        i = idx_map[id(item)]
        if isinstance(item, VarData):
            if item.fixed:
                bounds_map[BoundType.EQ][i] = value(item.value)
                continue
            if item.has_lb():
                bounds_map[BoundType.LO][i] = value(item.lb)
            if item.has_ub():
                bounds_map[BoundType.UP][i] = value(item.ub)
        elif isinstance(item, ConstraintData):
            if item.equality:
                bounds_map[BoundType.EQ][i] = value(item.lower)
                continue
            if item.has_lb():
                bounds_map[BoundType.LO][i] = value(item.lower)
            if item.has_ub():
                bounds_map[BoundType.UP][i] = value(item.upper)
    return bounds_map


def parse_var_types(
    variables: Iterable[VarData], idx_map: Mapping[int, int]
) -> Mapping[int, int]:
    var_types = {}
    for var in variables:
        i = idx_map[id(var)]
        if var.is_binary():
            var_types[i] = knitro.KN_VARTYPE_BINARY
        elif var.is_integer():
            var_types[i] = knitro.KN_VARTYPE_INTEGER
        elif var.is_continuous():
            var_types[i] = knitro.KN_VARTYPE_CONTINUOUS
        else:
            raise ValueError(f"Unsupported variable type for variable {var.name}.")
    return var_types


def get_param_setter(param_type: int) -> Callable[..., None]:
    if param_type == knitro.KN_PARAMTYPE_INTEGER:
        return knitro.KN_set_int_param
    elif param_type == knitro.KN_PARAMTYPE_FLOAT:
        return knitro.KN_set_double_param
    elif param_type == knitro.KN_PARAMTYPE_STRING:
        return knitro.KN_set_char_param
    raise UnreachableError()


def get_value_getter(
    item_type: type, value_type: ValueType
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
    raise UnreachableError()


def get_item_adder(item_type: type) -> Callable[..., Optional[list[int]]]:
    if item_type is VarData:
        return knitro.KN_add_vars
    elif item_type is ConstraintData:
        return knitro.KN_add_cons
    raise UnreachableError()


def get_bound_setter(item_type: type, bound_type: BoundType) -> Callable[..., None]:
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
    raise UnreachableError()


def get_structure_adder(
    is_obj: bool, structure_type: StructureType
) -> Callable[..., None]:
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


class Engine:
    """A wrapper around the KNITRO API for a single optimization problem."""

    has_objective: bool
    var_map: MutableMapping[int, int]
    con_map: MutableMapping[int, int]
    nonlinear_map: MutableMapping[Optional[int], NonlinearExpressionData]
    nonlinear_diff_order: int

    _kc: Optional[Any]
    _status: Optional[int]

    def __init__(self, *, nonlinear_diff_order: int = 2) -> None:
        self.var_map = {}
        self.con_map = {}
        self.nonlinear_map = {}
        self.has_objective = False
        self.nonlinear_diff_order = nonlinear_diff_order
        self._kc = None
        self._status = None

    def __del__(self) -> None:
        self.close()

    def renew(self) -> None:
        self.close()
        self._kc = Package.create_context()
        self.var_map.clear()
        self.con_map.clear()
        self.nonlinear_map.clear()
        self.has_objective = False
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
        self.add_items(VarData, variables, self.var_map)
        self.set_var_types(variables)
        self.set_bounds(VarData, variables, self.var_map)

    def add_cons(self, cons: Sequence[ConstraintData]) -> None:
        self.add_items(ConstraintData, cons, self.con_map)
        self.set_bounds(ConstraintData, cons, self.con_map)
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
            raise RuntimeError("Solver has not been run, so no status is available.")
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
        self, item_type: type, items: Union[Iterable[VarData], Iterable[ConstraintData]]
    ) -> list[int]:
        if item_type is VarData:
            return [self.var_map[id(var)] for var in items]
        elif item_type is ConstraintData:
            return [self.con_map[id(con)] for con in items]
        raise UnreachableError()

    def get_values(
        self,
        item_type: type,
        value_type: ValueType,
        items: Union[Sequence[VarData], Sequence[ConstraintData]],
    ) -> Optional[list[float]]:
        getter = get_value_getter(item_type, value_type)
        idxs = self.get_idxs(item_type, items)
        return self.execute(getter, idxs)

    def set_option(self, param: str, val) -> None:
        param_id = self.execute(knitro.KN_get_param_id, param)
        param_type = self.execute(knitro.KN_get_param_type, param_id)
        func = get_param_setter(param_type)
        self.execute(func, param_id, val)

    def set_obj_goal(self, sense: ObjectiveSense) -> None:
        obj_goal = (
            knitro.KN_OBJGOAL_MINIMIZE
            if sense == ObjectiveSense.minimize
            else knitro.KN_OBJGOAL_MAXIMIZE
        )
        self.execute(knitro.KN_set_obj_goal, obj_goal)

    def add_items(
        self,
        item_type: type,
        items: Union[Sequence[VarData], Sequence[ConstraintData]],
        idx_map: MutableMapping[int, int],
    ) -> None:
        func = get_item_adder(item_type)
        idxs = self.execute(func, len(items))
        if idxs is not None:
            idx_map.update(zip(map(id, items), idxs))

    def set_bounds(
        self,
        item_type: type,
        items: Union[Sequence[VarData], Sequence[ConstraintData]],
        idx_map: MutableMapping[int, int],
    ) -> None:
        bounds_map = parse_bounds(items, idx_map)
        for bound_type, bounds in bounds_map.items():
            if not bounds:
                continue

            func = get_bound_setter(item_type, bound_type)
            self.execute(func, bounds.keys(), bounds.values())

    def set_var_types(self, variables: Iterable[VarData]) -> None:
        var_types = parse_var_types(variables, self.var_map)
        if var_types:
            func = knitro.KN_set_var_types
            self.execute(func, var_types.keys(), var_types.values())

    def set_con_structures(self, cons: Iterable[ConstraintData]) -> None:
        for con in cons:
            i = self.con_map[id(con)]
            self.add_structures(i, con.body)

    def set_obj_structures(self, obj: ObjectiveData) -> None:
        self.add_structures(None, obj.expr)

    def add_structures(self, i: Optional[int], expr) -> None:
        repn = generate_standard_repn(expr)
        if repn is None:
            return

        is_obj = i is None
        base_args = () if is_obj else (i,)
        structure_type_seq, args_seq = [], []

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
            func = get_structure_adder(is_obj, structure_type)
            self.execute(func, *base_args, *args)

        if repn.nonlinear_expr is not None:
            self.nonlinear_map[i] = NonlinearExpressionData(
                repn.nonlinear_expr,
                repn.nonlinear_vars,
                var_map=self.var_map,
                diff_order=self.nonlinear_diff_order,
            )

    def register_callbacks(self) -> None:
        for i, expr in self.nonlinear_map.items():
            self.register_callback(i, expr)

    def register_callback(
        self, i: Optional[int], expr: NonlinearExpressionData
    ) -> None:
        is_obj = i is None
        callback_type = _ObjectiveCallback if is_obj else _ConstraintCallback
        callback_args = ((i,) if not is_obj else ()) + (expr,)
        callback = callback_type(*callback_args).expand()

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
