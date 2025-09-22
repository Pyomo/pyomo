from collections.abc import Iterable, Mapping, MutableMapping
from typing import List, Optional, Type, Union

from pyomo.common.enums import Enum, ObjectiveSense
from pyomo.common.numeric_types import value
from pyomo.core.base.constraint import ConstraintData
from pyomo.core.base.objective import ObjectiveData
from pyomo.core.base.var import VarData
from pyomo.repn.standard_repn import generate_standard_repn

from .api import knitro
from .package import Package
from .utils import NonlinearExpressionData
from .solution import LoadType


class Engine:
    """
    A wrapper around the KNITRO API for a single optimization problem.

    This class manages the lifecycle of a KNITRO problem instance (`kc`),
    including building the problem by adding variables and constraints,
    setting options, solving, and freeing the context.
    """

    mapping: Mapping[Type[Union[VarData, ConstraintData]], MutableMapping[int, int]]
    nonlinear_map: MutableMapping[Optional[int], NonlinearExpressionData]
    differentiation_order: int

    _status: Optional[int]

    def __init__(self, *, differentiation_order: int = 2):
        # VarData: id(var) -> idx_var
        # ConstraintData: id(con) -> idx_con
        self.mapping = {VarData: {}, ConstraintData: {}}
        # None -> objective
        # idx_con -> constraint
        self.nonlinear_map = {}
        self.differentiation_order = differentiation_order

        # Saving the KNITRO context
        self._kc = None
        # KNITRO status after solve
        self._status = None

    # ============= LIFECYCLE MANAGEMENT =============

    def __del__(self):
        self.close()

    def renew(self):
        self.close()
        self._kc = Package.create_context()
        # TODO: remove this when the tolerance tests are fixed in test_solvers
        tol = 1e-8
        self.set_options(ftol=tol, opttol=tol, xtol=tol)

    def close(self):
        if self._kc is not None:
            self._execute(knitro.KN_free)
            self._kc = None

    # ============= PROBLEM BUILDING =================

    def add_vars(self, variables: Iterable[VarData]):
        self._add_items(VarData, variables)
        self._set_var_types(variables)
        self._set_bnds(VarData, variables)

    def add_cons(self, cons: Iterable[ConstraintData]):
        self._add_items(ConstraintData, cons)
        self._set_bnds(ConstraintData, cons)

        for con in cons:
            i = self.mapping[ConstraintData][id(con)]
            self._add_structs(i, con.body)

    def set_obj(self, obj: ObjectiveData):
        self._set_obj_goal(obj.sense)
        self._add_structs(None, obj.expr)

    # ============= CONFIGURATION ===================

    def set_options(self, **options):
        for param, val in options.items():
            param_id = self._execute(knitro.KN_get_param_id, param)
            param_type = self._execute(knitro.KN_get_param_type, param_id)
            func = self.api_set_param(param_type)
            self._execute(func, param_id, val)

    def set_outlev(self, level: Optional[int] = None):
        if level is None:
            level = knitro.KN_OUTLEV_ALL
        self.set_options(outlev=level)

    def set_time_limit(self, time_limit: float):
        self.set_options(maxtime_cpu=time_limit)

    def set_num_threads(self, nthreads: int):
        self.set_options(threads=nthreads)

    # ============= SOLVING =========================

    def solve(self) -> int:
        self._register_callbacks()
        self._status = self._execute(knitro.KN_solve)
        return self._status

    # ============= INDEX RETRIEVAL =================

    def get_idx_vars(self, variables: Iterable[VarData]) -> List[int]:
        return self._get_idxs(VarData, variables)

    def get_idx_cons(self, constraints: Iterable[ConstraintData]) -> List[int]:
        return self._get_idxs(ConstraintData, constraints)

    # ============= SOLUTION RETRIEVAL ==============

    def get_status(self) -> int:
        if self._status is None:
            msg = "Solver has not been run yet. Since the solver has not been executed, no status is available."
            raise RuntimeError(msg)
        return self._status

    def get_num_iters(self) -> int:
        return self._execute(knitro.KN_get_number_iters)

    def get_num_solutions(self) -> int:
        _, _, x, _ = self._execute(knitro.KN_get_solution)
        return 1 if x is not None else 0

    def get_solve_time(self) -> float:
        return self._execute(knitro.KN_get_solve_time_real)

    def get_obj_value(self) -> Optional[float]:
        return self._execute(knitro.KN_get_obj_value)

    def get_values(
        self, item_type: Type[LoadType], items: Iterable[LoadType], *, is_dual: bool
    ):
        func = self.api_get_values(item_type, is_dual)
        idxs = self._get_idxs(item_type, items)
        return self._execute(func, idxs)

    # ============= PRIVATE UTILITIES ===============

    def _execute(self, api_func, *args, **kwargs):
        if self._kc is None:
            msg = "KNITRO context has been freed or has not been initialized and cannot be used."
            raise RuntimeError(msg)
        return api_func(self._kc, *args, **kwargs)

    # ======== PRIVATE COMPONENT MANAGEMENT =========

    def _get_idxs(self, item_type: Type[LoadType], items: Iterable[LoadType]):
        imap = self.mapping[item_type]
        return [imap[id(item)] for item in items]

    def _add_items(self, item_type: Type[LoadType], items: Iterable[LoadType]):
        func = self.api_add_items(item_type)
        items_seq = list(items)
        idxs = self._execute(func, len(items_seq))
        if idxs is not None:
            self.mapping[item_type].update(zip(map(id, items_seq), idxs))

    # ====== PRIVATE VARIABLE TYPE HANDLING =========

    def _set_var_types(self, variables: Iterable[VarData]):
        var_types = {}
        func = knitro.KN_set_var_types
        for var in variables:
            i = self.mapping[VarData][id(var)]
            self._parse_var_type(var, i, var_types)
        self._execute(func, var_types.keys(), var_types.values())

    def _parse_var_type(
        self, var: VarData, i: int, var_types: MutableMapping[int, int]
    ) -> int:
        if var.is_binary():
            var_types[i] = knitro.KN_VARTYPE_BINARY
        elif var.is_integer():
            var_types[i] = knitro.KN_VARTYPE_INTEGER
        elif var.is_continuous():
            var_types[i] = knitro.KN_VARTYPE_CONTINUOUS
        else:
            msg = f"Unsupported variable type for variable {var.name}."
            raise ValueError(msg)

    # ========== PRIVATE BOUNDS HANDLING ============

    class _BndType(Enum):
        EQ = 0
        LO = 1
        UP = 2

    def _get_parse_bnds(self, item_type: Type[LoadType]):
        if item_type is VarData:
            return self._parse_var_bnds
        elif item_type is ConstraintData:
            return self._parse_con_bnds

    def _set_bnds(self, item_type: Type[LoadType], items: Iterable[LoadType]):
        parse = self._get_parse_bnds(item_type)
        bnds_map = {bnd_type: {} for bnd_type in self._BndType}
        imap = self.mapping[item_type]
        for item in items:
            parse(item, imap[id(item)], bnds_map)

        for bnd_type, bnds in bnds_map.items():
            if bnds:
                func = self.api_set_bnds(item_type, bnd_type)
                self._execute(func, bnds.keys(), bnds.values())

    def _parse_var_bnds(
        self,
        var: VarData,
        i: int,
        bnds_map: Mapping[_BndType, MutableMapping[int, float]],
    ):
        if var.fixed:
            bnds_map[Engine._BndType.EQ][i] = value(var.value)
        else:
            if var.has_lb():
                bnds_map[Engine._BndType.LO][i] = value(var.lb)
            if var.has_ub():
                bnds_map[Engine._BndType.UP][i] = value(var.ub)

    def _parse_con_bnds(
        self,
        con: ConstraintData,
        i: int,
        bnds_map: Mapping[_BndType, MutableMapping[int, float]],
    ):
        if con.equality:
            bnds_map[Engine._BndType.EQ][i] = value(con.lower)
        else:
            if con.has_lb():
                bnds_map[Engine._BndType.LO][i] = value(con.lower)
            if con.has_ub():
                bnds_map[Engine._BndType.UP][i] = value(con.upper)

    # ===== PRIVATE OBJECTIVE GOAL HANDLING =========

    def _set_obj_goal(self, sense: ObjectiveSense):
        obj_goal = (
            knitro.KN_OBJGOAL_MINIMIZE
            if sense == ObjectiveSense.minimize
            else knitro.KN_OBJGOAL_MAXIMIZE
        )
        self._execute(knitro.KN_set_obj_goal, obj_goal)

    # ======= PRIVATE STRUCTURE BUILDING ============

    def _add_structs(self, i: Optional[int], expr):
        repn = generate_standard_repn(expr)

        is_obj = i is None
        base_args = () if is_obj else (i,)

        funcs, args_seq = [], []
        if repn.constant is not None:
            func = self.api_add_constant(is_obj)
            funcs.append(func)
            args_seq.append((repn.constant,))
        if repn.linear_vars:
            func = self.api_add_linear_struct(is_obj)
            idx_lin_vars = self.get_idx_vars(repn.linear_vars)
            lin_coefs = list(repn.linear_coefs)
            funcs.append(func)
            args_seq.append((idx_lin_vars, lin_coefs))
        if repn.quadratic_vars:
            func = self.api_add_quadratic_struct(is_obj)
            quad_vars1, quad_vars2 = zip(*repn.quadratic_vars)
            idx_quad_vars1 = self.get_idx_vars(quad_vars1)
            idx_quad_vars2 = self.get_idx_vars(quad_vars2)
            quad_coefs = list(repn.quadratic_coefs)
            funcs.append(func)
            args_seq.append((idx_quad_vars1, idx_quad_vars2, quad_coefs))

        for func, args in zip(funcs, args_seq):
            self._execute(func, *base_args, *args)

        if repn.nonlinear_expr is not None:
            self.nonlinear_map[i] = NonlinearExpressionData(
                repn.nonlinear_expr,
                repn.nonlinear_vars,
                differentiation_order=self.differentiation_order,
            )

    # ======= PRIVATE CALLBACK HANDLING =============

    def _register_callbacks(self):
        for i, expr in self.nonlinear_map.items():
            self._register_callback(i, expr)

    def _register_callback(self, i: Optional[int], expr: NonlinearExpressionData):
        callback = self._add_callback(knitro.KN_RC_EVALFC, i, expr)
        if expr.grad is not None:
            self._add_callback(knitro.KN_RC_EVALGA, i, expr, callback)
        if expr.hess is not None:
            self._add_callback(knitro.KN_RC_EVALH, i, expr, callback)

    def _add_callback(
        self, eval_type: int, i: Optional[int], expr: NonlinearExpressionData, *args
    ):
        func = self.api_add_callback(eval_type)
        func_callback = self._build_callback(eval_type, i, expr)

        if eval_type == knitro.KN_RC_EVALH:
            hess_vars1, hess_vars2 = zip(*expr.hess_vars)
            hess_idx_vars1 = self.get_idx_vars(hess_vars1)
            hess_idx_vars2 = self.get_idx_vars(hess_vars2)
            args += (hess_idx_vars1, hess_idx_vars2)
        elif eval_type == knitro.KN_RC_EVALGA:
            idx_vars = self.get_idx_vars(expr.grad_vars)
            is_obj = i is None
            obj_grad_idx_vars = idx_vars if is_obj else None
            jac_idx_cons = [i] * len(idx_vars) if not is_obj else None
            jac_idx_vars = idx_vars if not is_obj else None
            args += (obj_grad_idx_vars, jac_idx_cons, jac_idx_vars)
        elif eval_type == knitro.KN_RC_EVALFC:
            eval_obj = i is None
            idx_cons = [i] if not eval_obj else None
            args += (eval_obj, idx_cons)

        return self._execute(func, *args, func_callback)

    def _build_callback(
        self, eval_type: int, i: Optional[int], expr: NonlinearExpressionData
    ):
        is_obj = i is None
        func = self._get_evaluator(eval_type, expr)

        if is_obj and eval_type == knitro.KN_RC_EVALFC:

            def _callback(req, res):
                res.obj = func(req.x)
                return 0

        elif is_obj and eval_type == knitro.KN_RC_EVALGA:

            def _callback(req, res):
                res.objGrad[:] = func(req.x)
                return 0

        elif is_obj and eval_type == knitro.KN_RC_EVALH:

            def _callback(req, res):
                res.hess[:] = func(req.x, req.sigma)
                return 0

        elif eval_type == knitro.KN_RC_EVALFC:

            def _callback(req, res):
                res.c[:] = [func(req.x)]
                return 0

        elif eval_type == knitro.KN_RC_EVALGA:

            def _callback(req, res):
                res.jac[:] = func(req.x)
                return 0

        elif eval_type == knitro.KN_RC_EVALH:

            def _callback(req, res):
                res.hess[:] = func(req.x, req.lambda_[i])
                return 0

        return lambda *args: _callback(args[2], args[3])

    def _get_evaluator(self, eval_type: int, expr: NonlinearExpressionData):
        vmap = self.mapping[VarData]
        if eval_type == knitro.KN_RC_EVALH:
            func = expr.create_hessian_evaluator
        elif eval_type == knitro.KN_RC_EVALGA:
            func = expr.create_gradient_evaluator
        elif eval_type == knitro.KN_RC_EVALFC:
            func = expr.create_evaluator
        return func(vmap)

    # ========= API FUNCTION GETTERS ================

    def api_set_param(self, param_type: int):
        if param_type == knitro.KN_PARAMTYPE_INTEGER:
            return knitro.KN_set_int_param
        elif param_type == knitro.KN_PARAMTYPE_FLOAT:
            return knitro.KN_set_double_param
        elif param_type == knitro.KN_PARAMTYPE_STRING:
            return knitro.KN_set_char_param

    def api_add_items(self, item_type: Type[LoadType]):
        if item_type is VarData:
            return knitro.KN_add_vars
        elif item_type is ConstraintData:
            return knitro.KN_add_cons

    def api_get_values(self, item_type: Type[LoadType], is_dual: bool):
        if item_type is VarData and not is_dual:
            return knitro.KN_get_var_primal_values
        elif item_type is VarData and is_dual:
            return knitro.KN_get_var_dual_values
        elif item_type is ConstraintData and is_dual:
            return knitro.KN_get_con_dual_values
        elif item_type is ConstraintData and not is_dual:
            return knitro.KN_get_con_values

    def api_set_bnds(self, item_type: Type[LoadType], bnd_type: _BndType):
        if item_type is VarData and bnd_type == Engine._BndType.EQ:
            return knitro.KN_set_var_fxbnds
        elif item_type is VarData and bnd_type == Engine._BndType.LO:
            return knitro.KN_set_var_lobnds
        elif item_type is VarData and bnd_type == Engine._BndType.UP:
            return knitro.KN_set_var_upbnds
        elif item_type is ConstraintData and bnd_type == Engine._BndType.EQ:
            return knitro.KN_set_con_eqbnds
        elif item_type is ConstraintData and bnd_type == Engine._BndType.LO:
            return knitro.KN_set_con_lobnds
        elif item_type is ConstraintData and bnd_type == Engine._BndType.UP:
            return knitro.KN_set_con_upbnds

    def api_add_constant(self, is_obj: bool):
        if is_obj:
            return knitro.KN_add_obj_constant
        else:
            return knitro.KN_add_con_constants

    def api_add_linear_struct(self, is_obj: bool):
        if is_obj:
            return knitro.KN_add_obj_linear_struct
        else:
            return knitro.KN_add_con_linear_struct

    def api_add_quadratic_struct(self, is_obj: bool):
        if is_obj:
            return knitro.KN_add_obj_quadratic_struct
        else:
            return knitro.KN_add_con_quadratic_struct

    def api_add_callback(self, eval_type: int):
        if eval_type == knitro.KN_RC_EVALH:
            return knitro.KN_set_cb_hess
        elif eval_type == knitro.KN_RC_EVALGA:
            return knitro.KN_set_cb_grad
        elif eval_type == knitro.KN_RC_EVALFC:
            return knitro.KN_add_eval_callback
