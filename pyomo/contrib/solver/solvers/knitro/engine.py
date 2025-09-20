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

from collections.abc import Callable, Iterable, MutableMapping
from typing import List, Optional

from pyomo.common.numeric_types import value
from pyomo.core.base.constraint import ConstraintData
from pyomo.core.base.objective import ObjectiveData
from pyomo.core.base.var import VarData
from pyomo.core.plugins.transform.util import partial
from pyomo.repn.standard_repn import StandardRepn, generate_standard_repn

from .api import knitro
from .package import Package
from .utils import NonlinearExpressionData


class Engine:
    """
    A wrapper around the KNITRO API for a single optimization problem.

    This class manages the lifecycle of a KNITRO problem instance (`kc`),
    including building the problem by adding variables and constraints,
    setting options, solving, and freeing the context.
    """

    var_map: MutableMapping[int, int]
    con_map: MutableMapping[int, int]
    obj_nl_expr: Optional[NonlinearExpressionData]
    con_nl_expr_map: MutableMapping[int, NonlinearExpressionData]
    compute_nl_grad: bool

    _status: Optional[int]

    def __init__(self, *, compute_nl_grad: bool = True):
        self.var_map = {}
        self.con_map = {}
        self.obj_nl_expr = None
        self.con_nl_expr_map = {}
        self.compute_nl_grad = compute_nl_grad
        self._kc = None
        self._status = None

    def __del__(self):
        self.close()

    def renew(self):
        self.close()
        self._kc = Package.create_context()

    def close(self):
        if self._kc is not None:
            self._execute(knitro.KN_free)
            self._kc = None

    def add_vars(self, variables: Iterable[VarData]):
        n_vars = len(variables)
        idx_vars = self._execute(knitro.KN_add_vars, n_vars)
        if idx_vars is None:
            return

        for var, idx in zip(variables, idx_vars, strict=True):
            self.var_map[id(var)] = idx

        var_types, fxbnds, lobnds, upbnds = {}, {}, {}, {}
        for var in variables:
            idx = self.var_map[id(var)]
            if var.is_binary():
                var_types[idx] = knitro.KN_VARTYPE_BINARY
            elif var.is_integer():
                var_types[idx] = knitro.KN_VARTYPE_INTEGER
            elif not var.is_continuous():
                msg = f"Unknown variable type for variable {var.name}."
                raise ValueError(msg)

            if var.fixed:
                fxbnds[idx] = value(var.value)
            else:
                if var.has_lb():
                    lobnds[idx] = value(var.lb)
                if var.has_ub():
                    upbnds[idx] = value(var.ub)

        self._execute(knitro.KN_set_var_types, var_types.keys(), var_types.values())
        self._execute(knitro.KN_set_var_fxbnds, fxbnds.keys(), fxbnds.values())
        self._execute(knitro.KN_set_var_lobnds, lobnds.keys(), lobnds.values())
        self._execute(knitro.KN_set_var_upbnds, upbnds.keys(), upbnds.values())

    def add_cons(self, cons: Iterable[ConstraintData]):
        n_cons = len(cons)
        idx_cons = self._execute(knitro.KN_add_cons, n_cons)
        if idx_cons is None:
            return

        for con, idx in zip(cons, idx_cons, strict=True):
            self.con_map[id(con)] = idx

        eqbnds, lobnds, upbnds = {}, {}, {}
        for con in cons:
            idx = self.con_map[id(con)]
            if con.equality:
                eqbnds[idx] = value(con.lower)
            else:
                if con.has_lb():
                    lobnds[idx] = value(con.lower)
                if con.has_ub():
                    upbnds[idx] = value(con.upper)

        self._execute(knitro.KN_set_con_eqbnds, eqbnds.keys(), eqbnds.values())
        self._execute(knitro.KN_set_con_lobnds, lobnds.keys(), lobnds.values())
        self._execute(knitro.KN_set_con_upbnds, upbnds.keys(), upbnds.values())

        for con in cons:
            idx = self.con_map[id(con)]
            repn = generate_standard_repn(con.body)
            self._add_expr_structs_using_repn(
                repn,
                add_const_fn=partial(self._execute, knitro.KN_add_con_constants, idx),
                add_lin_fn=partial(self._execute, knitro.KN_add_con_linear_struct, idx),
                add_quad_fn=partial(
                    self._execute, knitro.KN_add_con_quadratic_struct, idx
                ),
            )
            if repn.nonlinear_expr is not None:
                self.con_nl_expr_map[idx] = NonlinearExpressionData(
                    repn.nonlinear_expr,
                    repn.nonlinear_vars,
                    compute_grad=self.compute_nl_grad,
                )

    def set_obj(self, obj: ObjectiveData):
        obj_goal = (
            knitro.KN_OBJGOAL_MINIMIZE
            if obj.is_minimizing()
            else knitro.KN_OBJGOAL_MAXIMIZE
        )
        self._execute(knitro.KN_set_obj_goal, obj_goal)
        repn = generate_standard_repn(obj.expr)
        self._add_expr_structs_using_repn(
            repn,
            add_const_fn=partial(self._execute, knitro.KN_add_obj_constant),
            add_lin_fn=partial(self._execute, knitro.KN_add_obj_linear_struct),
            add_quad_fn=partial(self._execute, knitro.KN_add_obj_quadratic_struct),
        )
        if repn.nonlinear_expr is not None:
            self.obj_nl_expr = NonlinearExpressionData(
                repn.nonlinear_expr,
                repn.nonlinear_vars,
                compute_grad=self.compute_nl_grad,
            )

    def solve(self) -> int:
        self._register_callback()
        # TODO: remove this when the tolerance test is fixed in test_solvers
        self._execute(knitro.KN_set_double_param, knitro.KN_PARAM_FTOL, 1e-10)
        self._execute(knitro.KN_set_double_param, knitro.KN_PARAM_OPTTOL, 1e-10)
        self._execute(knitro.KN_set_double_param, knitro.KN_PARAM_XTOL, 1e-10)
        self._status = self._execute(knitro.KN_solve)
        return self._status

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

    def get_var_idxs(self, variables: Iterable[VarData]) -> List[int]:
        return [self.var_map[id(var)] for var in variables]

    def get_con_idxs(self, constraints: Iterable[ConstraintData]) -> List[int]:
        return [self.con_map[id(con)] for con in constraints]

    def get_var_primals(self, variables: Iterable[VarData]) -> Optional[List[float]]:
        idx_vars = self.get_var_idxs(variables)
        return self._execute(knitro.KN_get_var_primal_values, idx_vars)

    def get_var_duals(self, variables: Iterable[VarData]) -> Optional[List[float]]:
        idx_vars = self.get_var_idxs(variables)
        return self._execute(knitro.KN_get_var_dual_values, idx_vars)

    def get_con_duals(self, cons: Iterable[ConstraintData]) -> Optional[List[float]]:
        idx_cons = self.get_con_idxs(cons)
        return self._execute(knitro.KN_get_con_dual_values, idx_cons)

    def get_obj_value(self) -> Optional[float]:
        return self._execute(knitro.KN_get_obj_value)

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

    def set_outlev(self, level: Optional[int] = None):
        if level is None:
            level = knitro.KN_OUTLEV_ALL
        self.set_options(outlev=level)

    def set_time_limit(self, time_limit: float):
        self.set_options(maxtime_cpu=time_limit)

    def set_num_threads(self, nthreads: int):
        self.set_options(threads=nthreads)

    # ----------------- Private methods -------------------------

    def _execute(self, api_fn, *args, **kwargs):
        if self._kc is None:
            msg = "KNITRO context has been freed or has not been initialized and cannot be used."
            raise RuntimeError(msg)
        return api_fn(self._kc, *args, **kwargs)

    def _add_expr_structs_using_repn(
        self,
        repn: StandardRepn,
        add_const_fn: Callable[[float], None],
        add_lin_fn: Callable[[Iterable[int], Iterable[float]], None],
        add_quad_fn: Callable[[Iterable[int], Iterable[int], Iterable[float]], None],
    ):
        if repn.constant is not None:
            add_const_fn(repn.constant)
        if repn.linear_vars:
            idx_lin_vars = self.get_var_idxs(repn.linear_vars)
            add_lin_fn(idx_lin_vars, list(repn.linear_coefs))
        if repn.quadratic_vars:
            quad_vars1, quad_vars2 = zip(*repn.quadratic_vars)
            idx_quad_vars1 = self.get_var_idxs(quad_vars1)
            idx_quad_vars2 = self.get_var_idxs(quad_vars2)
            add_quad_fn(idx_quad_vars1, idx_quad_vars2, list(repn.quadratic_coefs))

    def _build_callback_eval(self):
        obj_eval = (
            self.obj_nl_expr.create_evaluator(self.var_map)
            if self.obj_nl_expr is not None
            else None
        )

        con_eval_map = {
            i: nl_expr.create_evaluator(self.var_map)
            for i, nl_expr in self.con_nl_expr_map.items()
        }

        if obj_eval is None and not con_eval_map:
            return None

        def _callback_eval(_, cb, req, res, data=None):
            if req.type != knitro.KN_RC_EVALFC:
                return -1
            x = req.x
            if obj_eval is not None:
                res.obj = obj_eval(x)
            for i, con_eval in enumerate(con_eval_map.values()):
                res.c[i] = con_eval(x)
            return 0

        return _callback_eval

    def _build_callback_grad(self):
        if not self.compute_nl_grad:
            return None

        obj_grad = (
            self.obj_nl_expr.create_gradient_evaluator(self.var_map)
            if self.obj_nl_expr is not None and self.obj_nl_expr.grad is not None
            else None
        )
        con_grad_map = {
            i: expr.create_gradient_evaluator(self.var_map)
            for i, expr in self.con_nl_expr_map.items()
            if expr.grad is not None
        }

        if obj_grad is None and not con_grad_map:
            return None

        def _callback_grad(_, cb, req, res, data=None):
            if req.type != knitro.KN_RC_EVALGA:
                return -1
            x = req.x
            if obj_grad is not None:
                obj_g = obj_grad(x)
                for j, g in enumerate(obj_g):
                    res.objGrad[j] = g
            k = 0
            for con_grad in con_grad_map.values():
                con_g = con_grad(x)
                for g in con_g:
                    res.jac[k] = g
                    k += 1
            return 0

        return _callback_grad

    def _build_callback(self):
        callback_eval = self._build_callback_eval()
        callback_grad = self._build_callback_grad()
        return callback_eval, callback_grad

    def _register_callback(self):
        f, grad = self._build_callback()
        if f is not None:
            eval_obj = self.obj_nl_expr is not None
            idx_cons = list(self.con_nl_expr_map.keys())
            cb = self._execute(knitro.KN_add_eval_callback, eval_obj, idx_cons, f)
            if grad is not None:
                obj_var_idxs = (
                    self.get_var_idxs(self.obj_nl_expr.variables)
                    if self.obj_nl_expr is not None
                    else None
                )
                jac_idx_cons, jac_idx_vars = [], []
                for i, con_nl_expr in self.con_nl_expr_map.items():
                    idx_vars = self.get_var_idxs(con_nl_expr.variables)
                    n_vars = len(idx_vars)
                    jac_idx_cons.extend([i] * n_vars)
                    jac_idx_vars.extend(idx_vars)
                self._execute(
                    knitro.KN_set_cb_grad,
                    cb,
                    obj_var_idxs,
                    jac_idx_cons,
                    jac_idx_vars,
                    grad,
                )
