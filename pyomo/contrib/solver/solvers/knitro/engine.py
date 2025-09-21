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

from collections.abc import Iterable, Mapping, MutableMapping
from typing import List, Optional, Union

from pyomo.common.enums import ObjectiveSense
from pyomo.common.numeric_types import value
from pyomo.core.base.constraint import ConstraintData
from pyomo.core.base.objective import ObjectiveData
from pyomo.core.base.var import VarData
from pyomo.repn.standard_repn import generate_standard_repn

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

    mapping: Mapping[bool, MutableMapping[int, int]]
    nonlinear_map: MutableMapping[Optional[int], NonlinearExpressionData]
    compute_nl_grad: bool
    compute_nl_hess: bool

    _status: Optional[int]

    def __init__(
        self, *, compute_nl_grad: bool = True, compute_nl_hessian: bool = True
    ):
        # True: variables, False: constraints
        self.mapping = {True: {}, False: {}}
        self.nonlinear_map = {}
        self.compute_nl_grad = compute_nl_grad
        self.compute_nl_hess = compute_nl_hessian

        self._kc = None
        self._status = None

        self.param_setters = {
            knitro.KN_PARAMTYPE_INTEGER: knitro.KN_set_int_param,
            knitro.KN_PARAMTYPE_FLOAT: knitro.KN_set_double_param,
            knitro.KN_PARAMTYPE_STRING: knitro.KN_set_char_param,
        }

    def __del__(self):
        self.close()

    def renew(self):
        self.close()
        self._kc = Package.create_context()
        # TODO: remove this when the tolerance test is fixed in test_solvers
        self._execute(knitro.KN_set_double_param, knitro.KN_PARAM_FTOL, 1e-8)
        self._execute(knitro.KN_set_double_param, knitro.KN_PARAM_OPTTOL, 1e-8)
        self._execute(knitro.KN_set_double_param, knitro.KN_PARAM_XTOL, 1e-8)

    def close(self):
        if self._kc is not None:
            self._execute(knitro.KN_free)
            self._kc = None

    def add_vars(self, variables: Iterable[VarData]):
        self._add_comps(variables, is_var=True)
        self._set_var_types(variables)
        self._set_bnds(variables, is_var=True)

    def add_cons(self, cons: Iterable[ConstraintData]):
        self._add_comps(cons, is_var=False)
        self._set_bnds(cons, is_var=False)

        for con in cons:
            i = self.mapping[False][id(con)]
            self._add_structs(i, con.body)

    def set_obj(self, obj: ObjectiveData):
        self._set_obj_goal(obj.sense)
        self._add_structs(None, obj.expr)

    def solve(self) -> int:
        self._register_callbacks()
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
        return [self.mapping[True][id(var)] for var in variables]

    def get_con_idxs(self, constraints: Iterable[ConstraintData]) -> List[int]:
        return [self.mapping[False][id(con)] for con in constraints]

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
            param_setter = self.param_setters[param_type]
            self._execute(param_setter, param_id, val)

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

    def _add_comps(
        self, comps: Union[Iterable[VarData], Iterable[ConstraintData]], *, is_var: bool
    ):
        if is_var:
            adder = knitro.KN_add_vars
        else:
            adder = knitro.KN_add_cons
        idxs = self._execute(adder, len(comps))

        if idxs is not None:
            self.mapping[is_var].update(zip(map(id, comps), idxs))

    def _parse_var_type(
        self, var: VarData, i: int, var_types: MutableMapping[int, int]
    ) -> int:
        if var.is_binary():
            var_types[i] = knitro.KN_VARTYPE_BINARY
        elif var.is_integer():
            var_types[i] = knitro.KN_VARTYPE_INTEGER
        elif not var.is_continuous():
            msg = f"Unknown variable type for variable {var.name}."
            raise ValueError(msg)

    def _set_var_types(self, variables: Iterable[VarData]):
        var_types = {}
        for var in variables:
            i = self.mapping[True][id(var)]
            self._parse_var_type(var, i, var_types)
        self._execute(knitro.KN_set_var_types, var_types.keys(), var_types.values())

    def _parse_var_bnds(
        self,
        var: VarData,
        i: int,
        eqbnds: MutableMapping[int, float],
        lobnds: MutableMapping[int, float],
        upbnds: MutableMapping[int, float],
    ):
        if var.fixed:
            eqbnds[i] = value(var.value)
        else:
            if var.has_lb():
                lobnds[i] = value(var.lb)
            if var.has_ub():
                upbnds[i] = value(var.ub)

    def _parse_con_bnds(
        self,
        con: ConstraintData,
        i: int,
        eqbnds: MutableMapping[int, float],
        lobnds: MutableMapping[int, float],
        upbnds: MutableMapping[int, float],
    ):
        if con.equality:
            eqbnds[i] = value(con.lower)
        else:
            if con.has_lb():
                lobnds[i] = value(con.lower)
            if con.has_ub():
                upbnds[i] = value(con.upper)

    def _set_bnds(
        self, comps: Union[Iterable[VarData], Iterable[ConstraintData]], *, is_var: bool
    ):
        if is_var:
            parser = self._parse_var_bnds
        else:
            parser = self._parse_con_bnds

        eqbnds, lobnds, upbnds = {}, {}, {}
        for comp in comps:
            i = self.mapping[is_var][id(comp)]
            parser(comp, i, eqbnds, lobnds, upbnds)

        if is_var:
            setters = [
                knitro.KN_set_var_fxbnds,
                knitro.KN_set_var_lobnds,
                knitro.KN_set_var_upbnds,
            ]
        else:
            setters = [
                knitro.KN_set_con_eqbnds,
                knitro.KN_set_con_lobnds,
                knitro.KN_set_con_upbnds,
            ]
        for bnds, setter in zip([eqbnds, lobnds, upbnds], setters):
            self._execute(setter, bnds.keys(), bnds.values())

    def _add_structs(self, i: Optional[int], expr):
        is_obj = i is None
        repn = generate_standard_repn(expr)

        if is_obj:
            add_constant = knitro.KN_add_obj_constant
            add_linear = knitro.KN_add_obj_linear_struct
            add_quadratic = knitro.KN_add_obj_quadratic_struct
        else:
            add_constant = knitro.KN_add_con_constants
            add_linear = knitro.KN_add_con_linear_struct
            add_quadratic = knitro.KN_add_con_quadratic_struct

        base_args = [] if is_obj else [i]
        if repn.constant is not None:
            self._execute(add_constant, *base_args, repn.constant)
        if repn.linear_vars:
            idx_lin_vars = self.get_var_idxs(repn.linear_vars)
            lin_coefs = list(repn.linear_coefs)
            self._execute(add_linear, *base_args, idx_lin_vars, lin_coefs)
        if repn.quadratic_vars:
            quad_vars1, quad_vars2 = zip(*repn.quadratic_vars)
            idx_quad_vars1 = self.get_var_idxs(quad_vars1)
            idx_quad_vars2 = self.get_var_idxs(quad_vars2)
            quad_coefs = list(repn.quadratic_coefs)
            self._execute(
                add_quadratic, *base_args, idx_quad_vars1, idx_quad_vars2, quad_coefs
            )

        if repn.nonlinear_expr is not None:
            self.nonlinear_map[i] = NonlinearExpressionData(
                repn.nonlinear_expr,
                repn.nonlinear_vars,
                compute_grad=self.compute_nl_grad,
                compute_hess=self.compute_nl_hess,
            )

    def _set_obj_goal(self, sense: ObjectiveSense):
        obj_goal = (
            knitro.KN_OBJGOAL_MINIMIZE
            if sense == ObjectiveSense.minimize
            else knitro.KN_OBJGOAL_MAXIMIZE
        )
        self._execute(knitro.KN_set_obj_goal, obj_goal)

    def _build_callback(
        self,
        i: Optional[int],
        expr: NonlinearExpressionData,
        callback_type: int,
    ):
        is_obj = i is None
        vmap = self.mapping[True]
        if callback_type == knitro.KN_RC_EVALFC:
            evaluator = expr.create_evaluator(vmap)

            def _eval(_, cb, req, res, data=None):
                if req.type != knitro.KN_RC_EVALFC:
                    return -1
                if is_obj:
                    res.obj = evaluator(req.x)
                else:
                    res.c[0] = evaluator(req.x)
                return 0

            return _eval
        elif callback_type == knitro.KN_RC_EVALGA:
            grad = expr.create_gradient_evaluator(vmap)

            def _grad(_, cb, req, res, data=None):
                if req.type != knitro.KN_RC_EVALGA:
                    return -1
                if is_obj:
                    res.objGrad[:] = grad(req.x)
                else:
                    res.jac[:] = grad(req.x)
                return 0

            return _grad
        elif callback_type == knitro.KN_RC_EVALH:
            hess = expr.create_hessian_evaluator(vmap)

            def _hess(_, cb, req, res, data=None):
                if req.type != knitro.KN_RC_EVALH:
                    return -1
                mu = req.sigma if is_obj else req.lambda_[i]
                res.hess[:] = hess(req.x, mu)
                return 0

            return _hess

    def _add_eval_callback(self, i: Optional[int], expr: NonlinearExpressionData):
        func_callback = self._build_callback(i, expr, knitro.KN_RC_EVALFC)
        eval_obj = i is None
        idx_cons = [i] if not eval_obj else None
        return self._execute(
            knitro.KN_add_eval_callback, eval_obj, idx_cons, func_callback
        )

    def _add_grad_callback(
        self, i: Optional[int], expr: NonlinearExpressionData, callback
    ):
        idx_vars = self.get_var_idxs(expr.grad_vars)
        is_obj = i is None
        obj_grad_idx_vars = idx_vars if is_obj else None
        jac_idx_cons = [i] * len(idx_vars) if not is_obj else None
        jac_idx_vars = idx_vars if not is_obj else None
        grad_callback = self._build_callback(i, expr, knitro.KN_RC_EVALGA)
        self._execute(
            knitro.KN_set_cb_grad,
            callback,
            obj_grad_idx_vars,
            jac_idx_cons,
            jac_idx_vars,
            grad_callback,
        )

    def _add_hess_callback(
        self, i: Optional[int], expr: NonlinearExpressionData, callback
    ):
        hess_vars1, hess_vars2 = zip(*expr.hess_vars)
        hess_idx_vars1 = self.get_var_idxs(hess_vars1)
        hess_idx_vars2 = self.get_var_idxs(hess_vars2)
        hess_callback = self._build_callback(i, expr, knitro.KN_RC_EVALH)
        self._execute(
            knitro.KN_set_cb_hess,
            callback,
            hess_idx_vars1,
            hess_idx_vars2,
            hess_callback,
        )

    def _register_callback(self, i: Optional[int], expr: NonlinearExpressionData):
        callback = self._add_eval_callback(i, expr)
        if expr.grad is not None:
            self._add_grad_callback(i, expr, callback)
        if expr.hessian is not None:
            self._add_hess_callback(i, expr, callback)

    def _register_callbacks(self):
        for i, expr in self.nonlinear_map.items():
            self._register_callback(i, expr)
