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
        # None: objective
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
        # TODO: remove this when the tolerance tests are fixed in test_solvers
        tol = 1e-8
        self._execute(knitro.KN_set_double_param, knitro.KN_PARAM_FTOL, tol)
        self._execute(knitro.KN_set_double_param, knitro.KN_PARAM_OPTTOL, tol)
        self._execute(knitro.KN_set_double_param, knitro.KN_PARAM_XTOL, tol)

    def close(self):
        if self._kc is not None:
            self._execute(knitro.KN_free)
            self._kc = None

    def add_vars(self, variables: Iterable[VarData]):
        self._add_comps(variables, is_var=True)
        self._set_var_types(variables)
        self._set_comp_bnds(variables, is_var=True)

    def add_cons(self, cons: Iterable[ConstraintData]):
        self._add_comps(cons, is_var=False)
        self._set_comp_bnds(cons, is_var=False)

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
        return self._get_comp_idxs(variables, is_var=True)

    def get_con_idxs(self, constraints: Iterable[ConstraintData]) -> List[int]:
        return self._get_comp_idxs(constraints, is_var=False)

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

    def _get_comp_idxs(self, comps: Union[Iterable[VarData], Iterable[ConstraintData]], *, is_var: bool):
        return [self.mapping[is_var][id(comp)] for comp in comps]

    def _add_comps(
        self, comps: Union[Iterable[VarData], Iterable[ConstraintData]], *, is_var: bool
    ):
        adder = knitro.KN_add_vars if is_var else knitro.KN_add_cons
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

    def _get_bnd_setters(self, is_var: bool):
        if is_var:
            return [
                knitro.KN_set_var_fxbnds,
                knitro.KN_set_var_lobnds,
                knitro.KN_set_var_upbnds,
            ]
        else:
            return [
                knitro.KN_set_con_eqbnds,
                knitro.KN_set_con_lobnds,
                knitro.KN_set_con_upbnds,
            ]

    def _set_comp_bnds(
        self, comps: Union[Iterable[VarData], Iterable[ConstraintData]], *, is_var: bool
    ):
        parser = self._parse_var_bnds if is_var else self._parse_con_bnds
        bnds_data = [{}, {}, {}]  # eqbnds, lobnds, upbnds
        for comp in comps:
            i = self.mapping[is_var][id(comp)]
            parser(comp, i, *bnds_data)
        setters = self._get_bnd_setters(is_var)
        for bnds, setter in zip(bnds_data, setters):
            self._execute(setter, bnds.keys(), bnds.values())

    def _get_struct_api_funcs(self, i: Optional[int]):
        if i is None:
            add_constant = knitro.KN_add_obj_constant
            add_linear = knitro.KN_add_obj_linear_struct
            add_quadratic = knitro.KN_add_obj_quadratic_struct
        else:
            add_constant = knitro.KN_add_con_constants
            add_linear = knitro.KN_add_con_linear_struct
            add_quadratic = knitro.KN_add_con_quadratic_struct
        return add_constant, add_linear, add_quadratic

    def _add_structs(self, i: Optional[int], expr):
        repn = generate_standard_repn(expr)

        add_constant, add_linear, add_quadratic = self._get_struct_api_funcs(i)
        base_args = [] if i is None else [i]

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

    def _create_evaluator(self, expr: NonlinearExpressionData, eval_type: int):
        vmap = self.mapping[True]
        if eval_type == knitro.KN_RC_EVALFC:
            return expr.create_evaluator(vmap)
        elif eval_type == knitro.KN_RC_EVALGA:
            return expr.create_gradient_evaluator(vmap)
        else:
            return expr.create_hessian_evaluator(vmap)

    def _build_callback(
        self, i: Optional[int], expr: NonlinearExpressionData, eval_type: int
    ):
        is_obj = i is None
        func = self._create_evaluator(expr, eval_type)

        if is_obj:
            if eval_type == knitro.KN_RC_EVALFC:

                def _callback(kc, cb, req, res, _):
                    res.obj = func(req.x)
                    return 0

            elif eval_type == knitro.KN_RC_EVALGA:

                def _callback(kc, cb, req, res, _):
                    res.objGrad[:] = func(req.x)
                    return 0

            else:

                def _callback(kc, cb, req, res, _):
                    res.hess[:] = func(req.x, req.sigma)
                    return 0

        else:
            if eval_type == knitro.KN_RC_EVALFC:

                def _callback(kc, cb, req, res, _):
                    res.c[0] = func(req.x)
                    return 0

            elif eval_type == knitro.KN_RC_EVALGA:

                def _callback(kc, cb, req, res, _):
                    res.jac[:] = func(req.x)
                    return 0

            else:

                def _callback(kc, cb, req, res, _):
                    res.hess[:] = func(req.x, req.lambda_[i])
                    return 0

        return _callback

    def _add_callback(
        self,
        i: Optional[int],
        expr: NonlinearExpressionData,
        eval_type: int,
        callback=None,
    ):
        func_callback = self._build_callback(i, expr, eval_type)
        if eval_type == knitro.KN_RC_EVALFC:
            eval_obj = i is None
            idx_cons = [i] if not eval_obj else None
            return self._execute(
                knitro.KN_add_eval_callback, eval_obj, idx_cons, func_callback
            )
        elif eval_type == knitro.KN_RC_EVALGA:
            idx_vars = self.get_var_idxs(expr.grad_vars)
            is_obj = i is None
            obj_grad_idx_vars = idx_vars if is_obj else None
            jac_idx_cons = [i] * len(idx_vars) if not is_obj else None
            jac_idx_vars = idx_vars if not is_obj else None
            return self._execute(
                knitro.KN_set_cb_grad,
                callback,
                obj_grad_idx_vars,
                jac_idx_cons,
                jac_idx_vars,
                func_callback,
            )
        else:
            hess_vars1, hess_vars2 = zip(*expr.hess_vars)
            hess_idx_vars1 = self.get_var_idxs(hess_vars1)
            hess_idx_vars2 = self.get_var_idxs(hess_vars2)
            return self._execute(
                knitro.KN_set_cb_hess,
                callback,
                hess_idx_vars1,
                hess_idx_vars2,
                func_callback,
            )

    def _register_callback(self, i: Optional[int], expr: NonlinearExpressionData):
        callback = self._add_callback(i, expr, knitro.KN_RC_EVALFC)
        if expr.grad is not None:
            self._add_callback(i, expr, knitro.KN_RC_EVALGA, callback)
        if expr.hessian is not None:
            self._add_callback(i, expr, knitro.KN_RC_EVALH, callback)

    def _register_callbacks(self):
        for i, expr in self.nonlinear_map.items():
            self._register_callback(i, expr)
