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

    def __init__(self):
        self._kc = None
        self.var_map = {}
        self.con_map = {}
        self.obj_nl_expr = None
        self.con_nl_expr_map = {}

    def __del__(self):
        self.close()

    def renew(self):
        self.close()
        self._kc = Package.create_context()

    def close(self):
        if hasattr(self, "_kc") and self._kc is not None:
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
                    lobnds[i] = value(con.lower)
                if con.has_ub():
                    upbnds[i] = value(con.upper)

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
                self.con_nl_expr_map[i] = NonlinearExpressionData(
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
            self.obj_nl_expr = NonlinearExpressionData(
                repn.nonlinear_expr, repn.nonlinear_vars
            )

    def solve(self) -> int:
        self._register_callback()
        return self._execute(knitro.KN_solve)

    def get_status(self) -> int:
        status, _, _, _ = self._execute(knitro.KN_get_solution)
        return status

    def get_num_iters(self) -> int:
        return self._execute(knitro.KN_get_number_iters)

    def get_num_solutions(self) -> int:
        _, _, x, _ = self._execute(knitro.KN_get_solution)
        return 1 if x is not None else 0

    def get_solve_time(self) -> float:
        return self._execute(knitro.KN_get_solve_time_real)

    def get_primals(self, variables: Iterable[VarData]) -> Optional[List[float]]:
        idx_vars = [self.var_map[id(var)] for var in variables]
        return self._execute(knitro.KN_get_var_primal_values, idx_vars)

    def get_duals(self, cons: Iterable[ConstraintData]) -> Optional[List[float]]:
        idx_cons = [self.con_map[id(con)] for con in cons]
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

    def set_outlev(self, level: int = knitro.KN_OUTLEV_ALL):
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
                return -1
            x = req.x
            if obj_eval is not None:
                res.obj = obj_eval(x)
            for i, con_eval in enumerate(con_eval_map.values()):
                res.c[i] = con_eval(x)
            return 0

        return _callback

    def _register_callback(self):
        callback_fn = self._build_callback()
        if callback_fn is not None:
            eval_obj = self.obj_nl_expr is not None
            idx_cons = list(self.con_nl_expr_map.keys())
            self._execute(knitro.KN_add_eval_callback, eval_obj, idx_cons, callback_fn)
