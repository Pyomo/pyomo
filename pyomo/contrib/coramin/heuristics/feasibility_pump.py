import pyomo.environ as pe
from pyomo.core.base.block import _BlockData
from pyomo.contrib.appsi.base import Solver
from pyomo.common.collections import ComponentSet
from pyomo.core.expr.visitor import identify_variables
from pyomo.contrib.coramin.utils.pyomo_utils import get_objective
from typing import List, Sequence, Tuple
from pyomo.core.base.var import _GeneralVarData
import math
import time
from pyomo.common.modeling import unique_component_name
import random
from pyomo.common.dependencies import numpy as np


def collect_integer_vars(
    m: _BlockData,
) -> Tuple[List[_GeneralVarData], List[_GeneralVarData]]:
    binary_vars = ComponentSet()
    integer_vars = ComponentSet()
    for c in m.component_data_objects(
        pe.Constraint, active=True, descend_into=pe.Block
    ):
        for v in identify_variables(c.body, include_fixed=False):
            if v.is_binary():
                binary_vars.add(v)
            elif v.is_integer():
                integer_vars.add(v)
    obj = get_objective(m)
    if obj is not None:
        for v in identify_variables(obj.expr, include_fixed=False):
            if v.is_binary():
                binary_vars.add(v)
            elif v.is_integer():
                integer_vars.add(v)
    return list(binary_vars), list(integer_vars)


def relax_integers(
    binary_vars: Sequence[_GeneralVarData], integer_vars: Sequence[_GeneralVarData]
):
    for v in list(binary_vars) + list(integer_vars):
        lb, ub = v.bounds
        v.domain = pe.Reals
        v.setlb(lb)
        v.setub(ub)


def restore_integers(
    binary_vars: Sequence[_GeneralVarData], integer_vars: Sequence[_GeneralVarData]
):
    for v in binary_vars:
        v.domain = pe.Binary
    for v in integer_vars:
        v.domain = pe.Integers


def check_feasible(
    binary_vars: Sequence[_GeneralVarData],
    integer_vars: Sequence[_GeneralVarData],
    integer_tol=1e-4,
):
    feas = True
    for v in list(binary_vars) + list(integer_vars):
        v_val = v.value
        if not math.isclose(v_val, round(v_val), abs_tol=integer_tol, rel_tol=0):
            print(v_val)
            feas = False
            break
    return feas


def run_feasibility_pump(
    m: _BlockData,
    nlp_solver: Solver,
    time_limit: float = math.inf,
    iter_limit=300,
    integer_tol=1e-4,
    use_fixing: bool = False,
    use_flip: bool = True,
):
    t0 = time.time()

    binary_vars, integer_vars = collect_integer_vars(m)
    relax_integers(binary_vars, integer_vars)

    nlp_solver.config.load_solution = False

    res = nlp_solver.solve(m)
    if res.best_feasible_objective is None:
        restore_integers(binary_vars, integer_vars)
        return None

    res.solution_loader.load_vars(binary_vars)
    res.solution_loader.load_vars(integer_vars)
    is_feas = check_feasible(binary_vars, integer_vars, integer_tol)
    if is_feas:
        restore_integers(binary_vars, integer_vars)
        res.load_vars()
        return res

    orig_obj = get_objective(m)
    orig_obj.deactivate()

    feasible_results = None
    new_obj_name = unique_component_name(m, 'fp_obj')
    last_target_binary_vals = None
    last_target_integer_vals = None
    n_bin = len(binary_vars)
    for _iter in range(iter_limit):
        if time.time() - t0 > time_limit:
            break

        if hasattr(m, new_obj_name):
            delattr(m, new_obj_name)

        target_binary_vals = [round(v.value) for v in binary_vars]
        target_integer_vals = [round(v.value) for v in integer_vars]

        dist_list = list()
        ndx = 0
        for v, val in zip(binary_vars, target_binary_vals):
            dist_list.append((ndx, abs(v.value - val)))
            ndx += 1
        for v, val in zip(integer_vars, target_integer_vals):
            dist_list.append((ndx, abs(v.value - val)))
            ndx += 1
        dist_list.sort(key=lambda i: i[1], reverse=True)

        if use_fixing:
            ndx_to_fix = None
            ndx = len(binary_vars) + len(integer_vars) - 1
            while ndx >= 0:
                tmp = dist_list[ndx][0]
                if tmp < n_bin:
                    if not binary_vars[tmp].is_fixed():
                        ndx_to_fix = tmp
                        break
                else:
                    if not integer_vars[tmp - n_bin].is_fixed():
                        ndx_to_fix = tmp
                        break
                ndx -= 1
            if ndx_to_fix < n_bin:
                binary_vars[ndx_to_fix].fix(target_binary_vals[ndx_to_fix])
            else:
                _ndx = ndx_to_fix - n_bin
                integer_vars[_ndx].fix(target_integer_vals[_ndx])

        if last_target_binary_vals is not None and use_flip:
            if (
                target_binary_vals == last_target_binary_vals
                and target_integer_vals == last_target_integer_vals
            ):
                print('flipping')
                T = math.floor(0.5 * (len(binary_vars) + len(integer_vars)))
                T = 10
                num_flip = random.randint(math.floor(0.5 * T), math.ceil(1.5 * T))
                dist_list = list()
                ndx = 0
                for v, val in zip(binary_vars, target_binary_vals):
                    dist_list.append((ndx, abs(v.value - val)))
                    ndx += 1
                for v, val in zip(integer_vars, target_integer_vals):
                    dist_list.append((ndx, abs(v.value - val)))
                    ndx += 1
                dist_list.sort(key=lambda i: i[1], reverse=True)
                indices_to_flip = [i[0] for i in dist_list[:num_flip]]
                for ndx in indices_to_flip:
                    if ndx < n_bin:
                        if target_binary_vals[ndx] == 0:
                            target_binary_vals[ndx] = 1
                        else:
                            assert target_binary_vals[ndx] == 1
                            target_binary_vals[ndx] = 0
                    else:
                        _ndx = ndx - n_bin
                        if target_integer_vals[_ndx] == 0:
                            target_integer_vals[_ndx] = 1
                        else:
                            assert target_integer_vals[_ndx] == 1
                            target_integer_vals[_ndx] = 0

        last_target_binary_vals = target_binary_vals
        last_target_integer_vals = target_integer_vals

        obj_expr = 0
        for v, val in zip(binary_vars, target_binary_vals):
            if val == 0:
                obj_expr += v
            else:
                assert val == 1
                obj_expr += 1 - v
        for v, val in zip(integer_vars, target_integer_vals):
            obj_expr += (v - val) ** 2
        setattr(m, new_obj_name, pe.Objective(expr=obj_expr))

        res = nlp_solver.solve(m)
        if res.best_feasible_objective is None:
            print('failed')
            break
        res.solution_loader.load_vars([v for v in binary_vars if not v.is_fixed()])
        res.solution_loader.load_vars([v for v in integer_vars if not v.is_fixed()])

        is_feas = check_feasible(binary_vars, integer_vars, integer_tol)
        if is_feas:
            feasible_results = res
            break

    restore_integers(binary_vars, integer_vars)
    orig_obj.activate()
    if hasattr(m, new_obj_name):
        delattr(m, new_obj_name)
    if feasible_results is not None:
        feasible_results.solution_loader.load_vars()
    for v in binary_vars:
        v.unfix()
    for v in integer_vars:
        v.unfix()
    print(orig_obj.expr())

    return feasible_results
