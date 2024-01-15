from .relaxations import iterators
from .relaxations.copy_relaxation import copy_relaxation_with_local_data
import pyomo.environ as pe
from .utils.pyomo_utils import get_objective
from pyomo.repn.standard_repn import generate_standard_repn
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.base.block import _BlockData
from typing import List


def clone_active_flat(m1: _BlockData, num_clones: int = 1) -> List[_BlockData]:
    clone_list = [pe.Block(concrete=True) for i in range(num_clones)]
    for m2 in clone_list:
        m2.linear = pe.Block()
        m2.nonlinear = pe.Block()
        m2.linear.cons = pe.ConstraintList()
        m2.nonlinear.cons = pe.ConstraintList()
    all_vars = ComponentSet()

    # constraints
    for c in iterators.nonrelaxation_component_data_objects(
        m1, pe.Constraint, active=True, descend_into=True
    ):
        repn = generate_standard_repn(c.body, quadratic=False, compute_values=True)
        all_vars.update(repn.linear_vars)
        all_vars.update(repn.nonlinear_vars)
        body = repn.to_expression()
        if repn.nonlinear_expr is None:
            for m2 in clone_list:
                m2.linear.cons.add((c.lb, body, c.ub))
        else:
            for m2 in clone_list:
                m2.nonlinear.cons.add((c.lb, body, c.ub))

    # objective
    obj = get_objective(m1)
    if obj is not None:
        repn = generate_standard_repn(obj.expr, quadratic=False, compute_values=True)
        all_vars.update(repn.linear_vars)
        all_vars.update(repn.nonlinear_vars)
        obj_expr = repn.to_expression()
        if repn.nonlinear_expr is None:
            for m2 in clone_list:
                m2.linear.obj = pe.Objective(expr=obj_expr, sense=obj.sense)
        else:
            for m2 in clone_list:
                m2.nonlinear.obj = pe.Objective(expr=obj_expr, sense=obj.sense)

    rel_list = list()
    for r in iterators.relaxation_data_objects(m1, descend_into=True, active=True):
        rel_list.append(r)

    for ndx, r in enumerate(rel_list):
        var_map = ComponentMap()
        for v in r.get_rhs_vars():
            if not v.is_fixed():
                all_vars.add(v)
            var_map[v] = v
        aux_var = r.get_aux_var()
        var_map[aux_var] = aux_var
        if not pe.is_fixed(aux_var):
            all_vars.add(aux_var)
        new_rel = copy_relaxation_with_local_data(r, var_map)
        for m2 in clone_list:
            setattr(m2, f'rel{ndx}', new_rel)

    for m2 in clone_list:
        m2.vars = list(all_vars)

    return clone_list
