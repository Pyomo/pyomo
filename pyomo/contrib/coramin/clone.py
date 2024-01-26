from .relaxations import iterators
from .relaxations.copy_relaxation import copy_relaxation_with_local_data
import pyomo.environ as pe
from .utils.pyomo_utils import get_objective
from pyomo.repn.standard_repn import generate_standard_repn
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.base.block import _BlockData
from typing import List
from pyomo.core.expr.visitor import identify_variables
from pyomo.common.modeling import unique_component_name


def get_clone_and_var_map(m1: _BlockData):
    orig_vars = list()
    for c in iterators.nonrelaxation_component_data_objects(
        m1, pe.Constraint, active=True, descend_into=True
    ):
        for v in identify_variables(c.body, include_fixed=False):
            orig_vars.append(v)
    obj = get_objective(m1)
    if obj is not None:
        for v in identify_variables(obj.expr, include_fixed=False):
            orig_vars.append(v)
    for r in iterators.relaxation_data_objects(m1, descend_into=True, active=True):
        orig_vars.extend(r.get_rhs_vars())
        orig_vars.append(r.get_aux_var())
    orig_vars = list(ComponentSet(orig_vars))
    tmp_name = unique_component_name(m1, "active_vars")
    setattr(m1, tmp_name, orig_vars)
    m2 = m1.clone()
    new_vars = getattr(m2, tmp_name)
    var_map = ComponentMap(zip(new_vars, orig_vars))
    delattr(m1, tmp_name)
    delattr(m2, tmp_name)
    return m2, var_map


def clone_shallow_active_flat(m1: _BlockData, num_clones: int = 1) -> List[_BlockData]:
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
        var_map = dict()
        for v in r.get_rhs_vars():
            if not v.is_fixed():
                all_vars.add(v)
            var_map[id(v)] = v
        aux_var = r.get_aux_var()
        var_map[id(aux_var)] = aux_var
        if not pe.is_fixed(aux_var):
            all_vars.add(aux_var)
        new_rel = copy_relaxation_with_local_data(r, var_map)
        for m2 in clone_list:
            setattr(m2, f'rel{ndx}', new_rel)

    for m2 in clone_list:
        m2.vars = list(all_vars)

    return clone_list
