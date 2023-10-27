from .relaxations import iterators
from .relaxations.copy_relaxation import copy_relaxation_with_local_data
import pyomo.environ as pe
from .utils.pyomo_utils import get_objective
from pyomo.repn.standard_repn import generate_standard_repn
from pyomo.common.collections import ComponentSet, ComponentMap


def clone_active_flat(m1):
    m2 = pe.Block(concrete=True)
    m2.cons = pe.ConstraintList()
    all_vars = ComponentSet()

    # constraints
    for c in iterators.nonrelaxation_component_data_objects(
        m1,
        pe.Constraint,
        active=True,
        descend_into=True,
    ):
        lb = pe.value(c.lower)
        ub = pe.value(c.upper)
        repn = generate_standard_repn(c.body, quadratic=False, compute_values=True)
        all_vars.update(repn.linear_vars)
        all_vars.update(repn.nonlinear_vars)
        body = repn.to_expression()
        m2.cons.add((lb, body, ub))

    # objective
    obj = get_objective(m1)
    repn = generate_standard_repn(obj.expr, quadratic=False, compute_values=True)
    all_vars.update(repn.linear_vars)
    all_vars.update(repn.nonlinear_vars)
    obj_expr = repn.to_expression()
    m2.obj = pe.Objective(expr=obj_expr, sense=obj.sense)

    rel_list = list()
    for r in iterators.relaxation_data_objects(
        m1,
        descend_into=True,
        active=True,
    ):
        rel_list.append(r)

    for ndx, r in enumerate(rel_list):
        var_map = ComponentMap()
        for v in r.get_rhs_vars():
            if not v.is_fixed():
                all_vars.add(v)
            var_map[v] = v
        aux_var = r.get_aux_var()
        var_map[aux_var] = aux_var
        if not aux_var.is_fixed():
            all_vars.add(aux_var)
        new_rel = copy_relaxation_with_local_data(r, var_map)
        setattr(m2, f'rel{ndx}', new_rel)

    m2.vars = pe.Reference(list(all_vars))

    return m2
