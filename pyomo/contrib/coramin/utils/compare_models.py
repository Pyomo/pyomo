import pyomo.environ as pe
from pyomo.contrib.coramin.clone import clone_shallow_active_flat
from pyomo.core.base.block import _BlockData
from typing import Optional
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
from pyomo.contrib import appsi
from .pyomo_utils import active_vars, active_cons, simplify_expr
from pyomo.common.collections import ComponentSet, ComponentMap, OrderedSet
from pyomo.core.expr.visitor import identify_variables, replace_expressions
from pyomo.repn.standard_repn import StandardRepn, generate_standard_repn
from pyomo.common.modeling import unique_component_name


def _attempt_presolve(m, vars_to_presolve):
    vars_to_presolve = ComponentSet(vars_to_presolve)
    var_to_con_map = ComponentMap()
    for v in vars_to_presolve:
        var_to_con_map[v] = OrderedSet()
    for c in active_cons(m):
        for v in identify_variables(c.body, include_fixed=False):
            if v in vars_to_presolve:
                var_to_con_map[v].add(c)
    cname = unique_component_name(m, 'bound_constraints')
    bound_cons = pe.ConstraintList()
    setattr(m, cname, bound_cons)
    for v in list(var_to_con_map.keys()):
        con_list = var_to_con_map[v]
        v_expr = None
        v_con = None
        v_repn = None
        v_vars = None
        density = None
        for c in con_list:
            if not c.equality:
                continue
            if not c.active:
                continue
            repn: StandardRepn = generate_standard_repn(
                c.body - c.lb, compute_values=True, quadratic=False
            )
            lin_vars = ComponentSet(repn.linear_vars)
            nonlin_vars = ComponentSet(repn.nonlinear_vars)
            if v in lin_vars and v not in nonlin_vars:
                n_vars = len(
                    ComponentSet(list(repn.linear_vars) + list(repn.nonlinear_vars))
                )
                if density is None or n_vars < density:
                    v_expr = -repn.constant
                    for coef, other in zip(repn.linear_coefs, repn.linear_vars):
                        if v is other:
                            v_coef = coef
                        else:
                            v_expr -= coef * other
                    if repn.nonlinear_expr is not None:
                        v_expr -= repn.nonlinear_expr
                    v_expr /= v_coef
                    v_con = c
                    v_repn = repn
                    v_vars = ComponentSet(
                        [i for i in v_repn.linear_vars if i in var_to_con_map]
                    )
                    v_vars.update(
                        [i for i in v_repn.nonlinear_vars if i in var_to_con_map]
                    )
                    v_vars.remove(v)
                    density = n_vars
        if v_expr is None:
            return False

        v_con.deactivate()

        if v.lb is not None or v.ub is not None:
            new_con = bound_cons.add((v.lb, v_expr, v.ub))
            for _v in v_vars:
                var_to_con_map[_v].add(new_con)

        for c in con_list:
            if c is v_con:
                continue
            sub_map = {id(v): v_expr}
            new_body = simplify_expr(
                replace_expressions(c.body, substitution_map=sub_map)
            )
            c.set_value((c.lb, new_body, c.ub))
            for _v in v_vars:
                var_to_con_map[_v].add(c)

    return True


def is_relaxation(
    a: _BlockData,
    b: _BlockData,
    opt: appsi.base.Solver,
    feasibility_tol: float = 1e-6,
    bigM: Optional[float] = None,
):
    """
    Returns True if every feasible point in b is feasible for a
    (a is a relaxation of b)
    a and b should share variables
    """

    """
    if a has variables that b does not, this will not work
    see if we can presolve them out
    Note - it is okay if b has variables that a does not
    """
    a_vars = ComponentSet(active_vars(a))
    b_vars = ComponentSet(active_vars(b))
    vars_to_presolve = a_vars - b_vars
    if len(vars_to_presolve) > 0:
        a = clone_shallow_active_flat(a)[0]
        if not _attempt_presolve(a, vars_to_presolve):
            raise RuntimeError(
                'a has variables that b does not, which makes the following analysis invalid'
            )

    m = clone_shallow_active_flat(b)[0]
    if hasattr(m.linear, 'obj'):
        del m.linear.obj
    if hasattr(m.nonlinear, 'obj'):
        del m.nonlinear.obj

    m.max_viol = pe.Var(bounds=(None, 1))
    m.con_viol = pe.VarList()
    m.is_max = pe.VarList(domain=pe.Binary)
    m.max_viol_cons = pe.ConstraintList()
    u_y_pairs = list()
    default_M = bigM
    if default_M is None:
        bigM = 0
    else:
        bigM = default_M
    for con in a.component_data_objects(pe.Constraint, descend_into=True, active=True):
        elist = list()
        if con.ub is not None:
            elist.append(con.body - con.ub)
        if con.lb is not None:
            elist.append(-con.body + con.lb)
        for e in elist:
            u = m.con_viol.add()
            y = m.is_max.add()
            m.max_viol_cons.add(u <= e)
            u_y_pairs.append((u, y))
            if default_M is None:
                u_lb = compute_bounds_on_expr(e)[0]
                if u_lb is None:
                    raise RuntimeError('could not compute big M value')
                if u_lb > feasibility_tol:
                    return False
                if u_lb < 0:
                    bigM = max(bigM, abs(u_lb))
    m.max_viol_cons.add(sum(m.is_max.values()) == 1)
    for u, y in u_y_pairs:
        m.max_viol_cons.add(m.max_viol <= u + (1 - y) * bigM)
    m.obj = pe.Objective(expr=m.max_viol, sense=pe.maximize)

    res = opt.solve(m)
    assert res.termination_condition == appsi.base.TerminationCondition.optimal

    passed = res.best_feasible_objective <= feasibility_tol

    return passed


def is_equivalent(
    a: _BlockData,
    b: _BlockData,
    opt: appsi.base.Solver,
    feasibility_tol: float = 1e-6,
    bigM: Optional[float] = None,
):
    """
    Returns True if the feasible regions of a and b are the same
    a and b should share variables
    """
    cond1 = is_relaxation(a, b, opt=opt, feasibility_tol=feasibility_tol, bigM=bigM)
    cond2 = is_relaxation(b, a, opt=opt, feasibility_tol=feasibility_tol, bigM=bigM)
    return cond1 and cond2
