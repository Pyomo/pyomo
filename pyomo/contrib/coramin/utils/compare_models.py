import pyomo.environ as pe
from pyomo.contrib.coramin.clone import clone_shallow_active_flat
from pyomo.core.base.block import _BlockData
from typing import Optional
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
from pyomo.contrib import appsi


def is_relaxation(a: _BlockData, b: _BlockData, opt: appsi.base.Solver, feasibility_tol: float = 1e-6, bigM: Optional[float] = None):
    """
    Returns True if every feasible point in b is feasible for a
    (a is a relaxation of b)
    a and b should share variables
    """
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

    return res.best_feasible_objective <= feasibility_tol


def is_equivalent(a: _BlockData, b: _BlockData, opt: appsi.base.Solver, feasibility_tol: float = 1e-6, bigM: Optional[float] = None):
    """
    Returns True if the feasible regions of a and b are the same
    a and b should share variables
    """
    cond1 = is_relaxation(a, b, opt=opt, feasibility_tol=feasibility_tol, bigM=bigM)
    cond2 = is_relaxation(b, a, opt=opt, feasibility_tol=feasibility_tol, bigM=bigM)
    return cond1 and cond2
