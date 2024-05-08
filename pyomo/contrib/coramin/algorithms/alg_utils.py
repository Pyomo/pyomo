import pyomo.environ as pe
from pyomo.repn.standard_repn import generate_standard_repn, StandardRepn
from pyomo.contrib.coramin.relaxations.split_expr import split_expr
from pyomo.core.expr.numeric_expr import LinearExpression
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.base.block import _BlockData
from pyomo.common.collections import ComponentSet
from typing import Tuple, List, Sequence


def collect_vars(m: _BlockData) -> Tuple[List[_GeneralVarData], List[_GeneralVarData]]:
    binary_vars = ComponentSet()
    integer_vars = ComponentSet()
    for v in m.vars:
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


def impose_structure(m):
    m.aux_vars = pe.VarList()

    for key, c in list(m.nonlinear.cons.items()):
        repn: StandardRepn = generate_standard_repn(
            c.body, quadratic=False, compute_values=True
        )
        expr_list = split_expr(repn.nonlinear_expr)
        if len(expr_list) == 1:
            continue

        linear_coefs = list(repn.linear_coefs)
        linear_vars = list(repn.linear_vars)
        for term in expr_list:
            v = m.aux_vars.add()
            linear_coefs.append(1)
            linear_vars.append(v)
            m.vars.append(v)
            if c.equality or (c.lb == c.ub and c.lb is not None):
                m.nonlinear.cons.add(v == term)
            elif c.ub is None:
                m.nonlinear.cons.add(v <= term)
            elif c.lb is None:
                m.nonlinear.cons.add(v >= term)
            else:
                m.nonlinear.cons.add(v == term)
        new_expr = LinearExpression(
            constant=repn.constant, linear_coefs=linear_coefs, linear_vars=linear_vars
        )
        m.linear.cons.add((c.lb, new_expr, c.ub))
        del m.nonlinear.cons[key]

    if hasattr(m.nonlinear, 'obj'):
        obj = m.nonlinear.obj
        repn: StandardRepn = generate_standard_repn(
            obj.expr, quadratic=False, compute_values=True
        )
        expr_list = split_expr(repn.nonlinear_expr)
        if len(expr_list) > 1:
            linear_coefs = list(repn.linear_coefs)
            linear_vars = list(repn.linear_vars)
            for term in expr_list:
                v = m.aux_vars.add()
                linear_coefs.append(1)
                linear_vars.append(v)
                m.vars.append(v)
                if obj.sense == pe.minimize:
                    m.nonlinear.cons.add(v >= term)
                else:
                    assert obj.sense == pe.maximize
                    m.nonlinear.cons.add(v <= term)
            new_expr = LinearExpression(
                constant=repn.constant,
                linear_coefs=linear_coefs,
                linear_vars=linear_vars,
            )
            m.linear.obj = pe.Objective(expr=new_expr, sense=obj.sense)
            del m.nonlinear.obj
