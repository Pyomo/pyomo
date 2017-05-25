"""
The classes defined here are meant to facilitate the direct use of gurobipy through pyomo.
"""
import pyomo
import pyomo.environ as pe
import gurobipy


_using_pyomo4_trees = False
if pyomo.core.base.expr_common.mode == pyomo.core.base.expr_common.Mode.pyomo4_trees:
    _using_pyomo4_trees = True
assert _using_pyomo4_trees is False


def gurobi_vtype_from_domain(domain):
    if domain == pyomo.core.base.set_types.Reals:
        vtype = gurobipy.GRB.CONTINUOUS
    elif domain == pyomo.core.base.set_types.Binary:
        vtype = gurobipy.GRB.BINARY
    else:
        raise ValueError('Variable type is not recognized for {0}'.format(domain))
    return vtype


def get_gurobipy_expr_from_pyomo_expr(expr):
    expr_type = type(expr)
    assert expr_type is not list
    if expr.is_expression():
        if expr_type is pyomo.core.base.expr._SumExpression:
            return sum(expr._coef[i] * get_gurobipy_expr_from_pyomo_expr(expr._args[i]) for i in range(len(expr._args)))
        elif expr_type is pyomo.core.base.expr._ProductExpression:
            new_expr = expr._coef
            for i, child in enumerate(expr._args):
                new_expr *= get_gurobipy_expr_from_pyomo_expr(child)
            return new_expr
        elif expr_type is pyomo.core.base.expr._PowExpression:
            
    elif isinstance(expr, pyomo.core.base.var._VarData):
        return expr.gurobipy_var
    else:
        raise ValueError('Unrecognized pyomo expression type: {0}'.format(expr_type))


class GurobiModelContainer(object):
    """
    A container for gurobipy models

    Attributes
    ----------
    model: gurobi.Model
    """

    def __init__(self):
        self.model = gurobipy.Model()


class GurobiModel(pe.ConcreteModel):
    """
    A class to facilitate the use of gurobipy

    Attributes
    ----------
    gmc: GurobiModelContainer
    """

    def __init__(self):
        super(GurobiModel, self).__init__()
        self.gmc = GurobiModelContainer()

    def __setattr__(self, name, value):
        super(GurobiModel, self).__setattr__(name, value)

        if isinstance(value, pe.Var):
            self._add_gurobipy_var_to_gurobi_model(name, value)

        elif isinstance(value, pe.ConstraintList):
            value.gurobi_model = self

        elif isinstance(value, pe.Constraint):
            self._add_gurobipy_constraint_to_gurobi_model(value)

    def _add_gurobipy_var_to_gurobi_model(self, name, value):
        if value.is_indexed():
            index_set = list(value.index_set())
            domain = value[index_set[0]].domain
            for i in index_set:
                assert value[i].domain == domain
            vtype = gurobi_vtype_from_domain(domain)
            lb = {i: value[i].lb if value[i].lb is not None else -gurobipy.GRB.INFINITY for i in index_set}
            ub = {i: value[i].ub if value[i].lb is not None else gurobipy.GRB.INFINITY for i in index_set}
            gurobipy_var = self.gmc.model.addVars(index_set, lb=lb, ub=ub, vtype=vtype, name=name)
            setattr(self.gmc, name, gurobipy_var)
            for i in index_set:
                setattr(value[i], 'gurobipy_var', gurobipy_var[i])

        else:
            domain = value.domain
            vtype = gurobi_vtype_from_domain(domain)
            lb = value.lb
            ub = value.ub
            if lb is None:
                lb = -gurobipy.GRB.INFINITY
            if ub is None:
                ub = gurobipy.GRB.INFINITY
            gurobipy_var = self.gmc.model.addVar(lb=lb, ub=ub, vtype=vtype, name=name)
            setattr(self.gmc, name, gurobipy_var)
            setattr(value, 'gurobipy_var', gurobipy_var)

    def _add_gurobipy_constraint_to_gurobi_model(self, value):
        if value.is_indexed():
            index_set = list(value.index_set())
            for i in index_set:
                con = value[i]
                expr = con.expr
                get_gurobipy_expr_from_pyomo_expr(expr)
        else:
            con = value
            expr = con.expr
            get_gurobipy_expr_from_pyomo_expr(expr)
