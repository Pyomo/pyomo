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

import pyomo.environ as pyo
from pyomo.common.collections import ComponentMap
from pyomo.gdp.util import clone_without_expression_components
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
from pyomo.contrib.alternative_solutions import aos_utils


def _get_unique_name(collection, name):
    """Create a unique name for an item that will be added to a collection."""
    if name not in collection:
        return name
    else:
        i = 1
        while "{}_{}".format(name, i) not in collection:
            i += 1
        return "{}_{}".format(name, i)


def _set_slack_ub(expression, slack_var):
    """
    Use FBBT to compute an upper bound for a slack variable on an equality
    expression."""
    slack_lb, slack_ub = compute_bounds_on_expr(expression)
    assert slack_ub >= 0
    slack_var.setub(slack_ub)


def get_shifted_linear_model(model, block=None):
    """
    Converts an (MI)LP with bounded (discrete and) continuous variables
    (l <= x <= u) into a standard form where where all continuous variables
    are non-negative reals and all constraints are equalities. For a pure LP of
    the form,

    .. math::

       min/max cx
       s.t.
           A_1 * x  =  b_1
           A_2 * x <= b_2
           l <= x <= u

    a problem of the form,

    .. math::

       min/max c'z
       s.t.
           Bz = q
           z >= 0

    will be created and added to the returned block. z consists of var_lower
    and var_upper variables that are substituted into the original x variables,
    and slack_vars that are used to convert the original inequalities to
    equalities. Bounds are provided on all variables in z. For MILPs, only the
    continuous part of the problem is converted.

    See Lee, Sangbum., C. Phalakornkule, M. Domach, I. Grossmann, Recursive
    MILP model for finding all the alternate optima in LP models for metabolic
    networks, Computers & Chemical Engineering, Volume 24, Issues 2–7, 2000,
    page 712 for additional details.

    Parameters
    ----------
    model : ConcreteModel
        A concrete Pyomo model
    block : Block
        The Pyomo block that the new model should be added to.

    Returns
    -------
    block
        The block that holds the reformulated model.
    """

    # Gather all variables and confirm the model is bounded
    all_vars = aos_utils.get_model_variables(model)
    new_vars = {}
    all_vars_new = {}
    var_map = ComponentMap()
    var_range = {}
    for var in all_vars:
        assert var.lb is not None, (
            "Variable {} does not have a "
            "lower bound. All variables must be "
            "bounded.".format(var.name)
        )
        assert var.ub is not None, (
            "Variable {} does not have an "
            "upper bound. All variables must be "
            "bounded.".format(var.name)
        )
        if var.is_continuous():
            var_name = _get_unique_name(new_vars.keys(), var.name)
            new_vars[var_name] = var
            all_vars_new[var_name] = var
            var_map[var] = var_name
            var_range[var_name] = (0, var.ub - var.lb)
        else:
            all_vars_new[var.name] = var

    if block is None:
        block = model
    shifted_lp = aos_utils._add_aos_block(block, name="_shifted_lp")

    # Replace original variables with shifted lower and upper variables
    shifted_lp.var_lower = pyo.Var(
        new_vars.keys(), domain=pyo.NonNegativeReals, bounds=var_range
    )
    shifted_lp.var_upper = pyo.Var(
        new_vars.keys(), domain=pyo.NonNegativeReals, bounds=var_range
    )

    # Link the shifted lower and upper variables
    def link_vars_rule(m, var_index):
        return (
            m.var_lower[var_index] + m.var_upper[var_index] == m.var_upper[var_index].ub
        )

    shifted_lp.link_vars = pyo.Constraint(new_vars.keys(), rule=link_vars_rule)

    # Map the lower and upper variables to the original variables and their
    # lower bounds. This will be used to substitute x with var_lower + x.lb.
    var_lower_map = {id(var): shifted_lp.var_lower[i] for i, var in new_vars.items()}
    var_lower_bounds = {id(var): var.lb for var in new_vars.values()}
    var_zeros = {id(var): 0 for var in all_vars_new.values()}

    # Substitute the new s variables into the objective function
    # The c_fix_zeros calculation is used to find any constant terms that exist
    # in the objective expression to avoid double counting
    active_objective = aos_utils.get_active_objective(model)
    c_var_lower = clone_without_expression_components(
        active_objective.expr, substitute=var_lower_map
    )
    c_fix_lower = clone_without_expression_components(
        active_objective.expr, substitute=var_lower_bounds
    )
    c_fix_zeros = clone_without_expression_components(
        active_objective.expr, substitute=var_zeros
    )
    shifted_lp.objective = pyo.Objective(
        expr=c_var_lower - c_fix_zeros + c_fix_lower,
        name=active_objective.name + "_shifted",
        sense=active_objective.sense,
    )

    # Identify all of the shifted constraints and associated slack variables
    # that will need to be created
    new_constraints = {}
    constraint_map = ComponentMap()
    constraint_type = {}
    slacks = []
    for constraint in model.component_data_objects(pyo.Constraint, active=True):
        if constraint.parent_block() == shifted_lp:
            continue
        if constraint.equality:
            constraint_name = constraint.name + "_equal"
            constraint_name = _get_unique_name(new_constraints.keys(), constraint.name)
            new_constraints[constraint_name] = constraint
            constraint_map[constraint] = constraint_name
            constraint_type[constraint_name] = 0
        else:
            if constraint.lb is not None:
                constraint_name = constraint.name + "_lower"
                constraint_name = _get_unique_name(
                    new_constraints.keys(), constraint.name
                )
                new_constraints[constraint_name] = constraint
                constraint_map[constraint] = constraint_name
                constraint_type[constraint_name] = -1
                slacks.append(constraint_name)
            if constraint.ub is not None:
                constraint_name = constraint.name + "_upper"
                constraint_name = _get_unique_name(
                    new_constraints.keys(), constraint.name
                )
                new_constraints[constraint_name] = constraint
                constraint_map[constraint] = constraint_name
                constraint_type[constraint_name] = 1
                slacks.append(constraint_name)
    shifted_lp.constraint_index = pyo.Set(initialize=new_constraints.keys())
    shifted_lp.slack_index = pyo.Set(initialize=slacks)
    shifted_lp.slack_vars = pyo.Var(shifted_lp.slack_index, domain=pyo.NonNegativeReals)
    shifted_lp.constraints = pyo.Constraint(shifted_lp.constraint_index)

    for constraint_name, constraint in new_constraints.items():
        # The c_fix_zeros calculation is used to find any constant terms that
        # exist in the constraint expression to avoid double counting
        a_sub_var_lower = clone_without_expression_components(
            constraint.body, substitute=var_lower_map
        )
        a_sub_fix_lower = clone_without_expression_components(
            constraint.body, substitute=var_lower_bounds
        )
        a_sub_fix_zeros = clone_without_expression_components(
            constraint.body, substitute=var_zeros
        )
        b_lower = constraint.lb
        b_upper = constraint.ub
        con_type = constraint_type[constraint_name]
        if con_type == 0:
            expr = a_sub_var_lower - a_sub_fix_zeros + a_sub_fix_lower - b_lower == 0
        elif con_type == -1:
            expr_rhs = a_sub_var_lower - a_sub_fix_zeros + a_sub_fix_lower - b_lower
            expr = shifted_lp.slack_vars[constraint_name] == expr_rhs
            _set_slack_ub(expr_rhs, shifted_lp.slack_vars[constraint_name])
        elif con_type == 1:
            expr_rhs = b_upper - a_sub_var_lower + a_sub_fix_zeros - a_sub_fix_lower
            expr = shifted_lp.slack_vars[constraint_name] == expr_rhs
            _set_slack_ub(expr_rhs, shifted_lp.slack_vars[constraint_name])
        shifted_lp.constraints[constraint_name] = expr

    shifted_lp.var_map = var_map
    shifted_lp.new_vars = new_vars
    shifted_lp.constraint_map = constraint_map
    shifted_lp.new_constraints = new_constraints

    return shifted_lp
