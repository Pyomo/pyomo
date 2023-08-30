#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  ___________________________________________________________________________
#
#  This module was originally developed as part of the IDAES PSE Framework
#
#  Institute for the Design of Advanced Energy Systems Process Systems
#  Engineering Framework (IDAES PSE Framework) Copyright (c) 2018-2019, by the
#  software owners: The Regents of the University of California, through
#  Lawrence Berkeley National Laboratory,  National Technology & Engineering
#  Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia
#  University Research Corporation, et al. All rights reserved.
#
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.collections import ComponentSet
from pyomo.core.expr import identify_variables
from pyomo.environ import Constraint, value


def value_no_exception(c, div0=None):
    """
    Get value and ignore most exceptions (including division by 0).

    Args:
        c: a Pyomo component to get the value of
    Returns:
        A value, could be None
    """
    try:
        return value(c, exception=False)
    except ZeroDivisionError:
        return div0


def get_residual(ui_data, c):
    """
    Calculate the residual (constraint violation) of a constraint. This residual
    is always positive.

    Args:
        ui_data (UIData): user interface data, includes the cache for calculated
            values of the constraint body. This function uses the cached values
            and will not trigger recalculation. If variable values have changed,
            this may not yield accurate results.
        c(_ConstraintData): a constraint or constraint data
    Returns:
        (float) residual
    """
    if c.upper is None:
        ub = None  # This is no upper bound
    else:
        ub = value_no_exception(c.upper, "Divide_by_0")
        if ub is None or isinstance(ub, str):
            # This is calc error
            return ub
    if c.lower is None:
        lb = None  # This is no lower bound
    else:
        lb = value_no_exception(c.lower, "Divide_by_0")
        if lb is None or isinstance(lb, str):
            # This is calc error
            return lb
    try:
        v = ui_data.value_cache[c]
    except KeyError:
        return None
    if v is None or isinstance(v, str):
        return v
    if lb is not None and v < lb:
        return lb - v
    if ub is not None and v > ub:
        return v - ub
    return 0.0


def active_equalities(blk):
    """
    Generator returning active equality constraints in a model.

    Args:
        blk: a Pyomo block in which to look for variables.
    """
    for o in blk.component_data_objects(Constraint, active=True):
        try:
            u = value(o.upper, exception=False)
            l = value(o.lower, exception=False)
            if u == l and l is not None:
                yield o
        except ZeroDivisionError:
            pass


def active_constraint_set(blk):
    """
    Return a set of active constraints in a model.

    Args:
        blk: a Pyomo block in which to look for constraints.
    Returns:
        (ComponentSet): Active equality constraints
    """
    return ComponentSet(blk.component_data_objects(Constraint, active=True))


def active_equality_set(blk):
    """
    Return a set of active equalities.

    Args:
        blk: a Pyomo block in which to look for variables.
    Returns:
        (ComponentSet): Active constraints
    """
    return ComponentSet(active_equalities(blk))


def count_free_variables(blk):
    """
    Count free variables that are in active equality constraints.  Ignore
    inequalities, because this is used in the degrees of freedom calculations
    """
    return len(free_variables_in_active_equalities_set(blk))


def count_equality_constraints(blk):
    """
    Count active equality constraints.
    """
    return len(active_equality_set(blk))


def count_constraints(blk):
    """
    Count active constraints.
    """
    return len(active_constraint_set(blk))


def degrees_of_freedom(blk):
    """
    Return the degrees of freedom.

    Args:
        blk (Block or _BlockData): Block to count degrees of freedom in
    Returns:
        (int): Number of degrees of freedom
    """
    return count_free_variables(blk) - count_equality_constraints(blk)


def free_variables_in_active_equalities_set(blk):
    """
    Return a set of variables that are continued in active equalities.
    """
    vin = ComponentSet()
    for c in active_equalities(blk):
        for v in identify_variables(c.body):
            if not v.fixed:
                vin.add(v)
    return vin
