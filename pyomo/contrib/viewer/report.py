##############################################################################
# Institute for the Design of Advanced Energy Systems Process Systems
# Engineering Framework (IDAES PSE Framework) Copyright (c) 2018, by the
# software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia
# University Research Corporation, et al. All rights reserved.
#
# This software is distributed under the 3-clause BSD License.
##############################################################################
from pyomo.environ import *
from pyomo.core.expr.current import identify_variables
from pyomo.core.kernel.component_set import ComponentSet
from pyomo.network.port import _PortData, SimplePort

def large_residuals(blk, tol=1e-5):
    """
    Generator return active Pyomo constraints with residuals greater than tol.

    Args:
        blk: a Pyomo block in which to look for constraints
        tol: show constraints with residuals greated than tol
    """
    for o in blk.component_objects(Constraint, descend_into=True):
        for i, c in o.iteritems():
            if c.active and value(c.lower - c.body()) > tol:
                yield c
            elif c.active and value(c.body() - c.upper) > tol:
                yield c

def fixed_variables(blk):
    """
    Generator returning fixed variables in a model.

    Args:
        blk: a Pyomo block in which to look for variables.
    """
    for o in blk.component_data_objects(Var):
        if o.fixed: yield o

def unfixed_variables(blk):
    """
    Generator returning free variables in a model.

    Args:
        blk: a Pyomo block in which to look for variables.
    """
    for o in blk.component_data_objects(Var):
        if not o.fixed: yield item

def free_variables(blk):
    """
    Generator returning free variables in a model. same as unfixed

    Args:
        blk: a Pyomo block in which to look for variables.
    """
    for o in blk.component_data_objects(Var):
        if not o.fixed: yield o

def stale_variables(blk):
    """
    Generator returning stale variables in a model.

    Args:
        blk: a Pyomo block in which to look for variables.
    """
    for o in blk.component_data_objects(Var):
        if not o.fixed and o.stale: yield o

def active_equalities(blk):
    """
    Generator returning active equality constraints in a model.

    Args:
        blk: a Pyomo block in which to look for variables.
    """
    for o in blk.component_data_objects(Constraint, active=True):
        if o.upper == o.lower: yield o

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

def degrees_of_freedom(blk):
    """
    Return the degrees of freedom.
    """
    return count_free_variables(blk) - count_equality_constraints(blk)

def active_equality_set(blk):
    """
    Generator returning active equality constraints in a model.

    Args:
        blk: a Pyomo block in which to look for variables.
    """
    ac = ComponentSet()
    for c in active_equalities(blk):
        ac.add(c)
    return ac

def variables_in_active_equalities_set(blk):
    """
    Return a set of variables that are contined in active equalities.
    """
    vin = ComponentSet()
    for c in active_equalities(blk):
        for v in identify_variables(c.body):
            vin.add(v)
    return vin

def free_variables_in_active_equalities_set(blk):
    """
    Return a set of variables that are contined in active equalities.
    """
    vin = ComponentSet()
    for c in active_equalities(blk):
        for v in identify_variables(c.body):
            if not v.fixed: vin.add(v)
    return vin
