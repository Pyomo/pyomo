#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  __________________________________________________________________________
#
#
""" Pyomo Units Checking Module
This module has some helpful methods to support checking units on Pyomo
module objects.
"""
from pyomo.core.base.units_container import units, UnitsError
from pyomo.core.base import (Objective, Constraint, Var, Param,
                             Suffix, Set, RangeSet, Block,
                             ExternalFunction, Expression,
                             value, BooleanVar)
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.network import Port, Arc
from pyomo.mpec import Complementarity
from pyomo.gdp import Disjunct, Disjunction
from pyomo.core.expr.template_expr import IndexTemplate
from pyomo.core.expr.numvalue import native_types
from pyomo.util.components import iter_component

def check_units_equivalent(*args):
    """
    Returns True if the units associated with each of the
    expressions passed as arguments are all equivalent (and False
    otherwise).

    Note that this method will raise an exception if the units are
    inconsistent within an expression (since the units for that
    expression are not valid).

    Parameters
    ----------
    args : an argument list of Pyomo expressions

    Returns
    -------
    bool : True if all the expressions passed as argments have the same units
    """
    try:
        assert_units_equivalent(*args)
        return True
    except UnitsError:
        return False

def assert_units_equivalent(*args):
    """
    Raise an exception if the units are inconsistent within an
    expression, or not equivalent across all the passed
    expressions.

    Parameters
    ----------
    args : an argument list of Pyomo expressions
        The Pyomo expressions to test

    Raises
    ------
    :py:class:`pyomo.core.base.units_container.UnitsError`, :py:class:`pyomo.core.base.units_container.InconsistentUnitsError`
    """
    # this call will raise an exception if an inconsistency is found
    pint_units = [units._get_pint_units(arg) for arg in args]
    pint_unit_compare = pint_units[0]
    for pint_unit in pint_units:
        if not units._equivalent_pint_units(pint_unit_compare, pint_unit):
            raise UnitsError(
                "Units between {} and {} are not consistent.".format(
                    str(pint_unit_compare), str(pint_unit)))

def _assert_units_consistent_constraint_data(condata):
    """
    Raise an exception if the any units in lower, body, upper on a
    ConstraintData object are not consistent or are not equivalent
    with each other.
    """
    # Pyomo can rearrange expressions, resulting in a value
    # of 0 for upper or lower that does not have units associated
    # Therefore, if the lower and/or upper is 0, we allow it to be unitless
    # and check the consistency of the body only
    args = list()
    if condata.lower is not None and value(condata.lower) != 0.0:
        args.append(condata.lower)

    args.append(condata.body)

    if condata.upper is not None and value(condata.upper) != 0.0:
        args.append(condata.upper)

    if len(args) == 1:
        assert_units_consistent(*args)
    else:
        assert_units_equivalent(*args)

def _assert_units_consistent_arc_data(arcdata):
    """
    Raise an exception if the any units do not match for the connected ports
    """
    sport = arcdata.source
    dport = arcdata.destination
    if sport is None or dport is None:
        # nothing to check
        return

    # both sport and dport are not None
    # iterate over the vars in one and check against the other
    for key in sport.vars:
        svar = sport.vars[key]
        dvar = dport.vars[key]

        if svar.is_indexed():
            for k in svar:
                svardata = svar[k]
                dvardata = dvar[k]
                assert_units_equivalent(svardata, dvardata)
        else:
            assert_units_equivalent(svar, dvar)

def _assert_units_consistent_property_expr(obj):
    """
    Check the .expr property of the object and raise
    an exception if the units are not consistent
    """
    _assert_units_consistent_expression(obj.expr)

def _assert_units_consistent_expression(expr):
    """
    Raise an exception if any units in expr are inconsistent.
    """
    # this will raise an exception if the units are not consistent
    # in the expression
    pint_unit = units._get_pint_units(expr)
    # pyomo_unit = units.get_units(expr)

# Complementarities that are not in standard form do not
# current work with the checking code. The Units container
# should be modified to allow sum and relationals with zero
# terms (e.g., unitless). Then this code can be enabled.
#def _assert_units_complementarity(cdata):
#    """
#    Raise an exception if any units in either of the complementarity
#    expressions are inconsistent, and also check the standard block
#    methods.
#    """
#    if cdata._args[0] is not None:
#        pyomo_unit, pint_unit = units._get_units_tuple(cdata._args[0])
#    if cdata._args[1] is not None:
#        pyomo_unit, pint_unit = units._get_units_tuple(cdata._args[1])
#    _assert_units_consistent_block(cdata)

def _assert_units_consistent_block(obj):
    """
    This method gets all the components from the block
    and checks if the units are consistent on each of them
    """
    # check all the component objects
    for component in obj.component_objects(descend_into=False, active=True):
        assert_units_consistent(component)

_component_data_handlers = {
    Objective: _assert_units_consistent_property_expr,
    Constraint:  _assert_units_consistent_constraint_data,
    Var: _assert_units_consistent_expression,
    DerivativeVar: _assert_units_consistent_expression,
    Port: None,
    Arc: _assert_units_consistent_arc_data,
    Expression: _assert_units_consistent_property_expr,
    Suffix: None,
    Param: _assert_units_consistent_expression,
    Set: None,
    RangeSet: None,
    Disjunct: _assert_units_consistent_block,
    Disjunction: None,
    BooleanVar: None,
    Block: _assert_units_consistent_block,
    ExternalFunction: None,
    ContinuousSet: None, # ToDo: change this when continuous sets have units assigned
    # complementarities that are not in normal form are not working yet
    # see comment in test_check_units
    # Complementarity: _assert_units_complementarity
    }

def assert_units_consistent(obj):
    """
    This method raises an exception if the units are not
    consistent on the passed in object.  Argument obj can be one
    of the following components: Pyomo Block (or Model),
    Constraint, Objective, Expression, or it can be a Pyomo
    expression object

    Parameters
    ----------
    obj : Pyomo component (e.g., Block, Model, Constraint, Objective, or Expression) or Pyomo expression
       The object or expression to test

    Raises
    ------
    :py:class:`pyomo.core.base.units_container.UnitsError`, :py:class:`pyomo.core.base.units_container.InconsistentUnitsError`
    """
    objtype = type(obj)
    if objtype in native_types:
        return
    elif obj.is_expression_type() or objtype is IndexTemplate:
        try:
            _assert_units_consistent_expression(obj)
        except UnitsError:
            print('Units problem with expression {}'.format(obj))
            raise
        return

    # if object is not in our component handler, raise an exception
    if obj.ctype not in _component_data_handlers:
        raise TypeError("Units checking not supported for object of type {}.".format(obj.ctype))

    # get the function form the list of handlers
    handler = _component_data_handlers[obj.ctype]
    if handler is None:
        return

    if obj.is_indexed():
        # check all the component data objects
        for cdata in obj.values():
            try:
                handler(cdata)
            except UnitsError:
                print('Error in units when checking {}'.format(cdata))
                raise
    else:
        try:
            handler(obj)
        except UnitsError:
                print('Error in units when checking {}'.format(obj))
                raise
            
