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
from pyomo.core.base.units_container import units, UnitsError, UnitExtractionVisitor
from pyomo.core.base.objective import Objective
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.var import Var
from pyomo.core.base.param import Param
from pyomo.core.base.suffix import Suffix
from pyomo.core.base.set import Set, RangeSet
from pyomo.gdp import Disjunct
from pyomo.gdp import Disjunction
from pyomo.core.base.block import Block
from pyomo.core.base.external import ExternalFunction
from pyomo.core.base.expression import Expression
from pyomo.core.expr.numvalue import native_types

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
    pyomo_unit_compare, pint_unit_compare = units._get_units_tuple(args[0])
    for expr in args[1:]:
        pyomo_unit, pint_unit = units._get_units_tuple(expr)
        if not UnitExtractionVisitor(self)._pint_units_equivalent(pint_unit_compare, pint_unit):
            return False
    # made it through all of them successfully
    return True

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
    pyomo_unit_compare, pint_unit_compare = units._get_units_tuple(args[0])
    for expr in args[1:]:
        # this call will raise an exception if an inconsistency is found
        pyomo_unit, pint_unit = units._get_units_tuple(expr)
        if not UnitExtractionVisitor(units)._pint_units_equivalent(pint_unit_compare, pint_unit):
            raise UnitsError \
                ("Units between {} and {} are not consistent.".format(str(pyomo_unit_compare), str(pyomo_unit)))

def _assert_units_consistent_constraint_data(condata):
    """
    Raise an exception if the any units in lower, body, upper on a
    ConstraintData object are not consistent or are not equivalent
    with each other.
    """
    if condata.equality:
        if condata.lower == 0.0:
            # Pyomo can rearrange expressions, resulting in a value
            # of 0 for the RHS that does not have units associated
            # Therefore, if the RHS is 0, we allow it to be unitless
            # and check the consistency of the body only
            assert condata.upper == 0.0
            _assert_units_consistent_expression(condata.body)
        else:
            assert_units_equivalent(condata.lower, condata.body)
    else:
        assert_units_equivalent(condata.lower, condata.body, condata.upper)

def _assert_units_consistent_property_expr(obj):
    """
    Check the .expr property of the object and raise
    an exception if the units are not consistent
    """
    _assert_units_consistent_expression(obj.expr)

def _assert_units_consistent_expression(expr):
    """
    Raise an exception if any units in expr are inconsistent.
    # this call will raise an error if an inconsistency is found
    pyomo_unit, pint_unit = units._get_units_tuple(expr=expr)
    """
    pyomo_unit, pint_unit = units._get_units_tuple(expr)

def _assert_units_consistent_block(obj):
    """
    This method gets all the components from the block
    and checks if the units are consistent on each of them
    """
    # check all the component objects
    for component in obj.component_objects(descend_into=True):
        assert_units_consistent(component)

_component_data_handlers = {
    Objective: _assert_units_consistent_property_expr,
    Constraint:  _assert_units_consistent_constraint_data,
    Var: _assert_units_consistent_expression,
    Expression: _assert_units_consistent_property_expr,
    Suffix: None,
    Param: _assert_units_consistent_expression,
    Set: None,
    RangeSet: None,
    Disjunct:_assert_units_consistent_block,
    Disjunction: None,
    Block: _assert_units_consistent_block,
    ExternalFunction: None
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
    if obj in native_types:
        return
    elif obj.is_expression_type():
        _assert_units_consistent_expression(obj)

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
            handler(cdata)
    else:
        handler(obj)
