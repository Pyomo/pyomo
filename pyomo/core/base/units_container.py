#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#
#

"""Pyomo Units Container Module

This module provides support for including units within Pyomo expressions. This module
can be used to define units on a model, and to check the consistency of units
within the underlying constraints and expressions in the model. The module also
supports conversion of units within expressions using the `convert` method to support
construction of constraints that contain embedded unit conversions.

To use this package within your Pyomo model, you first need an instance of a
PyomoUnitsContainer. You can use the module level instance already defined as
'units'. This object 'contains' the units - that is, you can access units on
this module using common notation.

    .. doctest::
       :skipif: not pint_available

       >>> from pyomo.environ import units as u
       >>> print(3.0*u.kg)
       3.0*kg

Units can be assigned to Var, Param, and ExternalFunction components, and can
be used directly in expressions (e.g., defining constraints). You can also
verify that the units are consistent on a model, or on individual components
like the objective function, constraint, or expression using
`assert_units_consistent` (from pyomo.util.check_units).
There are other methods there that may be helpful for verifying correct units on a model.

    .. doctest::
       :skipif: not pint_available

       >>> from pyomo.environ import ConcreteModel, Var, Objective
       >>> from pyomo.environ import units as u
       >>> from pyomo.util.check_units import assert_units_consistent, assert_units_equivalent, check_units_equivalent
       >>> model = ConcreteModel()
       >>> model.acc = Var(initialize=5.0, units=u.m/u.s**2)
       >>> model.obj = Objective(expr=(model.acc - 9.81*u.m/u.s**2)**2)
       >>> assert_units_consistent(model.obj) # raise exc if units invalid on obj
       >>> assert_units_consistent(model) # raise exc if units invalid anywhere on the model
       >>> assert_units_equivalent(model.obj.expr, u.m**2/u.s**4) # raise exc if units not equivalent
       >>> print(u.get_units(model.obj.expr)) # print the units on the objective
       m**2/s**4
       >>> print(check_units_equivalent(model.acc, u.m/u.s**2))
       True

The implementation is currently based on the `pint
<http://pint.readthedocs.io>`_ package and supports all the units that
are supported by pint.  The list of units that are supported by pint
can be found at the following url:
https://github.com/hgrecco/pint/blob/master/pint/default_en.txt.

If you need a unit that is not in the standard set of defined units,
you can create your own units by adding to the unit definitions within
pint. See :py:meth:`PyomoUnitsContainer.load_definitions_from_file` or
:py:meth:`PyomoUnitsContainer.load_definitions_from_strings` for more
information.

.. note:: In this implementation of units, "offset" units for
          temperature are not supported within expressions (i.e. the
          non-absolute temperature units including degrees C and
          degrees F).  This is because there are many non-obvious
          combinations that are not allowable. This concern becomes
          clear if you first convert the non-absolute temperature
          units to absolute and then perform the operation. For
          example, if you write 30 degC + 30 degC == 60 degC, but
          convert each entry to Kelvin, the expression is not true
          (i.e., 303.15 K + 303.15 K is not equal to 333.15
          K). Therefore, there are several operations that are not
          allowable with non-absolute units, including addition,
          multiplication, and division.

          This module does support conversion of offset units to
          absolute units numerically, using convert_value_K_to_C,
          convert_value_C_to_K, convert_value_R_to_F,
          convert_value_F_to_R.  These are useful for converting input
          data to absolute units, and for converting data to
          convenient units for reporting.

          Please see the pint documentation `here
          <https://pint.readthedocs.io/en/0.9/nonmult.html>`_ for more
          discussion. While pint implements "delta" units (e.g.,
          delta_degC) to support correct unit conversions, it can be
          difficult to identify and guarantee valid operations in a
          general algebraic modeling environment. While future work
          may support units with relative scale, the current
          implementation requires use of absolute temperature units
          (i.e. K and R) within expressions and a direct conversion of
          numeric values using specific functions for converting input
          data and reporting.

"""
# TODO
#    * create a new pint unit definition file (and load from that file)
#       since the precision in pint seems insufficient for 1e-8 constraint tolerances
#    * Investigate when we can and cannot handle offset units and expand capabilities if possible
#    * Further investigate issues surrounding absolute and relative temperatures (delta units)
#    * Extend external function interface to support units for the arguments in addition to the function itself

import logging
import sys

from pyomo.common.dependencies import attempt_import
from pyomo.common.modeling import NOTSET
from pyomo.core.expr.numvalue import (
    NumericValue,
    nonpyomo_leaf_types,
    value,
    native_types,
    native_numeric_types,
    pyomo_constant_types,
)
from pyomo.core.expr.template_expr import IndexTemplate
from pyomo.core.expr.visitor import ExpressionValueVisitor
import pyomo.core.expr as EXPR

pint_module, pint_available = attempt_import(
    'pint',
    defer_check=True,
    error_message=(
        'The "pint" package failed to import. '
        'This package is necessary to use Pyomo units.'
    ),
)

logger = logging.getLogger(__name__)


class UnitsError(Exception):
    """
    An exception class for all general errors/warnings associated with units
    """

    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return str(self.msg)


class InconsistentUnitsError(UnitsError):
    """
    An exception indicating that inconsistent units are present on an expression.

    E.g., x == y, where x is in units of kg and y is in units of meter
    """

    def __init__(self, exp1, exp2, msg):
        msg = f'{msg}: {exp1} not compatible with {exp2}.'
        super(InconsistentUnitsError, self).__init__(msg)


def _pint_unit_mapper(encode, val):
    if encode:
        return str(val)
    else:
        return units._pint_registry(val).units


def _pint_registry_mapper(encode, val):
    if encode:
        if val is not units._pint_registry:
            # FIXME: we currently will not correctly unpickle units
            # associated with a unit manager other than the default
            # singleton.  If we wanted to support this, we would need to
            # do something like create a global units manager registry
            # that would associate each unit manager with a name.  We
            # could then pickle that name and then attempt to restore
            # the association with the original units manager.  As we
            # expect all users to just use the global default, for the
            # time being we will just issue a warning that things may
            # break.
            logger.warning(
                "pickling a _PyomoUnit associated with a PyomoUnitsContainer "
                "that is not the default singleton (%s.units).  Restoring "
                "this state will attempt to return a unit associated with "
                "the default singleton." % (__name__,)
            )
        return None
    elif val is None:
        return units._pint_registry
    else:
        return val


class _PyomoUnit(NumericValue):
    """An object that represents a single unit in Pyomo (e.g., kg, meter)

    Users should not create instances of _PyomoUnit directly, but rather access
    units as attributes on an instance of a :class:`PyomoUnitsContainer`.
    This module contains a global PyomoUnitsContainer object :py:data:`units`.
    See module documentation for more information.
    """

    __slots__ = ('_pint_unit', '_pint_registry')
    __autoslot_mappers__ = {
        '_pint_unit': _pint_unit_mapper,
        '_pint_registry': _pint_registry_mapper,
    }

    def __init__(self, pint_unit, pint_registry):
        super(_PyomoUnit, self).__init__()
        assert pint_unit is not None
        assert pint_registry is not None
        self._pint_unit = pint_unit
        self._pint_registry = pint_registry

    def _get_pint_unit(self):
        """Return the pint unit corresponding to this Pyomo unit."""
        return self._pint_unit

    def _get_pint_registry(self):
        """Return the pint registry (pint.UnitRegistry) object used to create this unit."""
        return self._pint_registry

    def getname(self, fully_qualified=False, name_buffer=None):
        """
        Returns the name of this unit as a string.
        Overloaded from: :py:class:`NumericValue`. See this class for a description of the
        arguments. The value of these arguments are ignored here.

        Returns
        -------
        : str
           Returns the name of the unit
        """
        return str(self)

    # methods/properties that use the NumericValue base class implementation
    # name property
    # local_name
    # cname

    def is_constant(self):
        """
        Indicates if the NumericValue is constant and can be replaced with a plain old number
        Overloaded from: :py:class:`NumericValue`

        This method indicates if the NumericValue is a constant and can be replaced with a plain
        old number. Although units are, in fact, constant, we do NOT want this replaced - therefore
        we return False here to prevent replacement.

        Returns
        =======
        : bool
           False (This method always returns False)
        """
        return False

    def is_fixed(self):
        """
        Indicates if the NumericValue is fixed with respect to a "solver".
        Overloaded from: :py:class:`NumericValue`

        Indicates if the Unit should be treated as fixed. Since the Unit is always treated as
        a constant value of 1.0, it is fixed.

        Returns
        =======
        : bool
           True (This method always returns True)

        """
        return True

    def is_parameter_type(self):
        """This is not a parameter type (overloaded from NumericValue)"""
        return False

    def is_variable_type(self):
        """This is not a variable type (overloaded from NumericValue)"""
        return False

    def is_potentially_variable(self):
        """
        This is not potentially variable (does not and cannot contain a variable).
        Overloaded from NumericValue
        """
        return False

    def is_named_expression_type(self):
        """This is not a named expression (overloaded from NumericValue)"""
        return False

    def is_expression_type(self, expression_system=None):
        """This is a leaf, not an expression (overloaded from NumericValue)"""
        return False

    def is_component_type(self):
        """This is not a component type (overloaded from NumericValue)"""
        return False

    def is_indexed(self):
        """This is not indexed (overloaded from NumericValue)"""
        return False

    def _compute_polynomial_degree(self, result):
        """Returns the polynomial degree - since units are constants, they have degree of zero.
        Note that :py:meth:`NumericValue.polynomial_degree` calls this method.
        """
        return 0

    def __deepcopy__(self, memo):
        # Note that while it is possible to deepcopy the _pint_unit and
        # _pint_registry object (in pint>0.10), that version does not
        # support all Python versions currently supported by Pyomo.
        # Further, Pyomo's use of units relies on a model using a single
        # instance of the pint unit registry.  As we regularly assemble
        # block models using multiple clones (deepcopies) of a base
        # model, it is important that we treat _PyomoUnit objects
        # as outside the model scope and DO NOT duplicate them.
        return self

    def __eq__(self, other):
        if other.__class__ is _PyomoUnit:
            return (
                self._pint_registry is other._pint_registry
                and self._pint_unit == other._pint_unit
            )
        return super().__eq__(other)

    # __bool__ uses NumericValue base class implementation
    # __float__ uses NumericValue base class implementation
    # __int__ uses NumericValue base class implementation
    # __lt__ uses NumericValue base class implementation
    # __gt__ uses NumericValue base class implementation
    # __le__ uses NumericValue base class implementation
    # __ge__ uses NumericValue base class implementation
    # __eq__ uses NumericValue base class implementation
    # __add__ uses NumericValue base class implementation
    # __sub__ uses NumericValue base class implementation
    # __mul__ uses NumericValue base class implementation
    # __div__ uses NumericValue base class implementation
    # __truediv__ uses NumericValue base class implementation
    # __pow__ uses NumericValue vase class implementation
    # __radd__ uses NumericValue base class implementation
    # __rsub__ uses NumericValue base class implementation
    # __rmul__ uses NumericValue base class implementation
    # __rdiv__ uses NumericValue base class implementation
    # __rtruediv__ uses NumericValue base class implementation
    # __rpow__ uses NumericValue base class implementation
    # __iadd__ uses NumericValue base class implementation
    # __isub__ uses NumericValue base class implementation
    # __imul__ uses NumericValue base class implementation
    # __idiv__ uses NumericValue base class implementation
    # __itruediv__ uses NumericValue base class implementation
    # __ipow__ uses NumericValue base class implementation
    # __neg__ uses NumericValue base class implementation
    # __pos__ uses NumericValue base class implementation
    # __add__ uses NumericValue base class implementation

    def __str__(self):
        """Returns a string representing the unit"""

        # The ~ returns the short form of the pint unit if the unit is
        # an instance of the unit 'dimensionless', then pint returns ''
        # which causes problems with some string processing in Pyomo
        # that expects a name
        #
        # Note: Some pint units contain unicode characters (notably
        # delta temperatures).  So that things work cleanly in Python 2
        # and 3, we will generate the string as unicode, then explicitly
        # encode it to UTF-8 in Python 2
        retstr = u'{:~C}'.format(self._pint_unit)
        if retstr == '':
            retstr = 'dimensionless'
        return retstr

    def to_string(self, verbose=None, labeler=None, smap=None, compute_values=False):
        """
        Return a string representation of the expression tree.

        See documentation on :py:class:`NumericValue`

        Returns
        -------
        : bool
           A string representation for the expression tree.
        """
        _str = str(self)
        if any(map(_str.__contains__, ' */')):
            return "(" + _str + ")"
        else:
            return _str

    def __call__(self, exception=True):
        """Unit is treated as a constant value, and this method always returns 1.0

        Returns
        -------
        : float
           Returns 1.0
        """
        return 1.0

    @property
    def value(self):
        return 1.0

    def pprint(self, ostream=None, verbose=False):
        """Display a user readable string description of this object."""
        if ostream is None:  # pragma:nocover
            ostream = sys.stdout
        ostream.write(str(self))
        # There is also a long form, but the verbose flag is not really the correct indicator
        # if verbose:
        #     ostream.write('{:s}'.format(self._pint_unit))
        # else:
        #     ostream.write('{:~s}'.format(self._pint_unit))


class PintUnitExtractionVisitor(EXPR.StreamBasedExpressionVisitor):
    def __init__(self, pyomo_units_container, units_equivalence_tolerance=1e-12):
        """
        Visitor class used to determine units of an expression. Do not use
        this class directly, but rather use
        "py:meth:`PyomoUnitsContainer.assert_units_consistent`
        or :py:meth:`PyomoUnitsContainer.get_units`

        Parameters
        ----------
        pyomo_units_container : PyomoUnitsContainer
           Instance of the PyomoUnitsContainer that was used for the units
           in the expressions. Pyomo does not support "mixing" units from
           different containers

        units_equivalence_tolerance : float (default 1e-12)
            Floating point tolerance used when deciding if units are equivalent
            or not.

        Notes
        -----
        This class inherits from the :class:`StreamBasedExpressionVisitor` to implement
        a walker that returns the pyomo units and pint units corresponding to an
        expression.

        There are class attributes (dicts) that map the expression node type to the
        particular method that should be called to return the units of the node based
        on the units of its child arguments. This map is used in exitNode.
        """
        super(PintUnitExtractionVisitor, self).__init__()
        self._pyomo_units_container = pyomo_units_container
        self._pint_dimensionless = None
        self._equivalent_pint_units = pyomo_units_container._equivalent_pint_units
        self._equivalent_to_dimensionless = (
            pyomo_units_container._equivalent_to_dimensionless
        )

    def _get_unit_for_equivalent_children(self, node, child_units):
        """
        Return (and test) the units corresponding to an expression node in the
        expression tree where all children should have the same units (e.g. sum).

        Parameters
        ----------
        node : Pyomo expression node
            The parent node of the children

        child_units : list
           This is a list of pint units (one for each of the children)

        Returns
        -------
        : pint unit
        """
        # TODO: This may be expensive for long summations and, in the
        # case of reporting only, we may want to skip the checks
        assert bool(child_units)

        # verify that the pint units are equivalent from each
        # of the child nodes - assume that PyomoUnits are equivalent
        pint_unit_0 = child_units[0]
        for pint_unit_i in child_units:
            if not self._equivalent_pint_units(pint_unit_0, pint_unit_i):
                raise InconsistentUnitsError(
                    pint_unit_0,
                    pint_unit_i,
                    'Error in units found in expression: %s' % (node,),
                )

        # checks were OK, return the first one in the list
        return pint_unit_0

    def _get_unit_for_product(self, node, child_units):
        """
        Return (and test) the units corresponding to a product expression node
        in the expression tree.

        Parameters
        ----------
        node : Pyomo expression node
            The parent node of the children

        child_units : list
           This is a list of pint units (one for each of the children)

        Returns
        -------
        : pint unit
        """
        assert len(child_units) == 2

        pint_unit = child_units[0] * child_units[1]

        if hasattr(pint_unit, 'units'):
            return pint_unit.units

        return pint_unit

    def _get_unit_for_division(self, node, child_units):
        """
        Return (and test) the units corresponding to a division expression node
        in the expression tree.

        Parameters
        ----------
        node : Pyomo expression node
            The parent node of the children

        child_units : list
           This is a list of pint units (one for each of the children)

        Returns
        -------
        : pint unit
        """
        assert len(child_units) == 2

        # this operation can create a quantity, but we want a pint unit object
        return child_units[0] / child_units[1]

    def _get_unit_for_pow(self, node, child_units):
        """
        Return (and test) the units corresponding to a pow expression node
        in the expression tree.

        Parameters
        ----------
        node : Pyomo expression node
            The parent node of the children

        child_units : list
           This is a list of pint units (one for each of the children)

        Returns
        -------
        : pint unit
        """
        assert len(child_units) == 2

        # the exponent needs to be dimensionless
        if not self._equivalent_to_dimensionless(child_units[1]):
            # todo: allow radians?
            raise UnitsError(
                f"Error in sub-expression: {node}. "
                "Exponents in a pow expression must be dimensionless."
            )

        # common case - exponent is a constant number
        exponent = node.args[1]
        if type(exponent) in nonpyomo_leaf_types:
            return child_units[0] ** value(exponent)

        # if base is dimensioness, exponent doesn't matter
        if self._equivalent_to_dimensionless(child_units[0]):
            return self._pint_dimensionless

        # base is not dimensionless, exponent is dimensionless
        # ensure that the exponent is fixed
        if not exponent.is_fixed():
            raise UnitsError(
                f"The base of an exponent has units {child_units[0]}, but "
                "the exponent is not a fixed numerical value."
            )

        return child_units[0] ** value(exponent)

    def _get_unit_for_single_child(self, node, child_units):
        """
        Return (and test) the units corresponding to a unary operation (e.g. negation)
        expression node in the expression tree.

        Parameters
        ----------
        node : Pyomo expression node
            The parent node of the children

        child_units : list
           This is a list of pint units (one for each of the children)

        Returns
        -------
        : pint unit
        """
        assert len(child_units) == 1
        return child_units[0]

    def _get_units_ExternalFunction(self, node, child_units):
        """
        Check to make sure that any child arguments are consistent with
        arg_units return the value from node.get_units() This
        was written for ExternalFunctionExpression where the external
        function has units assigned to its return value and arguments

        Parameters
        ----------
        node : Pyomo expression node
            The parent node of the children

        child_units : list
           This is a list of pint units (one for each of the children)

        Returns
        -------
        : pint unit
        """
        # get the list of arg_units
        arg_units = node.get_arg_units()
        dless = self._pint_dimensionless
        if arg_units is None:
            # they should all be dimensionless
            arg_units = [dless] * len(child_units)
        else:
            # copy arg_units so we don't overwrite the ones in the expression object
            arg_units = list(arg_units)
            for i, a in enumerate(arg_units):
                arg_units[i] = self._pyomo_units_container._get_pint_units(a)

        for arg_unit, pint_unit in zip(arg_units, child_units):
            assert arg_unit is not None
            if not self._equivalent_pint_units(arg_unit, pint_unit):
                raise InconsistentUnitsError(
                    arg_unit, pint_unit, 'Inconsistent units found in ExternalFunction.'
                )

        # now return the units in node.get_units
        return self._pyomo_units_container._get_pint_units(node.get_units())

    def _get_dimensionless_with_dimensionless_children(self, node, child_units):
        """
        Check to make sure that any child arguments are unitless /
        dimensionless (for functions like exp()) and return dimensionless.

        Parameters
        ----------
        node : Pyomo expression node
            The parent node of the children

        child_units : list
           This is a list of pint units (one for each of the children)

        Returns
        -------
        : pint unit
        """
        for pint_unit in child_units:
            if not self._equivalent_to_dimensionless(pint_unit):
                raise UnitsError(
                    f'Expected no units or dimensionless units in {node}, '
                    f'but found {pint_unit}.'
                )

        return self._pint_dimensionless

    def _get_dimensionless_no_children(self, node, child_units):
        """
        Check to make sure the length of child_units is zero, and returns
        dimensionless. Used for leaf nodes that should not have any units.

        Parameters
        ----------
        node : Pyomo expression node
            The parent node of the children

        child_units : list
           This is a list of pint units (one for each of the children)

        Returns
        -------
        : pint unit
        """
        assert len(child_units) == 0
        # may need more checks for dimensionless for other types
        assert type(node) is IndexTemplate
        return self._pint_dimensionless

    def _get_unit_for_unary_function(self, node, child_units):
        """
        Return (and test) the units corresponding to a unary function expression node
        in the expression tree. Checks that child_units is of length 1
        and calls the appropriate method from the unary function method map.

        Parameters
        ----------
        node : Pyomo expression node
            The parent node of the children

        child_units : list
           This is a list of pint units (one for each of the children)

        Returns
        -------
        : pint unit
        """
        assert len(child_units) == 1
        func_name = node.getname()
        node_func = self.unary_function_method_map.get(func_name, None)
        if node_func is None:
            raise TypeError(
                f'An unhandled unary function: {func_name} was encountered '
                f'while retrieving the units of expression {node}'
            )
        return node_func(self, node, child_units)

    def _get_unit_for_expr_if(self, node, child_units):
        """
        Return (and test) the units corresponding to an Expr_if expression node
        in the expression tree. The _if relational expression is validated and
        the _then/_else are checked to ensure they have the same units. Also checks
        to make sure length of child_units is 3

        Parameters
        ----------
        node : Pyomo expression node
            The parent node of the children

        child_units : list
           This is a list of pint units (one for each of the children)

        Returns
        -------
        : pint unit
        """
        assert len(child_units) == 3

        # the _if should already be consistent (since the children were
        # already checked)
        if not self._equivalent_pint_units(child_units[1], child_units[2]):
            raise InconsistentUnitsError(
                child_units[1],
                child_units[2],
                'Error in units found in expression: %s' % (node,),
            )

        return child_units[1]

    def _get_dimensionless_with_radians_child(self, node, child_units):
        """
        Return (and test) the units corresponding to a trig function expression node
        in the expression tree. Checks that the length of child_units is 1
        and that the units of that child expression are unitless or radians and
        returns dimensionless for the units.

        Parameters
        ----------
        node : Pyomo expression node
            The parent node of the children

        child_units : list
           This is a list of pint units (one for each of the children)

        Returns
        -------
        : pint unit
        """
        assert len(child_units) == 1

        if self._equivalent_to_dimensionless(child_units[0]):
            return self._pint_dimensionless
        if self._equivalent_pint_units(
            child_units[0], self._pyomo_units_container._pint_registry.radian
        ):
            return self._pint_dimensionless

        # units are not None, dimensionless, or radians
        raise UnitsError(
            'Expected radians or dimensionless in argument to function '
            'in expression %s, but found %s' % (node, child_units[0])
        )

    def _get_radians_with_dimensionless_child(self, node, child_units):
        """
        Return (and test) the units corresponding to an inverse trig expression node
        in the expression tree. Checks that the length of child_units is 1
        and that the child argument is dimensionless, and returns radians

        Parameters
        ----------
        node : Pyomo expression node
            The parent node of the children

        child_units : list
           This is a list of pint units (one for each of the children)

        Returns
        -------
        : pint unit
        """
        assert len(child_units) == 1

        if self._equivalent_to_dimensionless(child_units[0]):
            return self._pyomo_units_container._pint_registry.radian

        raise UnitsError(
            f'Expected dimensionless argument to function in expression {node},'
            f' but found {child_units[0]}'
        )

    def _get_unit_sqrt(self, node, child_units):
        """
        Return (and test) the units corresponding to a sqrt expression node
        in the expression tree. Checks that the length of child_units is one.

        Parameters
        ----------
        node : Pyomo expression node
            The parent node of the children

        child_units : list
           This is a list of pint units (one for each of the children)

        Returns
        -------
        : pint unit
        """
        assert len(child_units) == 1
        return child_units[0] ** 0.5

    node_type_method_map = {
        EXPR.EqualityExpression: _get_unit_for_equivalent_children,
        EXPR.InequalityExpression: _get_unit_for_equivalent_children,
        EXPR.RangedExpression: _get_unit_for_equivalent_children,
        EXPR.SumExpression: _get_unit_for_equivalent_children,
        EXPR.NPV_SumExpression: _get_unit_for_equivalent_children,
        EXPR.ProductExpression: _get_unit_for_product,
        EXPR.MonomialTermExpression: _get_unit_for_product,
        EXPR.NPV_ProductExpression: _get_unit_for_product,
        EXPR.DivisionExpression: _get_unit_for_division,
        EXPR.NPV_DivisionExpression: _get_unit_for_division,
        EXPR.PowExpression: _get_unit_for_pow,
        EXPR.NPV_PowExpression: _get_unit_for_pow,
        EXPR.NegationExpression: _get_unit_for_single_child,
        EXPR.NPV_NegationExpression: _get_unit_for_single_child,
        EXPR.AbsExpression: _get_unit_for_single_child,
        EXPR.NPV_AbsExpression: _get_unit_for_single_child,
        EXPR.UnaryFunctionExpression: _get_unit_for_unary_function,
        EXPR.NPV_UnaryFunctionExpression: _get_unit_for_unary_function,
        EXPR.Expr_ifExpression: _get_unit_for_expr_if,
        IndexTemplate: _get_dimensionless_no_children,
        EXPR.Numeric_GetItemExpression: (
            _get_dimensionless_with_dimensionless_children
        ),
        EXPR.NPV_Numeric_GetItemExpression: (
            _get_dimensionless_with_dimensionless_children
        ),
        EXPR.ExternalFunctionExpression: _get_units_ExternalFunction,
        EXPR.NPV_ExternalFunctionExpression: _get_units_ExternalFunction,
        EXPR.LinearExpression: _get_unit_for_equivalent_children,
    }

    unary_function_method_map = {
        'log': _get_dimensionless_with_dimensionless_children,
        'log10': _get_dimensionless_with_dimensionless_children,
        'sin': _get_dimensionless_with_radians_child,
        'cos': _get_dimensionless_with_radians_child,
        'tan': _get_dimensionless_with_radians_child,
        'sinh': _get_dimensionless_with_radians_child,
        'cosh': _get_dimensionless_with_radians_child,
        'tanh': _get_dimensionless_with_radians_child,
        'asin': _get_radians_with_dimensionless_child,
        'acos': _get_radians_with_dimensionless_child,
        'atan': _get_radians_with_dimensionless_child,
        'exp': _get_dimensionless_with_dimensionless_children,
        'sqrt': _get_unit_sqrt,
        'asinh': _get_radians_with_dimensionless_child,
        'acosh': _get_radians_with_dimensionless_child,
        'atanh': _get_radians_with_dimensionless_child,
        'ceil': _get_unit_for_single_child,
        'floor': _get_unit_for_single_child,
    }

    def initializeWalker(self, expr):
        # Refresh the cached dimensionless (in case the underlying pint
        # registry was either changed or had not been set when the
        # PyomoUnitsContainer was originally created).
        self._pint_dimensionless = self._pyomo_units_container._pint_dimensionless
        walk, result = self.beforeChild(None, expr, 0)
        if not walk:
            result = self.finalizeResult(result)
        return walk, result

    def beforeChild(self, node, child, child_idx):
        ctype = child.__class__
        if ctype in native_types or ctype in pyomo_constant_types:
            return False, self._pint_dimensionless

        if child.is_expression_type():
            return True, None

        # this is a leaf, but not a native type
        if ctype is _PyomoUnit:
            return False, child._get_pint_unit()
        elif hasattr(child, 'get_units'):
            # might want to add other common types here
            pyomo_unit = child.get_units()
            pint_unit = self._pyomo_units_container._get_pint_units(pyomo_unit)
            return False, pint_unit

        return True, None

    def exitNode(self, node, data):
        """Visitor callback when moving up the expression tree.

        Callback for
        :class:`pyomo.core.current.StreamBasedExpressionVisitor`. This
        method is called when moving back up the tree in a depth first
        search.

        """
        node_func = self.node_type_method_map.get(node.__class__, None)
        if node_func is not None:
            return node_func(self, node, data)

        # not a leaf - check if it is a named expression
        if (
            hasattr(node, 'is_named_expression_type')
            and node.is_named_expression_type()
        ):
            pint_unit = self._get_unit_for_single_child(node, data)
            return pint_unit

        raise TypeError(
            f'An unhandled expression node type: {type(node)} was encountered '
            f'while retrieving the units of expression {node}'
        )

    def finalizeResult(self, result):
        if hasattr(result, 'units'):
            # likely, we got a quantity object and not a units object
            return result.units
        return result


class PyomoUnitsContainer(object):
    """Class that is used to create and contain units in Pyomo.

    This is the class that is used to create, contain, and interact
    with units in Pyomo.  The module
    (:mod:`pyomo.core.base.units_container`) also contains a module
    level units container :py:data:`units` that is an instance of a
    PyomoUnitsContainer. This module instance should typically be used
    instead of creating your own instance of a
    :py:class:`PyomoUnitsContainer`.  For an overview of the usage of
    this class, see the module documentation
    (:mod:`pyomo.core.base.units_container`)

    This class is based on the "pint" module. Documentation for
    available units can be found at the following url:
    https://github.com/hgrecco/pint/blob/master/pint/default_en.txt

    .. note::

        Pre-defined units can be accessed through attributes on the
        PyomoUnitsContainer class; however, these attributes are created
        dynamically through the __getattr__ method, and are not present
        on the class until they are requested.

    """

    def __init__(self, pint_registry=NOTSET):
        """Create a PyomoUnitsContainer instance."""
        if pint_registry is NOTSET:
            pint_registry = pint_module.UnitRegistry()
        self._pint_registry = pint_registry
        if pint_registry is None:
            self._pint_dimensionless = None
        else:
            self._pint_dimensionless = self._pint_registry.dimensionless
        self._pintUnitExtractionVisitor = PintUnitExtractionVisitor(self)

    def load_definitions_from_file(self, definition_file):
        """Load new units definitions from a file

        This method loads additional units definitions from a user
        specified definition file. An example of a definitions file
        can be found at:
        https://github.com/hgrecco/pint/blob/master/pint/default_en.txt

        If we have a file called ``my_additional_units.txt`` with the
        following lines::

            USD = [currency]

        Then we can add this to the container with:

        .. doctest::
            :skipif: not pint_available
            :hide:

            # Get a local units object (to avoid duplicate registration
            # with the example in load_definitions_from_strings)
            >>> import pyomo.core.base.units_container as _units
            >>> u = _units.PyomoUnitsContainer()
            >>> with open('my_additional_units.txt', 'w') as FILE:
            ...     tmp = FILE.write("USD = [currency]\\n")

        .. doctest::
            :skipif: not pint_available

            >>> u.load_definitions_from_file('my_additional_units.txt')
            >>> print(u.USD)
            USD

        .. doctest::
            :skipif: not pint_available
            :hide:

            # Clean up the file we just created
            >>> import os
            >>> os.remove('my_additional_units.txt')

        """
        self._pint_registry.load_definitions(definition_file)
        self._pint_dimensionless = self._pint_registry.dimensionless

    def load_definitions_from_strings(self, definition_string_list):
        """Load new units definitions from a string

        This method loads additional units definitions from a list of
        strings (one for each line). An example of the definitions
        strings can be found at:
        https://github.com/hgrecco/pint/blob/master/pint/default_en.txt

        For example, to add the currency dimension and US dollars as a
        unit, use

        .. doctest::
            :skipif: not pint_available
            :hide:

            # get a local units object (to avoid duplicate registration
            # with the example in load_definitions_from_strings)
            >>> import pint
            >>> import pyomo.core.base.units_container as _units
            >>> u = _units.PyomoUnitsContainer()

        .. doctest::
            :skipif: not pint_available

            >>> u.load_definitions_from_strings(['USD = [currency]'])
            >>> print(u.USD)
            USD

        """
        self._pint_registry.load_definitions(definition_string_list)

    def __getattr__(self, item):
        """Here, __getattr__ is implemented to automatically create the
        necessary unit if the attribute does not already exist.

        Parameters
        ----------
        item : str
            the name of the new field requested external

        Returns
        -------
        PyomoUnit
            returns a PyomoUnit corresponding to the requested attribute,
            or None if it cannot be created.

        """
        # since __getattr__ was called, we must not have this field yet
        # try to build a unit from the requested item
        pint_registry = self._pint_registry
        try:
            pint_unit = getattr(pint_registry, item)

            if pint_unit is not None:
                # check if the unit is an offset unit and throw an exception if necessary
                # TODO: should we prevent delta versions: delta_degC and delta_degF as well?
                pint_unit_container = pint_module.util.to_units_container(
                    pint_unit, pint_registry
                )
                for u, e in pint_unit_container.items():
                    if not pint_registry._units[u].is_multiplicative:
                        raise UnitsError(
                            'Pyomo units system does not support the offset '
                            f'units "{item}". Use absolute units '
                            '(e.g. kelvin instead of degC) instead.'
                        )

                unit = _PyomoUnit(pint_unit, pint_registry)
                setattr(self, item, unit)
                return unit
        except pint_module.errors.UndefinedUnitError as exc:
            pint_unit = None

        if pint_unit is None:
            raise AttributeError(f'Attribute {item} not found.')

    # We added support to specify a units definition file instead of this programmatic interface
    # def create_new_base_dimension(self, dimension_name, base_unit_name):
    #     """
    #     Use this method to create a new base dimension (e.g. a new dimension other than Length, Mass) for the unit manager.
    #
    #     Parameters
    #     ----------
    #     dimension_name : str
    #        name of the new dimension (needs to be unique from other dimension names)
    #
    #     base_unit_name : str
    #        base_unit_name: name of the base unit for this dimension
    #
    #     """
    #     # TODO: Error checking - if dimension already exists then we should return a useful error message.
    #     defn_str = str(base_unit_name) + ' = [' + str(dimension_name) + ']'
    #     self._pint_registry.define(defn_str)
    #
    # def create_new_unit(self, unit_name, base_unit_name, conv_factor, conv_offset=None):
    #     """
    #     Create a new unit that is not already included in the units manager.
    #
    #     Examples:
    #         # create a new unit of length called football field that is 1000 yards
    #         # defines: x (in yards) = y (in football fields) X 100.0
    #         >>> um.create_new_unit('football_field', 'yards', 100.0)
    #
    #         # create a new unit of temperature that is half the size of a degree F
    #         # defines x (in K) = y (in half degF) X 10/9 + 255.3722222 K
    #         >>> um.create_new_unit('half_degF', 'kelvin', 10.0/9.0, 255.3722222)
    #
    #     Parameters
    #     ----------
    #     unit_name : str
    #        name of the new unit to create
    #     base_unit_name : str
    #        name of the base unit from the same "dimension" as the new unit
    #     conv_factor : float
    #        value of the multiplicative factor needed to convert the new unit
    #        to the base unit
    #     conv_offset : float
    #        value of any offset between the new unit and the base unit
    #        Note that the units of this offset are the same as the base unit,
    #        and it is applied after the factor conversion
    #        (e.g., base_value = new_value*conv_factor + conv_offset)
    #
    #     """
    #     if conv_offset is None:
    #         defn_str = '{0!s} = {1:g} * {2!s}'.format(unit_name, float(conv_factor), base_unit_name)
    #     else:
    #         defn_str = '{0!s} = {1:17.16g} * {2!s}; offset: {3:17.16g}'.format(unit_name, float(conv_factor), base_unit_name,
    #                                                                  float(conv_offset))
    #     self._pint_registry.define(defn_str)

    def _rel_diff(self, a, b):
        scale = min(abs(a), abs(b))
        if scale < 1.0:
            scale = 1.0
        return abs(a - b) / scale

    def _equivalent_pint_units(self, a, b, TOL=1e-12):
        if a is b or a == b:
            return True
        base_a = self._pint_registry.get_base_units(a)
        base_b = self._pint_registry.get_base_units(b)
        if base_a[1] != base_b[1]:
            uc_a = base_a[1].dimensionality
            uc_b = base_b[1].dimensionality
            for key in uc_a.keys() | uc_b.keys():
                if self._rel_diff(uc_a.get(key, 0), uc_b.get(key, 0)) >= TOL:
                    return False
        return self._rel_diff(base_a[0], base_b[0]) <= TOL

    def _equivalent_to_dimensionless(self, a, TOL=1e-12):
        if a is self._pint_dimensionless or a == self._pint_dimensionless:
            return True
        base_a = self._pint_registry.get_base_units(a)
        if not base_a[1].dimensionless:
            return False
        return self._rel_diff(base_a[0], 1.0) <= TOL

    def _get_pint_units(self, expr):
        """
        Return the pint units corresponding to the expression. This does
        a number of checks as well.

        Parameters
        ----------
        expr : Pyomo expression
           the input expression for extracting units

        Returns
        -------
        : pint unit
        """
        if expr is None:
            return self._pint_dimensionless
        return self._pintUnitExtractionVisitor.walk_expression(expr=expr)

    def get_units(self, expr):
        """Return the Pyomo units corresponding to this expression (also
        performs validation and will raise an exception if units are not
        consistent).

        Parameters
        ----------
        expr : Pyomo expression
            The expression containing the desired units

        Returns
        -------
        : Pyomo unit (expression)
           Returns the units corresponding to the expression

        Raises
        ------
        :py:class:`pyomo.core.base.units_container.UnitsError`, :py:class:`pyomo.core.base.units_container.InconsistentUnitsError`

        """
        return _PyomoUnit(self._get_pint_units(expr), self._pint_registry)

    def _pint_convert_temp_from_to(
        self, numerical_value, pint_from_units, pint_to_units
    ):
        if type(numerical_value) not in native_numeric_types:
            raise UnitsError(
                'Conversion routines for absolute and relative temperatures '
                'require a numerical value only. Pyomo objects (Var, Param, '
                'expressions) are not supported. Please use value(x) to '
                'extract the numerical value if necessary.'
            )

        src_quantity = self._pint_registry.Quantity(numerical_value, pint_from_units)
        dest_quantity = src_quantity.to(pint_to_units)
        return dest_quantity.magnitude

    def convert_temp_K_to_C(self, value_in_K):
        """
        Convert a value in Kelvin to degrees Celsius.  Note that this method
        converts a numerical value only. If you need temperature
        conversions in expressions, please work in absolute
        temperatures only.
        """
        return self._pint_convert_temp_from_to(
            value_in_K, self._pint_registry.K, self._pint_registry.degC
        )

    def convert_temp_C_to_K(self, value_in_C):
        """
        Convert a value in degrees Celsius to Kelvin Note that this
        method converts a numerical value only. If you need
        temperature conversions in expressions, please work in
        absolute temperatures only.
        """
        return self._pint_convert_temp_from_to(
            value_in_C, self._pint_registry.degC, self._pint_registry.K
        )

    def convert_temp_R_to_F(self, value_in_R):
        """
        Convert a value in Rankine to degrees Fahrenheit.  Note that
        this method converts a numerical value only. If you need
        temperature conversions in expressions, please work in
        absolute temperatures only.
        """
        return self._pint_convert_temp_from_to(
            value_in_R, self._pint_registry.rankine, self._pint_registry.degF
        )

    def convert_temp_F_to_R(self, value_in_F):
        """
        Convert a value in degrees Fahrenheit to Rankine.  Note that
        this method converts a numerical value only. If you need
        temperature conversions in expressions, please work in
        absolute temperatures only.
        """
        return self._pint_convert_temp_from_to(
            value_in_F, self._pint_registry.degF, self._pint_registry.rankine
        )

    def convert(self, src, to_units=None):
        """
        This method returns an expression that contains the
        explicit conversion from one unit to another.

        Parameters
        ----------
        src : Pyomo expression
           The source value that will be converted. This could be a
           Pyomo Var, Pyomo Param, or a more complex expression.
        to_units : Pyomo units expression
           The desired target units for the new expression

        Returns
        -------
           ret : Pyomo expression
        """
        src_pint_unit = self._get_pint_units(src)
        to_pint_unit = self._get_pint_units(to_units)
        if src_pint_unit == to_pint_unit:
            return src

        # We disallow offset units, so we only need a factor to convert
        # between the two
        src_base_factor, base_units_src = self._pint_registry.get_base_units(
            src_pint_unit, check_nonmult=True
        )
        to_base_factor, base_units_to = self._pint_registry.get_base_units(
            to_pint_unit, check_nonmult=True
        )

        if base_units_src != base_units_to:
            raise InconsistentUnitsError(
                src_pint_unit, to_pint_unit, 'Error in convert: units not compatible.'
            )

        return (
            (src_base_factor / to_base_factor)
            * _PyomoUnit(to_pint_unit / src_pint_unit, self._pint_registry)
            * src
        )

    def convert_value(self, num_value, from_units=None, to_units=None):
        """
        This method performs explicit conversion of a numerical value
        from one unit to another, and returns the new value.

        The argument "num_value" must be a native numeric type (e.g. float).
        Note that this method returns a numerical value only, and not an
        expression with units.

        Parameters
        ----------
        num_value : float or other native numeric type
           The value that will be converted
        from_units : Pyomo units expression
           The units to convert from
        to_units : Pyomo units expression
           The units to convert to

        Returns
        -------
           float : The converted value

        """
        if type(num_value) not in native_numeric_types:
            raise UnitsError(
                'The argument "num_value" in convert_value must be a native '
                'numeric type, but instead type {type(num_value)} was found.'
            )

        from_pint_unit = self._get_pint_units(from_units)
        to_pint_unit = self._get_pint_units(to_units)
        if from_pint_unit == to_pint_unit:
            return num_value

        # We disallow offset units, so we only need a factor to convert
        # between the two
        #
        # TODO: Do we need to disallow offset units here? Should we
        # assume the user knows what they are doing?
        #
        # TODO: This check may be overkill - pint will raise an error
        # that may be sufficient
        from_base_factor, from_base_units = self._pint_registry.get_base_units(
            from_pint_unit, check_nonmult=True
        )
        to_base_factor, to_base_units = self._pint_registry.get_base_units(
            to_pint_unit, check_nonmult=True
        )
        if from_base_units != to_base_units:
            raise UnitsError(
                'Cannot convert %s to %s. Units are not compatible.'
                % (from_units, to_units)
            )

        # convert the values
        from_quantity = num_value * from_pint_unit
        to_quantity = from_quantity.to(to_pint_unit)
        return to_quantity.magnitude

    def set_pint_registry(self, pint_registry):
        if pint_registry is self._pint_registry:
            return
        if self._pint_registry is not None:
            logger.warning(
                "Changing the pint registry used by the Pyomo Units "
                "system after the PyomoUnitsContainer was constructed.  "
                "Pint requires that all units and dimensioned quantities "
                "are generated by a single pint registry."
            )
        self._pint_registry = pint_registry
        self._pint_dimensionless = self._pint_registry.dimensionless

    @property
    def pint_registry(self):
        return self._pint_registry


class _QuantityVisitor(ExpressionValueVisitor):
    def __init__(self):
        self.native_types = set(nonpyomo_leaf_types)
        self.native_types.add(units._pint_registry.Quantity)
        self._unary_inverse_trig = {'asin', 'acos', 'atan', 'asinh', 'acosh', 'atanh'}

    def visit(self, node, values):
        """Visit nodes that have been expanded"""
        if node.__class__ in self.handlers:
            return self.handlers[node.__class__](self, node, values)
        return node._apply_operation(values)

    def visiting_potential_leaf(self, node):
        """
        Visiting a potential leaf.

        Return True if the node is not expanded.
        """
        if node.__class__ in self.native_types:
            return True, node

        if node.is_expression_type():
            return False, None

        if node.is_numeric_type():
            if hasattr(node, 'get_units'):
                unit = node.get_units()
                if unit is not None:
                    return True, value(node) * unit._pint_unit
                else:
                    return True, value(node)
            elif node.__class__ is _PyomoUnit:
                return True, node._pint_unit
            else:
                return True, value(node)
        elif node.is_logical_type():
            return True, value(node)
        else:
            return True, node

    def finalize(self, val):
        if val.__class__ is units._pint_registry.Quantity:
            return val
        elif val.__class__ is units._pint_registry.Unit:
            return 1.0 * val
        # else
        try:
            return val * units._pint_dimensionless
        except:
            return val

    def _handle_unary_function(self, node, values):
        ans = node._apply_operation(values)
        if node.getname() in self._unary_inverse_trig:
            ans = ans * units._pint_registry.radian
        return ans

    def _handle_external(self, node, values):
        # External functions are units-unaware
        ans = node._apply_operation(
            [
                val.magnitude if val.__class__ is units._pint_registry.Quantity else val
                for val in values
            ]
        )
        unit = node.get_units()
        if unit is not None:
            ans = ans * unit._pint_unit
        return ans


_QuantityVisitor.handlers = {
    EXPR.UnaryFunctionExpression: _QuantityVisitor._handle_unary_function,
    EXPR.NPV_UnaryFunctionExpression: _QuantityVisitor._handle_unary_function,
    EXPR.ExternalFunctionExpression: _QuantityVisitor._handle_external,
    EXPR.NPV_ExternalFunctionExpression: _QuantityVisitor._handle_external,
}


def as_quantity(expr):
    return _QuantityVisitor().dfs_postorder_stack(expr)


class _DeferredUnitsSingleton(PyomoUnitsContainer):
    """A class supporting deferred interrogation of pint_available.

    This class supports creating a module-level singleton, but deferring
    the interrogation of the pint_available flag until the first time
    the object is actually used.  If pint is available, this instance
    object is replaced by an actual PyomoUnitsContainer.  Otherwise this
    leverages the pint_module to raise an (informative)
    DeferredImportError exception.

    """

    def __init__(self):
        # do NOT call the base class __init__ so that the pint_module is
        # not accessed
        pass

    def __getattribute__(self, attr):
        # Note that this method will only be called ONCE: either pint is
        # present, at which point this instance __class__ will fall back
        # to PyomoUnitsContainer (where this method is not declared, OR
        # pint is not available and an ImportError will be raised.
        #
        # We need special case handling for __class__: gurobipy
        # interrogates things by looking at their __class__ during
        # python shutdown.  Unfortunately, interrogating this
        # singleton's __class__ evaluates `pint_available`, which - if
        # DASK is installed - imports dask.  Importing dask creates
        # threading objects.  Unfortunately, creating threading objects
        # during interpreter shutdown generates a RuntimeError.  So, our
        # solution is to special-case the resolution of __class__ here
        # to avoid accidentally triggering the imports.
        if attr == "__class__":
            return _DeferredUnitsSingleton
        #
        if pint_available:
            # If the first thing that is being called is
            # "units.set_pint_registry(...)", then we will call __init__
            # with None so that the subsequent call to set_pint_registry
            # will work cleanly.  In all other cases, we will initialize
            # PyomoUnitsContainer with a new (default) pint registry.
            if attr == 'set_pint_registry':
                pint_registry = None
            else:
                pint_registry = pint_module.UnitRegistry()
            self.__class__ = PyomoUnitsContainer
            self.__init__(pint_registry)
            return getattr(self, attr)
        else:
            # Generate the ImportError
            return getattr(pint_module, attr)


# Define a module level instance of a PyomoUnitsContainer to use for
# all units within a Pyomo model. If pint is not available, this will
# cause an error at the first usage See module level documentation for
# an example.
units = _DeferredUnitsSingleton()
