#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#
#

"""Pyomo Units Container Module

.. warning:: This module is in beta and is not yet complete.

This module provides support for including units within Pyomo expressions, and provides
methods for checking the consistency of units within those expresions.

To use this package within your Pyomo model, you first need an instance of a PyomoUnitsContainer.
You can use the module level instance called `units` and use the pre-defined units in expressions or
components.

Examples:
    To use a unit within an expression, simply reference the desired unit as an attribute on the
    module singleton `units`.

    .. doctest::

       >>> from pyomo.environ import ConcreteModel, Var, Objective, units # import components and 'units' instance
       >>> model = ConcreteModel()
       >>> model.acc = Var()
       >>> model.obj = Objective(expr=(model.acc*units.m/units.s**2 - 9.81*units.m/units.s**2)**2)
       >>> print(units.get_units(model.obj.expr))
       m ** 2 / s ** 4

.. note:: This module has a module level instance of a PyomoUnitsContainer called `units` that you
         should use for creating, retreiving, and checking units

.. note:: This is a work in progress. Once the components units implementations are complete, the units will eventually
          work similar to the following.

          .. code-block:: python

             from pyomo.environ import ConcreteModel, Var, Objective, units
             model = ConcreteModel()
             model.x = Var(units=units.kg/units.m)
             model.obj = Objective(expr=(model.x - 97.2*units.kg/units.m)**2)

Notes:
    * The implementation is currently based on the `pint <http://pint.readthedocs.io>`_
      package and supports all the units that are supported by pint.
    * The list of units that are supported by pint can be found at
      the following url: https://github.com/hgrecco/pint/blob/master/pint/default_en.txt
    * Currently, we do NOT test units of unary functions that include native data types
      e.g. explicit float (3.0) since these are removed by the expression system
      before getting to the code that checks the units.

.. note:: In this implementation of units, "offset" units for temperature are not supported within
          expressions (i.e. the non-absolute temperature units including degrees C and degrees F).
          This is because there are many non-obvious combinations that are not allowable. This
          concern becomes clear if you first convert the non-absolute temperature units to absolute
          and then perform the operation. For example, if you write 30 degC + 30 degC == 60 degC,
          but convert each entry to Kelvin, the expression is not true (i.e., 303.15 K + 303.15 K
          is not equal to 333.15 K). Therefore, there are several operations that are not allowable
          with non-absolute units, including addition, multiplication, and division.

          Please see the pint documentation `here <https://pint.readthedocs.io/en/0.9/nonmult.html>`_
          for more discussion. While pint implements "delta" units (e.g., delta_degC) to support correct
          unit conversions, it can be difficult to identify and guarantee valid operations in a general
          algebraic modeling environment. While future work may support units with relative scale, the current
          implementation requires use of absolute temperature units (i.e. K and R) within expressions and
          a direct conversion of numeric values using specific functions for converting input data and reporting.

"""
# TODO
#    * implement specific functions for converting numeric values of absolute temperatures
#    * implement convert functionality
#    * create a new pint unit definition file (and load from that file)
#      since the precision in pint seems insufficient for 1e-8 constraint tolerances
#    * clean up use of unit and units in the naming
#    * implement and test pickling and un-pickling
#    * implement ignore_unit(x, expected_unit) that returns a dimensionless version of the expression
#      (Note that this may need to be a special expression object that may appear in the tree)
#    * Add units capabilities to Var and Param
#    * Investigate issues surrounding absolute and relative temperatures (delta units)
#    * Implement external function interface that specifies units for the arguments and the function itself


from pyomo.core.expr.numvalue import NumericValue, nonpyomo_leaf_types, value
from pyomo.core.base.template_expr import IndexTemplate
from pyomo.core.expr import current as expr
import six
try:
    import pint as pint_module
except ImportError:
    pint_module = None

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

    E.g., x == y, where x is in units of units.kg and y is in units of units.meter
    """
    def __init__(self, exp1, exp2, msg):
        msg = '{}: {} not compatible with {}.'.format(str(msg), str(exp1), str(exp2))
        super(InconsistentUnitsError, self).__init__(msg)


class _PyomoUnit(NumericValue):
    """An object that represents a single unit in Pyomo (e.g., kg, meter)

    Users should not create instances of _PyomoUnit directly, but rather access
    units as attributes on an instance of a :class:`PyomoUnitsContainer`.
    This module contains a global PyomoUnitContainer :py:data:`units`.
    See module documentation for more information.
    """
    def __init__(self, pint_unit, pint_registry):
        super(_PyomoUnit, self).__init__()
        assert pint_unit is not None
        assert pint_registry is not None
        self._pint_unit = pint_unit
        self._pint_registry = pint_registry

    def _get_pint_unit(self):
        """ Return the pint unit corresponding to this Pyomo unit. """
        return self._pint_unit

    def _get_pint_registry(self):
        """ Return the pint registry (pint.UnitRegistry) object used to create this unit. """
        return self._pint_registry

    # Todo: test pickle and implement __getstate__/__setstate__ to do the right thing

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
        """ This is not a parameter type (overloaded from NumericValue) """
        return False

    def is_variable_type(self):
        """ This is not a variable type (overloaded from NumericValue) """
        return False

    def is_potentially_variable(self):
        """
        This is not potentially variable (does not and cannot contain a variable).
        Overloaded from NumericValue
        """
        return False

    def is_named_expression_type(self):
        """ This is not a named expression (overloaded from NumericValue) """
        return False

    def is_expression_type(self):
        """ This is a leaf, not an expression (overloaded from NumericValue) """
        return False

    def is_component_type(self):
        """ This is not a component type (overloaded from NumericValue) """
        return False

    def is_relational(self):
        """ This is not relational (overloaded from NumericValue) """
        return False

    def is_indexed(self):
        """ This is not indexed (overloaded from NumericValue) """
        return False

    def _compute_polynomial_degree(self, result):
        """ Returns the polynomial degree - since units are constants, they have degree of zero.
        Note that :py:meth:`NumericValue.polynomial_degree` calls this method.
        """
        return 0

    def __float__(self):
        """
        Coerce the value to a floating point

        Raises:
            TypeError
        """
        raise TypeError(
            "Implicit conversion of Pyomo Unit `%s' to a float is "
            "disabled. This error is often the result of treating a unit "
            "as though it were a number (e.g., passing a unit to a built-in "
            "math function). Avoid this error by using Pyomo-provided math "
            "functions."
            % self.name)

    def __int__(self):
        """
        Coerce the value to an integer

        Raises:
            TypeError
        """
        raise TypeError(
            "Implicit conversion of Pyomo Unit `%s' to an int is "
            "disabled. This error is often the result of treating a unit "
            "as though it were a number (e.g., passing a unit to a built-in "
            "math function). Avoid this error by using Pyomo-provided math "
            "math function). Avoid this error by using Pyomo-provided math "
            "functions."
            % self.name)

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
        """ Returns a string representing the unit """

        # The ~ returns the short form of the pint unit if the unit is
        # an instance of the unit 'dimensionless', then pint returns ''
        # which causes problems with some string processing in Pyomo
        # that expects a name
        #
        # Note: Some pint units contain unicode characters (notably
        # delta temperatures).  So that things work cleanly in Python 2
        # and 3, we will generate the string as unicode, then explicitly
        # encode it to UTF-8 in Python 2
        retstr = u'{:!~s}'.format(self._pint_unit)
        if retstr == '':
            retstr = 'dimensionless'
        if six.PY2:
            return str(retstr.encode('utf8'))
        else:
            return retstr

    def to_string(self, verbose=None, labeler=None, smap=None,
                  compute_values=False):
        """
        Return a string representation of the expression tree.

        See documentation on :py:class:`NumericValue`

        Returns
        -------
        : bool
           A string representation for the expression tree.
        """
        return str(self)

    def __nonzero__(self):
        """Unit is treated as a constant value of 1.0. Therefore, it is always nonzero
        Returns
        -------
        : bool
           Returns whether on not the object is non-zero
        """
        return self.__bool__()

    def __bool__(self):
        """Unit is treated as a constant value of 1.0. Therefore, it is always "True"

        Returns
        -------
        : bool
           Returns whether or not the object is "empty"
        """
        return True

    def __call__(self, exception=True):
        """Unit is treated as a constant value, and this method always returns 1.0

        Returns
        -------
        : float
           Returns 1.0
        """
        return 1.0

    def pprint(self, ostream=None, verbose=False):
        """Display a user readable string description of this object.
        """
        if ostream is None: #pragma:nocover
            ostream = sys.stdout
        ostream.write(str(self))
        # There is also a long form, but the verbose flag is not really the correct indicator
        # if verbose:
        #     ostream.write('{:!s}'.format(self._pint_unit))
        # else:
        #     ostream.write('{:!~s}'.format(self._pint_unit))


class _UnitExtractionVisitor(expr.StreamBasedExpressionVisitor):
    def __init__(self, pyomo_units_container, units_equivalence_tolerance=1e-12):
        """
        Visitor class used to determine units of an expression. Do not use
        this class directly, but rather use :func:`get_units` or
        :func:`check_units_consistency`.

        Parameters
        ----------
        pyomo_units_container : PyomoUnitsContainer
           Instance of the PyomoUnitsContainer that was used for the units
           in the expressions. Pyomo does not support "mixing" units from
           different containers

        units_equivalence_tolerance : float (default 1e-12)
            Floating point tolerance used when deciding if units are equivalent
            or not. (It can happen that units

        Notes
        -----
        This class inherits from the :class:`StreamBasedExpressionVisitor` to implement
        a walker that returns the pyomo units and pint units corresponding to an
        expression.

        There are class attributes (dicts) that map the expression node type to the
        particular method that should be called to return the units of the node based
        on the units of its child arguments. This map is used in exitNode.
        """
        super(_UnitExtractionVisitor, self).__init__()
        self._pyomo_units_container = pyomo_units_container
        self._pint_registry = self._pyomo_units_container._pint_registry
        self._units_equivalence_tolerance = units_equivalence_tolerance

    def _pint_unit_equivalent_to_dimensionless(self, pint_unit):
        """
        Check if a pint unit is equivalent to 'dimensionless' (this is true if it
        is either None or 'dimensionless' (or even radians)

        Parameters
        ----------
        pint_unit : pint unit
           The pint unit you want to check

        Returns
        -------
        : bool
           Returns True if pint_unit is equivalent to dimensionless units.

        """
        if pint_unit is None:
            return True
        return self._pint_units_equivalent(pint_unit, self._pint_registry.dimensionless)

    def _pint_units_equivalent(self, lhs, rhs):
        """
        Check if two pint units are equivalent

        Parameters
        ----------
        lhs : pint unit
            first pint unit to compare
        rhs : pint unit
            second pint unit to compare

        Returns
        -------
        : bool
           True if they are equivalent, and False otherwise
        """
        if lhs == rhs:
            # units are the same objects (or both None)
            return True
        elif lhs is None:
            # lhs is None, but rhs is not
            # check if rhs is equivalent to dimensionless (e.g. dimensionless or radians)
            return self._pint_unit_equivalent_to_dimensionless(rhs)
        elif rhs is None:
            # rhs is None, but lhs is not
            # check if lhs is equivalent to dimensionless (e.g. dimensionless or radians)
            return self._pint_unit_equivalent_to_dimensionless(lhs)

        # Units are not the same objects, and they are both not None
        # Now, use pint mechanisms to check by converting to Quantity objects
        lhsq = (1.0 * lhs).to_base_units()
        rhsq = (1.0 * rhs).to_base_units()

        if lhsq.dimensionality == rhsq.dimensionality and \
               abs(lhsq.magnitude/rhsq.magnitude - 1.0) < self._units_equivalence_tolerance:
            return True

        return False

    def _get_unit_for_equivalent_children(self, node, list_of_unit_tuples):
        """
        Return (and test) the units corresponding to an expression node in the
        expression tree where all children should have the same units (e.g. sum).

        Parameters
        ----------
        node : Pyomo expression node
            The parent node of the children

        list_of_unit_tuples : list
           This is a list of tuples (one for each of the children) where each tuple
           is a PyomoUnit, pint unit pair

        Returns
        -------
        : tuple (PyomoUnit, pint unit)
        """
        # TODO: This may be expensive for long summations and, in the case of reporting only, we may want to skip the checks
        assert len(list_of_unit_tuples) > 0

        # verify that the pint units are equivalent from each
        # of the child nodes - assume that PyomoUnits are equivalent
        pint_unit_0 = list_of_unit_tuples[0][1]
        for i in range(1, len(list_of_unit_tuples)):
            pint_unit_i = list_of_unit_tuples[i][1]
            if not self._pint_units_equivalent(pint_unit_0, pint_unit_i):
                raise InconsistentUnitsError(pint_unit_0, pint_unit_i,
                        'Error in units found in expression: {}'.format(str(node)))

        # checks were OK, return the first one in the list
        return (list_of_unit_tuples[0][0], list_of_unit_tuples[0][1])

    def _get_unit_for_linear_expression(self, node, list_of_unit_tuples):
        """
        Return (and test) the units corresponding to a :py:class:`LinearExpression` node
        in the expression tree.

        This is a special node since it does not use "args" the way
        other expression types do. Because of this, the StreamBasedExpressionVisitor
        does not pick up on the "children", and list_of_unit_tuples is empty.
        Therefore, we implement the recursion into coeffs and vars ourselves.

        Parameters
        ----------
        node : Pyomo expression node
            The parent node of the children

        list_of_unit_tuples : list
           This is a list of tuples (one for each of the children) where each tuple
           is a PyomoUnit, pint unit pair (Note: this is different for LinearExpression,
           - see method documentation above).

        Returns
        -------
        : tuple (PyomoUnit, pint unit)
        """
        # StreamBasedExpressionVisitor does not handle the children of this node
        assert len(list_of_unit_tuples) == 0

        # TODO: This may be expensive for long summations and, in the case of reporting only, we may want to skip the checks
        term_unit_list = []
        if node.constant:
            # we have a non-zero constant term, get its units
            const_units = self._pyomo_units_container.get_units(node.constant)
            term_unit_list.append(const_units)

        # go through the coefficients and variables
        assert len(node.linear_coefs) == len(node.linear_vars)
        for k,v in enumerate(node.linear_vars):
            c = node.linear_coefs[k]
            v_units = self._pyomo_units_container._get_units_tuple(v)
            c_units = self._pyomo_units_container._get_units_tuple(c)
            if c_units[1] is None and v_units[1] is None:
                term_unit_list.append((None,None))
            elif c_units[1] is None:
                # v_units[1] is not None
                term_unit_list.append(v_units)
            elif v_units[1] is None:
                # c_units[1] is not None
                term_unit_list.append(c_units)
            else:
                # both are not none
                term_unit_list.append((c_units[0]*v_units[0], c_units[1]*v_units[1]))

        assert len(term_unit_list) > 0

        # collected the units for all the terms, so now
        # verify that the pint units are equivalent from each
        # of the child nodes - assume that PyomoUnits are equivalent
        pint_unit_0 = term_unit_list[0][1]
        for i in range(1, len(term_unit_list)):
            pint_unit_i = term_unit_list[i][1]
            if not self._pint_units_equivalent(pint_unit_0, pint_unit_i):
                raise InconsistentUnitsError(pint_unit_0, pint_unit_i,
                        'Error in units found in expression: {}'.format(str(node)))

        # checks were OK, return the first one in the list
        return (term_unit_list[0][0], term_unit_list[0][1])

    def _get_unit_for_product(self, node, list_of_unit_tuples):
        """
        Return (and test) the units corresponding to a product expression node
        in the expression tree.

        Parameters
        ----------
        node : Pyomo expression node
            The parent node of the children

        list_of_unit_tuples : list
           This is a list of tuples (one for each of the children) where each tuple
           is a PyomoUnit, pint unit pair

        Returns
        -------
        : tuple (PyomoUnit, pint unit)
        """
        assert len(list_of_unit_tuples) > 1

        pyomo_unit = None
        pint_unit = None
        for i in range(len(list_of_unit_tuples)):
            pyomo_unit_i = list_of_unit_tuples[i][0]
            pint_unit_i = list_of_unit_tuples[i][1]
            if pint_unit_i is not None:
                assert pyomo_unit_i is not None
                if pint_unit is None:
                    assert pyomo_unit is None
                    pint_unit = pint_unit_i
                    pyomo_unit = pyomo_unit_i
                else:
                    pyomo_unit = pyomo_unit * pyomo_unit_i
                    pint_unit = pint_unit * pint_unit_i

        return (pyomo_unit, pint_unit)

    def _get_unit_for_division(self, node, list_of_unit_tuples):
        """
        Return (and test) the units corresponding to a division expression node
        in the expression tree.

        Parameters
        ----------
        node : Pyomo expression node
            The parent node of the children

        list_of_unit_tuples : list
           This is a list of tuples (one for each of the children) where each tuple
           is a PyomoUnit, pint unit pair

        Returns
        -------
        : tuple (PyomoUnit, pint unit)
        """
        assert len(list_of_unit_tuples) == 2

        num_pyomo_unit, num_pint_unit = list_of_unit_tuples[0]
        den_pyomo_unit, den_pint_unit = list_of_unit_tuples[1]
        if den_pint_unit is None:
            assert den_pyomo_unit is None
            return (num_pyomo_unit, num_pint_unit)
        # CDL using **(-1) since this returns a pint Unit and not a Quantity,
        # CDL but we could also return the units from the Quantity
        # CDL return (1.0/pyomo_unit, 1.0/pint_unit)
        if num_pint_unit is None:
            assert num_pyomo_unit is None
            return (1.0/den_pyomo_unit, den_pint_unit**(-1))
        else:
            return (num_pyomo_unit/den_pyomo_unit, num_pint_unit/den_pint_unit)

    def _get_unit_for_reciprocal(self, node, list_of_unit_tuples):
        """
        Return (and test) the units corresponding to a reciprocal expression node
        in the expression tree.

        Parameters
        ----------
        node : Pyomo expression node
            The parent node of the children

        list_of_unit_tuples : list
           This is a list of tuples (one for each of the children) where each tuple
           is a PyomoUnit, pint unit pair

        Returns
        -------
        : tuple (PyomoUnit, pint unit)
        """
        assert len(list_of_unit_tuples) == 1

        pyomo_unit, pint_unit = list_of_unit_tuples[0]
        if pint_unit is None:
            assert pyomo_unit is None
            return (None, None)
        # CDL using **(-1) since this returns a pint Unit and not a Quantity,
        # CDL but we could also return the units from the Quantity
        # CDL return (1.0/pyomo_unit, 1.0/pint_unit)
        return (1.0/pyomo_unit, pint_unit**(-1))

    def _get_unit_for_pow(self, node, list_of_unit_tuples):
        """
        Return (and test) the units corresponding to a pow expression node
        in the expression tree.

        Parameters
        ----------
        node : Pyomo expression node
            The parent node of the children

        list_of_unit_tuples : list
           This is a list of tuples (one for each of the children) where each tuple
           is a PyomoUnit, pint unit pair

        Returns
        -------
        : tuple (PyomoUnit, pint unit)
        """
        assert len(list_of_unit_tuples) == 2

        # this operation is x^y
        # x should be in list_of_unit_tuples[0] and
        # y should be in list_of_unit_tuples[1]
        pyomo_unit_base = list_of_unit_tuples[0][0]
        pint_unit_base = list_of_unit_tuples[0][1]
        pyomo_unit_exponent = list_of_unit_tuples[1][0]
        pint_unit_exponent = list_of_unit_tuples[1][1]

        # check to make sure that the exponent is unitless or dimensionless
        if pint_unit_exponent is not None and \
            not self._pint_unit_equivalent_to_dimensionless(pint_unit_exponent):
            # unit is not unitless, dimensionless, or equivalent to dimensionless (e.g. radians)
            raise UnitsError("Error in sub-expression: {}. "
                             "Exponents in a pow expression must be dimensionless."
                             "".format(node))

        # if the base is unitless or dimensionless, it does not matter if the exponent is fixed
        if pint_unit_base is None or \
            self._pint_unit_equivalent_to_dimensionless(pint_unit_base):
            return (pyomo_unit_base, pyomo_unit_exponent)

        # the base is NOT (None, dimensionless, or equivalent to dimensionless)
        # need to make sure that the exponent is a fixed number
        exponent = node.args[1]
        if type(exponent) not in nonpyomo_leaf_types \
            and not (exponent.is_fixed() or exponent.is_constant()):
            raise UnitsError("The base of an exponent has units {}, but "
                             "the exponent is not a fixed numerical value."
                             "".format(str(list_of_unit_tuples[0][0])))

        # base has units and exponent is fixed, return the appropriate unit
        exponent_value = value(exponent)
        pyomo_unit = pyomo_unit_base**exponent_value
        pint_unit = pint_unit_base**exponent_value

        return (pyomo_unit, pint_unit)

    def _get_unit_for_single_child(self, node, list_of_unit_tuples):
        """
        Return (and test) the units corresponding to a unary operation (e.g. negation)
        expression node in the expression tree.

        Parameters
        ----------
        node : Pyomo expression node
            The parent node of the children

        list_of_unit_tuples : list
           This is a list of tuples (one for each of the children) where each tuple
           is a PyomoUnit, pint unit pair

        Returns
        -------
        : tuple (PyomoUnit, pint unit)
        """
        assert len(list_of_unit_tuples) == 1

        pyomo_unit = list_of_unit_tuples[0][0]
        pint_unit = list_of_unit_tuples[0][1]
        return (pyomo_unit, pint_unit)

    def _get_dimensionless_with_dimensionless_children(self, node, list_of_unit_tuples):
        """
        Check to make sure that any child arguments are unitless / dimensionless (for functions like exp())
        and return (None, None) if successful. Although odd that this does not just return
        a boolean, it is done this way to match the signature of the other methods used to get
        units for expressions.

        Parameters
        ----------
        node : Pyomo expression node
            The parent node of the children

        list_of_unit_tuples : list
           This is a list of tuples (one for each of the children) where each tuple
           is a PyomoUnit, pint unit pair

        Returns
        -------
        : tuple (None, None)
        """
        for (pyomo_unit, pint_unit) in list_of_unit_tuples:
            if not self._pint_unit_equivalent_to_dimensionless(pint_unit):
                raise UnitsError('Expected no units or dimensionless units in {}, but found {}.'.format(str(node), str(pyomo_unit)))

        # if we make it here, then all are equal to None
        return (None, None)

    def _get_dimensionless_no_children(self, node, list_of_unit_tuples):
        """
        Check to make sure the length of list_of_unit_tuples is zero, and returns
        (None, None) to indicate unitless. Used for leaf nodes that should not have any units.

        Parameters
        ----------
        node : Pyomo expression node
            The parent node of the children

        list_of_unit_tuples : list
           This is a list of tuples (one for each of the children) where each tuple
           is a PyomoUnit, pint unit pair

        Returns
        -------
        : tuple (None, None)
        """
        assert len(list_of_unit_tuples) == 0
        assert type(node) != _PyomoUnit

        # # check that the leaf does not have any units
        # # TODO: Leave this commented code here since this "might" be similar to the planned mechanism for getting units from Pyomo component leaves
        # if hasattr(node, 'get_units') and node.get_units() is not None:
        #     raise UnitsError('Expected dimensionless units in {}, but found {}.'.format(str(node),
        #                         str(node.get_units())))

        return (None, None)

    def _get_unit_for_unary_function(self, node, list_of_unit_tuples):
        """
        Return (and test) the units corresponding to a unary function expression node
        in the expression tree. Checks that the list_of_unit_tuples is of length 1
        and calls the appropriate method from the unary function method map.

        Parameters
        ----------
        node : Pyomo expression node
            The parent node of the children

        list_of_unit_tuples : list
           This is a list of tuples (one for each of the children) where each tuple
           is a PyomoUnit, pint unit pair

        Returns
        -------
        : tuple (PyomoUnit, pint unit)
        """
        assert len(list_of_unit_tuples) == 1
        func_name = node.getname()
        if func_name in self.unary_function_method_map:
            node_func = self.unary_function_method_map[func_name]
            if node_func is not None:
                return node_func(self, node, list_of_unit_tuples)

        raise TypeError('An unhandled unary function: {} was encountered while retrieving the'
                        ' units of expression {}'.format(func_name, str(node)))

    def _get_unit_for_expr_if(self, node, list_of_unit_tuples):
        """
        Return (and test) the units corresponding to an Expr_if expression node
        in the expression tree. The _if relational expression is validated and
        the _then/_else are checked to ensure they have the same units. Also checks
        to make sure length of list_of_unit_tuples is 3

        Parameters
        ----------
        node : Pyomo expression node
            The parent node of the children

        list_of_unit_tuples : list
           This is a list of tuples (one for each of the children) where each tuple
           is a PyomoUnit, pint unit pair

        Returns
        -------
        : tuple (PyomoUnit, pint unit)
        """
        assert len(list_of_unit_tuples) == 3

        # the _if should already be consistent (since the children were
        # already checked)

        # verify that they _then and _else are equivalent
        then_pyomo_unit, then_pint_unit = list_of_unit_tuples[1]
        else_pyomo_unit, else_pint_unit = list_of_unit_tuples[2]

        if not self._pint_units_equivalent(then_pint_unit, else_pint_unit):
            raise InconsistentUnitsError(then_pyomo_unit, else_pyomo_unit,
                    'Error in units found in expression: {}'.format(str(node)))

        # then and else are the same
        return (then_pyomo_unit, then_pint_unit)

    def _get_dimensionless_with_radians_child(self, node, list_of_unit_tuples):
        """
        Return (and test) the units corresponding to a trig function expression node
        in the expression tree. Checks that the length of list_of_unit_tuples is 1
        and that the units of that child expression are unitless or radians and
        returns (None, None) for the units.

        Parameters
        ----------
        node : Pyomo expression node
            The parent node of the children

        list_of_unit_tuples : list
           This is a list of tuples (one for each of the children) where each tuple
           is a PyomoUnit, pint unit pair

        Returns
        -------
        : tuple (None, None)
        """
        assert len(list_of_unit_tuples) == 1

        pyomo_unit = list_of_unit_tuples[0][0]
        pint_unit = list_of_unit_tuples[0][1]
        if pint_unit is None:
            assert pyomo_unit is None
            # unitless, all is OK
            return (None, None)
        else:
            # pint unit is not None, now compare against radians / dimensionless
            if self._pint_units_equivalent(pint_unit, self._pint_registry.radians):
                return (None, None)

        # units are not None, dimensionless, or radians
        raise UnitsError('Expected radians in argument to function in expression {}, but found {}'.format(
            str(node), str(pyomo_unit)))

    def _get_radians_with_dimensionless_child(self, node, list_of_unit_tuples):
        """
        Return (and test) the units corresponding to an inverse trig expression node
        in the expression tree. Checks that the length of list_of_unit_tuples is 1
        and that the child argument is dimensionless, and returns radians

        Parameters
        ----------
        node : Pyomo expression node
            The parent node of the children

        list_of_unit_tuples : list
           This is a list of tuples (one for each of the children) where each tuple
           is a PyomoUnit, pint unit pair

        Returns
        -------
        : tuple (PyomoUnit for radians, pint_unit for radians)
        """
        assert len(list_of_unit_tuples) == 1

        pyomo_unit = list_of_unit_tuples[0][0]
        pint_unit = list_of_unit_tuples[0][1]
        if not self._pint_unit_equivalent_to_dimensionless(pint_unit):
            raise UnitsError('Expected dimensionless argument to function in expression {},'
                             ' but found {}'.format(
                             str(node), str(pyomo_unit)))

        uc = self._pyomo_units_container
        return (uc.radians, self._pint_registry.radians)

    def _get_unit_sqrt(self, node, list_of_unit_tuples):
        """
        Return (and test) the units corresponding to a sqrt expression node
        in the expression tree. Checks that the length of list_of_unit_tuples is one.

        Parameters
        ----------
        node : Pyomo expression node
            The parent node of the children

        list_of_unit_tuples : list
           This is a list of tuples (one for each of the children) where each tuple
           is a PyomoUnit, pint unit pair

        Returns
        -------
        : tuple (PyomoUnit, pint unit)
        """
        assert len(list_of_unit_tuples) == 1
        pint_unit = list_of_unit_tuples[0][1]
        if pint_unit is None:
            return (None, None)

        # units are not None
        return (list_of_unit_tuples[0][0]**0.5, list_of_unit_tuples[0][1]**0.5)

    node_type_method_map = {
        expr.EqualityExpression: _get_unit_for_equivalent_children,
        expr.InequalityExpression: _get_unit_for_equivalent_children,
        expr.RangedExpression: _get_unit_for_equivalent_children,
        expr.SumExpression: _get_unit_for_equivalent_children,
        expr.NPV_SumExpression: _get_unit_for_equivalent_children,
        expr.ProductExpression: _get_unit_for_product,
        expr.MonomialTermExpression: _get_unit_for_product,
        expr.NPV_ProductExpression: _get_unit_for_product,
        expr.DivisionExpression: _get_unit_for_division,
        expr.NPV_DivisionExpression: _get_unit_for_division,
        expr.ReciprocalExpression: _get_unit_for_reciprocal,
        expr.NPV_ReciprocalExpression: _get_unit_for_reciprocal,
        expr.PowExpression: _get_unit_for_pow,
        expr.NPV_PowExpression: _get_unit_for_pow,
        expr.NegationExpression: _get_unit_for_single_child,
        expr.NPV_NegationExpression: _get_unit_for_single_child,
        expr.AbsExpression: _get_unit_for_single_child,
        expr.NPV_AbsExpression: _get_unit_for_single_child,
        expr.UnaryFunctionExpression: _get_unit_for_unary_function,
        expr.NPV_UnaryFunctionExpression: _get_unit_for_unary_function,
        expr.Expr_ifExpression: _get_unit_for_expr_if,
        IndexTemplate: _get_dimensionless_no_children,
        expr.GetItemExpression: _get_dimensionless_with_dimensionless_children,
        expr.ExternalFunctionExpression: _get_dimensionless_with_dimensionless_children,
        expr.NPV_ExternalFunctionExpression: _get_dimensionless_with_dimensionless_children,
        expr.LinearExpression: _get_unit_for_linear_expression
    }

    unary_function_method_map = {
        'log': _get_dimensionless_with_dimensionless_children,
        'log10':_get_dimensionless_with_dimensionless_children,
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
        'floor': _get_unit_for_single_child
    }

    def exitNode(self, node, data):
        """ Callback for :class:`pyomo.core.current.StreamBasedExpressionVisitor`. This
        method is called when moving back up the tree in a depth first search."""
        
        # first check if the node is a leaf
        if type(node) in nonpyomo_leaf_types \
                or not node.is_expression_type():
            if isinstance(node, _PyomoUnit):
                return (node, node._get_pint_unit())

            # TODO: Check for Var or Param and return their units...
            # I have a leaf, but this is not a PyomoUnit - (treat as dimensionless)
            return (None, None)

        # not a leaf - get the appropriate function for type of the node
        node_func = self.node_type_method_map.get(type(node), None)
        if node_func is not None:
            pyomo_unit, pint_unit = node_func(self, node, data)
            if pint_unit is not None and \
                 self._pint_unit_equivalent_to_dimensionless(pint_unit):
                # I want to return None instead of dimensionless
                # but radians are also equivalent to dimensionless
                # For now, this test works, but we need to find a better approach
                teststr = '{:!~s}'.format(pint_unit)
                if teststr == '':
                    return (None, None)

            return (pyomo_unit, pint_unit)

        raise TypeError('An unhandled expression node type: {} was encountered while retrieving the'
                        ' units of expression'.format(str(node_type), str(node)))


class PyomoUnitsContainer(object):
    """Class that is used to create and contain units in Pyomo.

    This is the class that is used to create, contain, and interact with units in Pyomo.
    The module (:mod:`pyomo.core.base.units_container`) also contains a module attribute
    called `units` that is a singleton instance of a PyomoUnitsContainer. This singleton should be
    used instead of creating your own instance of a :py:class:`PyomoUnitsContainer`.
    For an overview of the usage of this class, see the module documentation
    (:mod:`pyomo.core.base.units_container`)

    This class is based on the "pint" module. Documentation for available units can be found
    at the following url: https://github.com/hgrecco/pint/blob/master/pint/default_en.txt

    Note: Pre-defined units can be accessed through attributes on the PyomoUnitsContainer
    class; however, these attributes are created dynamically through the __getattr__ method,
    and are not present on the class until they are requested.
    """
    def __init__(self):
        """Create a PyomoUnitsContainer instance. """
        # Developers: Do not interact with this attribute directly, but instead
        # access through the property _pint_registry since that is where the import
        # of the 'pint' module is checked
        self.__pint_registry = None

    @property
    def _pint_registry(self):
        """ Return the pint.UnitsRegistry instance corresponding to this container. """
        if pint_module is None:
            # pint was not imported for some reason
            raise RuntimeError("The PyomoUnitsContainer in the units_container module requires"
                              " the package 'pint', but this package could not be imported."
                              " Please make sure you have 'pint' installed.")

        if self.__pint_registry is None:
            self.__pint_registry = pint_module.UnitRegistry()

        return self.__pint_registry

    def __getattr__(self, item):
        """
        Here, __getattr__ is implemented to automatically create the necessary unit if
        the attribute does not already exist.

        Parameters
        ----------
        item : str
            the name of the new field requested

        Returns
        -------
        : PyomoUnit
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
                pint_unit_container = pint_module.util.to_units_container(pint_unit, pint_registry)
                for (u, e) in six.iteritems(pint_unit_container):
                    if not pint_registry._units[u].is_multiplicative:
                        raise UnitsError('Pyomo units system does not support the offset units "{}".'
                                         ' Use absolute units (e.g. kelvin instead of degC) instead.'
                                         ''.format(item))

                unit = _PyomoUnit(pint_unit, pint_registry)
                setattr(self, item, unit)
                return unit
        except pint_module.errors.UndefinedUnitError as exc:
            pint_unit = None

        if pint_unit is None:
            raise AttributeError('Attribute {0} not found.'.format(str(item)))

    def create_PyomoUnit(self, pint_unit):
        return _PyomoUnit(pint_unit, self._pint_registry)

    # TODO: Add support to specify a units definition file instead of this programatic interface
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

    def _get_units_tuple(self, expr):
        """
        Return a tuple of the PyomoUnit, and pint_unit corresponding to the expression in expr.

        Parameters
        ----------
        expr : Pyomo expression
           the input expression for extracting units

        Returns
        -------
        : tuple (PyomoUnit, pint unit)
        """
        pyomo_unit, pint_unit = _UnitExtractionVisitor(self).walk_expression(expr=expr)

        if pint_unit is not None:
            assert pyomo_unit is not None
            if type(pint_unit) != type(self._pint_registry.kg):
                pint_unit = pint_unit.units
            return (_PyomoUnit(pint_unit, self._pint_registry), pint_unit)
        return (None, None)

    def get_units(self, expr):
        """
        Return the Pyomo units corresponding to this expression (also performs validation
        and will raise an exception if units are not consistent).

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
        pyomo_unit, pint_unit = self._get_units_tuple(expr=expr)

        # Currently testing the idea that a PyomoUnit can contain a pint_unit expression
        # (instead of just a single unit). If this is OK long term, then we can clean up the
        # visitor code to only track the pint units
        return pyomo_unit

    def check_units_consistency(self, expr, allow_exceptions=True):
        """
        Check the consistency of the units within an expression. IF allow_exceptions is False,
        then this function swallows the exception and returns only True or False. Otherwise,
        it will throw an exception if the units are inconsistent.

        Parameters
        ----------
        expr : Pyomo expression
            The source expression to check.

        allow_exceptions: bool
            True if you want any exceptions to be thrown, False if you only want a boolean
            (and the exception is ignored).

        Returns
        -------
        : bool
           True if units are consistent, and False if not

        Raises
        ------
        :py:class:`pyomo.core.base.units_container.UnitsError`, :py:class:`pyomo.core.base.units_container.InconsistentUnitsError`

        """
        try:
            pyomo_unit, pint_unit = self._get_units_tuple(expr=expr)
        except (UnitsError, InconsistentUnitsError):
            if allow_exceptions:
                raise
            return False

        return True


    def check_units_equivalent(self, expr1, expr2):
        """
        Check if the units associated with each of the expressions are equivalent.

        Parameters
        ----------
        expr1 : Pyomo expression
           The first expression.
        expr2 : Pyomo expression
           The second expression.

        Returns
        -------
        : bool
           True if the expressions have equivalent units, False otherwise.

        Raises
        ------
        :py:class:`pyomo.core.base.units_container.UnitsError`, :py:class:`pyomo.core.base.units_container.InconsistentUnitsError`

        """
        pyomo_unit1, pint_unit1 = self._get_units_tuple(expr1)
        pyomo_unit2, pint_unit2 = self._get_units_tuple(expr2)
        return _UnitExtractionVisitor(self)._pint_units_equivalent(pint_unit1, pint_unit2)

    # def convert_value(self, src_value, from_units=None, to_units=None):
    #     """
    #     This method performs explicit conversion of a numerical value in
    #     one unit to a numerical value in another unit.
    #
    #     Parameters
    #     ----------
    #     src_value : float
    #        The numeric value that will be converted
    #     from_units : Pyomo expression with units
    #        The source units for value
    #     to_units : Pyomo expression with units
    #        The desired target units for the new value
    #
    #     Returns
    #     -------
    #        float : The new value (src_value converted from from_units to to_units)
    #     """
    #     from_pyomo_unit, from_pint_unit = self._get_units_tuple(from_units)
    #     to_pyomo_unit, to_pint_unit = self._get_units_tuple(to_units)
    #
    #     src_quantity = src_value * pint_src_unit
    #     dest_quantity = src_quantity.to(pint_dest_unit)
    #     return dest_quantity.magnitude

#: Module level instance of a PyomoUnitsContainer to use for all units within a Pyomo model
# See module level documentation for an example.
units = PyomoUnitsContainer()


