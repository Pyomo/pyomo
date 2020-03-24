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

This module provides support for including units within Pyomo expressions. This module
can be used to define units on a model, and to check the consistency of units
within the underlying constraints and expressions in the model. The module also
supports conversion of units within expressions to support construction of constraints
that contain embedded unit conversions.

To use this package within your Pyomo model, you first need an instance of a
PyomoUnitsContainer. You can use the module level instance already defined as
'units'. This object 'contains' the units - that is, you can access units on
this module using common notation.

    .. doctest::

       >>> from pyomo.environ import units as u
       >>> print(3.0*u.kg)
       3.0*kg

Units can be assigned to Var, Param, and ExternalFunction components, and can
be used directly in expressions (e.g., defining constraints). You can also
verify that the units are consistent on a model, or on individual components
like the objective function, constraint, or expression using
`assert_units_consistent`. There are other methods that may be helpful
for verifying correct units on a model.

    .. doctest::

       >>> from pyomo.environ import ConcreteModel, Var, Objective
       >>> from pyomo.environ import units as u
       >>> model = ConcreteModel()
       >>> model.acc = Var(initialize=5.0, units=u.m/u.s**2)
       >>> model.obj = Objective(expr=(model.acc - 9.81*u.m/u.s**2)**2)
       >>> u.assert_units_consistent(model.obj) # raise exc if units invalid on obj
       >>> u.assert_units_consistent(model) # raise exc if units invalid anywhere on the model
       >>> u.assert_units_equivalent(model.obj.expr, u.m**2/u.s**4) # raise exc if units not equivalent
       >>> print(u.get_units(model.obj.expr)) # print the units on the objective
       m ** 2 / s ** 4
       >>> print(u.check_units_equivalent(model.acc.get_units(), u.m/u.s**2))
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

import six


        Parameters
        ----------
        pyomo_units_container : PyomoUnitsContainer
           Instance of the PyomoUnitsContainer that was used for the units
           different containers

        units_equivalence_tolerance : float (default 1e-12)
            Floating point tolerance used when deciding if units are equivalent
            or not.

        Notes
        -----
        This class inherits from the :class:`StreamBasedExpressionVisitor` to implement
        a walker that returns the pyomo units and pint units corresponding to an

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
        if lhs is rhs:
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

        pyomo_unit, pint_unit = list_of_unit_tuples[0]
        return (pyomo_unit, pint_unit)

    def _get_units_with_dimensionless_children(self, node, list_of_unit_tuples):
        """
        Check to make sure that any child arguments are unitless /
        dimensionless and return the value from node.get_units() This
        was written for ExternalFunctionExpression where the external
        function has units assigned to its return value.

        Parameters
        ----------
        node : Pyomo expression node
            The parent node of the children

        list_of_unit_tuples : list
           This is a list of tuples (one for each of the children) where each tuple
           is a PyomoUnit, pint unit pair

        Returns
        -------
        : tuple (pyomo_unit, pint_unit)

        """
        for (pyomo_unit, pint_unit) in list_of_unit_tuples:
            if not self._pint_unit_equivalent_to_dimensionless(pint_unit):
                raise UnitsError('Expected no units or dimensionless units in {}, but found {}.'.format(str(node), str(pyomo_unit)))

        # now return the units in node.get_units
        return self._pyomo_units_container._get_units_tuple(node.get_units())

    def _get_dimensionless_with_dimensionless_children(self, node, list_of_unit_tuples):
        """
        Check to make sure that any child arguments are unitless /
        dimensionless (for functions like exp()) and return (None,
        None) if successful. Although odd that this does not just
        return a boolean, it is done this way to match the signature
        of the other methods used to get units for expressions.

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

        pyomo_unit, pint_unit = list_of_unit_tuples[0]
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

        pyomo_unit, pint_unit = list_of_unit_tuples[0]
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
        EXPR.ReciprocalExpression: _get_unit_for_reciprocal,
        EXPR.NPV_ReciprocalExpression: _get_unit_for_reciprocal,
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
        EXPR.GetItemExpression: _get_dimensionless_with_dimensionless_children,
        EXPR.ExternalFunctionExpression: _get_units_with_dimensionless_children,
        EXPR.NPV_ExternalFunctionExpression: _get_units_with_dimensionless_children,
        EXPR.LinearExpression: _get_unit_for_linear_expression
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
            if type(node) in native_numeric_types:
                # this is a number - return dimensionless                                                                      
                return (None, None)
            elif isinstance(node, _PyomoUnit):
                return (node, node._get_pint_unit())
            # CDL using the hasattr code below since it is more general
            #elif isinstance(node, _VarData) or \
            #     isinstance(node, _ParamData):
            #    pyomo_unit, pint_unit = self._pyomo_units_container._get_units_tuple(node.get_units())
            #    return (pyomo_unit, pint_unit)
            elif hasattr(node, 'get_units'):
                pyomo_unit, pint_unit = self._pyomo_units_container._get_units_tuple(node.get_units())
                return (pyomo_unit, pint_unit)
            
            # I have a leaf, but this is not a PyomoUnit - (treat as dimensionless)
            return (None, None)

        # not a leaf - check if it is a named expression
        if hasattr(node, 'is_named_expression_type') and node.is_named_expression_type():
            return self._get_unit_for_single_child(node, data)

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
    def __init__(self):
        """Create a PyomoUnitsContainer instance."""
        self._pint_registry = pint_module.UnitRegistry()

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
            :hide:

            # get a local units object (to avoid duplicate registration
            # with the example in load_definitions_from_strings)
            >>> import pyomo.core.base.units_container as _units
            >>> u = _units.PyomoUnitsContainer()
            >>> with open('my_additional_units.txt', 'w') as FILE:
            ...     tmp = FILE.write("USD = [currency]\\n")

        .. doctest::

            >>> u.load_definitions_from_file('my_additional_units.txt')
            >>> print(u.USD)
            USD

        """
        self._pint_registry.load_definitions(definition_file)

    def load_definitions_from_strings(self, definition_string_list):
        """Load new units definitions from a string

        This method loads additional units definitions from a list of
        strings (one for each line). An example of the definitions
        strings can be found at:
        https://github.com/hgrecco/pint/blob/master/pint/default_en.txt

        For example, to add the currency dimension and US dollars as a
        unit, use

        .. doctest::
            :hide:

            # get a local units object (to avoid duplicate registration
            # with the example in load_definitions_from_strings)
            >>> import pyomo.core.base.units_container as _units
            >>> u = _units.PyomoUnitsContainer()

        .. doctest::

            >>> u.load_definitions_from_strings(['USD = [currency]'])
            >>> print(u.USD)
            USD

        """
        self._pint_registry.load_definitions(definition_string_list)

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

    # We added support to specify a units definition file instead of this programatic interface
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

    def _pint_convert_temp_from_to(self, numerical_value, pint_from_units, pint_to_units):
        if type(numerical_value) not in native_numeric_types:
            raise UnitsError('Conversion routines for absolute and relative temperatures require a numerical value only.'
                             ' Pyomo objects (Var, Param, expressions) are not supported. Please use value(x) to'
                             ' extract the numerical value if necessary.')
        
        src_quantity = self._pint_registry.Quantity(numerical_value, pint_from_units)
        dest_quantity = src_quantity.to(pint_to_units)
        return dest_quantity.magnitude
        
    def convert_temp_K_to_C(self, value_in_K):
        """
        Convert a value in Kelvin to degrees Celcius.  Note that this method
        converts a numerical value only. If you need temperature
        conversions in expressions, please work in absolute
        temperatures only.
        """
        return self._pint_convert_temp_from_to(value_in_K, self._pint_registry.K, self._pint_registry.degC)

    def convert_temp_C_to_K(self, value_in_C):
        """
        Convert a value in degrees Celcius to Kelvin Note that this
        method converts a numerical value only. If you need
        temperature conversions in expressions, please work in
        absolute temperatures only.
        """
        return self._pint_convert_temp_from_to(value_in_C, self._pint_registry.degC, self._pint_registry.K)

    def convert_temp_R_to_F(self, value_in_R):
        """
        Convert a value in Rankine to degrees Fahrenheit.  Note that
        this method converts a numerical value only. If you need
        temperature conversions in expressions, please work in
        absolute temperatures only.
        """
        return self._pint_convert_temp_from_to(value_in_R, self._pint_registry.rankine, self._pint_registry.degF)

    def convert_temp_F_to_R(self, value_in_F):
        """
        Convert a value in degrees Fahrenheit to Rankine.  Note that
        this method converts a numerical value only. If you need
        temperature conversions in expressions, please work in
        absolute temperatures only.
        """
        return self._pint_convert_temp_from_to(value_in_F, self._pint_registry.degF, self._pint_registry.rankine)

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
        src_pyomo_unit, src_pint_unit = self._get_units_tuple(src)
        to_pyomo_unit, to_pint_unit = self._get_units_tuple(to_units)

        # check if any units have offset
        # CDL: This is no longer necessary since we don't allow
        # offset units, but let's keep the code in case we change
        # our mind about offset units
        #  src_unit_container = pint.util.to_units_container(src_unit, self._pint_ureg)
        # dest_unit_container = pint.util.to_units_container(dest_unit, self._pint_ureg)
        # src_offset_units = [(u, e) for u, e in src_unit_container.items()
        #                     if not self._pint_ureg._units[u].is_multiplicative]
        # 
        #  dest_offset_units = [(u, e) for u, e in dest_unit_container.items()
        #                 if not self._pint_ureg._units[u].is_multiplicative]

        # if len(src_offset_units) + len(dest_offset_units) != 0:
        #     raise UnitsError('Offset unit detected in call to convert. Offset units are not supported at this time.')

        # no offsets, we only need a factor to convert between the two
        fac_b_src, base_units_src = self._pint_registry.get_base_units(src_pint_unit, check_nonmult=True)
        fac_b_dest, base_units_dest = self._pint_registry.get_base_units(to_pint_unit, check_nonmult=True)

        if base_units_src != base_units_dest:
            raise UnitsError('Cannot convert {0:s} to {1:s}. Units are not compatible.'.format(str(src_pyomo_unit), str(to_pyomo_unit)))

        return fac_b_src/fac_b_dest*to_pyomo_unit/src_pyomo_unit*src

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
            raise UnitsError('The argument "num_value" in convert_value must be a native numeric type, but'
                             ' instead type {} was found.'.format(type(num_value)))
        
        from_pyomo_unit, from_pint_unit = self._get_units_tuple(from_units)
        to_pyomo_unit, to_pint_unit = self._get_units_tuple(to_units)

        # ToDo: This check may be overkill - pint will raise an error that may be sufficient
        fac_b_src, base_units_src = self._pint_registry.get_base_units(from_pint_unit, check_nonmult=True)
        fac_b_dest, base_units_dest = self._pint_registry.get_base_units(to_pint_unit, check_nonmult=True)
        if base_units_src != base_units_dest:
            raise UnitsError('Cannot convert {0:s} to {1:s}. Units are not compatible.'.format(str(from_pyomo_unit), str(to_pyomo_unit)))

        # convert the values
        src_quantity = num_value * from_pint_unit
        dest_quantity = src_quantity.to(to_pint_unit)
