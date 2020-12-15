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


#ToDo: check documentation strings
from pyomo.core.expr.numvalue import NumericValue, nonpyomo_leaf_types, value, native_types, native_numeric_types
from pyomo.core.base.units_container import _PyomoUnit
from pyomo.core.expr.template_expr import IndexTemplate
from pyomo.core.expr import current as EXPR

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
        self._pint_registry = pyomo_units_container._pint_registry

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
        # TODO: This may be expensive for long summations and, in the case of reporting only, we may want to skip the checks
        assert len(list_of_unit_tuples) > 0

        # verify that the pint units are equivalent from each
        # of the child nodes - assume that PyomoUnits are equivalent
        pint_unit_0 = child_units[0]
        for i in range(1, len(child_units)):
            pint_unit_i = child_units[i]
            if pint_unit_0 != pint_unit_i:
                raise InconsistentUnitsError(pint_unit_0, pint_unit_i,
                        'Error in units found in expression: {}'.format(str(node)))

        # checks were OK, return the first one in the list
        return pint_unit_0

    def _get_unit_for_linear_expression(self, node, child_units):
        """
        Return (and test) the units corresponding to a :py:class:`LinearExpression` node
        in the expression tree.

        This is a special node since it does not use "args" the way
        other expression types do. Because of this, the StreamBasedExpressionVisitor
        does not pick up on the "children", and child_units is empty.
        Therefore, we implement the recursion into coeffs and vars ourselves.

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
        # StreamBasedExpressionVisitor does not handle the children of this node
        assert len(child_units) == 0

        # TODO: This may be expensive for long summations and, in the case of reporting only, we may want to skip the checks
        term_unit_list = []
        if node.constant:
            # we have a non-zero constant term, get its units
            const_pint_units = self._pyomo_units_container.get_pint_units(node.constant)
            term_unit_list.append(const_pint_units)

        # go through the coefficients and variables
        assert len(node.linear_coefs) == len(node.linear_vars)
        for k,v in enumerate(node.linear_vars):
            c = node.linear_coefs[k]
            v_units = self._pyomo_units_container._get_pint_units(v)
            c_units = self._pyomo_units_container._get_pint_units(c)
            term_unit_list.append(c_units*v_units)

        assert len(term_unit_list) > 0

        return self._get_unit_for_equivalent_children(node, term_unit_list)

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
        assert len(list_of_unit_tuples) > 1

        pint_unit = child_units[0]
        for i in range(1, len(child_units)):
            pint_unit *= child_units[1]

        # this operation can create a quantity, but we want a pint unit object
        return pint_unit.units

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
        return (child_units[0]/child_units[1]).units

    def _get_unit_for_reciprocal(self, node, child_units):
        """
        Return (and test) the units corresponding to a reciprocal expression node
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
        assert len(child_units) == 1

        # this operation can create a quantity, but we want a pint unit object
        return (1.0/child_units).units

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

        # common case - exponent is a constant number
        exponent = node.args[1]
        if type(exponent) in nonpyomo_leaf_types:
            return child_units[0]**value(exponent)

        # if base is dimensioness, exponent doesn't matter
        if child_units[0] is self._pint_registry.dimensionless or \
           child_units[0] == self._pint_registry.dimensionless:
            return self._pint_registry.dimensionless

        # base is not dimensionless, exponent should be dimensionless
        if child_units[1] != self._pint_registry.dimensionless:
            # todo: allow radians?
            raise UnitsError("Error in sub-expression: {}. "
                             "Exponents in a pow expression must be dimensionless."
                             "".format(node))

        # base is not dimensionless, exponent is dimensionless
        # ensure that the exponent is fixed
        if exponent.is_constant() == False and exponent.is_fixed() == False:
            raise UnitsError("The base of an exponent has units {}, but "
                             "the exponent is not a fixed numerical value."
                             "".format(str(list_of_unit_tuples[0][0])))

        return child_units[0]**value(exponent)

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
        if arg_units is None:
            # they should all be dimensionless
            arg_units = [self._pint_registry.dimensionless]*len(child_units)
            
        for (arg_unit, pint_unit) in zip(arg_units, child_units):
            if not arg_unit is pint_unit and arg_unit != pint_unit:
                raise InconsistentUnitsError(arg_unit, pint_unit, 'Inconsistent units found in ExternalFunction.')

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
        dless = self._pint_registry.dimensionless
        for pint_unit in child_units:
            if not pint_unit is dless and \
               not pint_unit == dless:
                raise UnitsError('Expected no units or dimensionless units in {}, but found {}.'.format(str(node), str(pint_unit)))

        return dless

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
        return self._pint_registry.dimensionless

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
            raise TypeError('An unhandled unary function: {} was encountered while retrieving the'
                        ' units of expression {}'.format(func_name, str(node)))
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
        if child_units[1] is not child_units[2] and \
           child_units[1] != child_units[2]:
            raise InconsistentUnitsError(then_pyomo_unit, else_pyomo_unit,
                    'Error in units found in expression: {}'.format(str(node)))

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

        dless = self._pint_registry.dimensionless
        rads = self._pint_registry.radians
        pint_units = child_units[0] 
        if pint_units is dless or pint_units is rads or\
           pint_units == dless or pint_units == rads:
            return dless

        # units are not None, dimensionless, or radians
        raise UnitsError('Expected radians or dimensionless in argument to function in expression {}, but found {}'.format(
            str(node), str(pint_units)))

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
        pint_units = child_units[0]

        dless = self._pint_registry.dimensionless
        if pint_units is not dless and pint_units != dless:
            raise UnitsError('Expected dimensionless argument to function in expression {},'
                             ' but found {}'.format(
                             str(node), str(pint_units)))

        return self._pint_registry.radians

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
        return child_units**0.5

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
        EXPR.ExternalFunctionExpression: _get_units_ExternalFunction,
        EXPR.NPV_ExternalFunctionExpression: _get_units_ExternalFunction,
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
        nodetype = type(node)
        if nodetype in native_types:
            return self._pint_registry.dimensionless

        elif not node.is_expression_type():
            # this is a leaf, but not a native type
            if nodetype is _PyomoUnit:
                return node._get_pint_unit()
            
            # might want to add other common types here
            pyomo_unit = node.get_units()

            # need to change this to only get pint units
            pyomo_unit, pint_unit = self._pyomo_units_container._get_units_tuple(node.get_units())
            return pint_unit

        # not a leaf - get the appropriate function for type of the node
        node_func = self.node_type_method_map.get(type(node), None)
        if node_func is not None:
            pint_unit = node_func(self, node, data)
            return pint_unit

        # not a leaf - check if it is a named expression
        if hasattr(node, 'is_named_expression_type') and node.is_named_expression_type():
            return self._get_unit_for_single_child(node, data)

        raise TypeError('An unhandled expression node type: {} was encountered while retrieving the'
                        ' units of expression'.format(str(node_type), str(node)))

