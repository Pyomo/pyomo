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

import enum

from pyomo.common.dependencies import attempt_import
from pyomo.common.numeric_types import native_types
from pyomo.core.pyomoobject import PyomoObject
from pyomo.core.expr.expr_common import OperatorAssociativity

visitor, _ = attempt_import('pyomo.core.expr.visitor')


class ExpressionBase(PyomoObject):
    """The base class for all Pyomo expression systems.

    This class is used to define nodes in a general expression tree.
    Individual expression systems (numeric, logical, etc.) will mix this
    class in with their fundamental base data type (NumericValue,
    BooleanValue, etc) to form the base node of that expression system.
    """

    __slots__ = ()

    PRECEDENCE = 0

    # Most operators in Python are left-to-right associative
    """Return the associativity of this operator.

    Returns 1 if this operator is left-to-right associative or -1 if
    it is right-to-left associative.  Any other return value will be
    interpreted as "not associative" (implying any arguments that
    are at this operator's PRECEDENCE will be enclosed in parens).
    """
    ASSOCIATIVITY = OperatorAssociativity.LEFT_TO_RIGHT

    def nargs(self):
        """Returns the number of child nodes.

        Note
        ----

        Individual expression nodes may use different internal storage
        schemes, so it is imperative that developers use this method and
        not assume the existence of a particular attribute!

        Returns
        -------
        int: A nonnegative integer that is the number of child nodes.

        """
        raise NotImplementedError(
            f"Derived expression ({self.__class__}) failed to implement nargs()"
        )

    def arg(self, i):
        """Return the i-th child node.

        Parameters
        ----------
        i: int
            Index of the child argument to return

        Returns: The i-th child node.

        """
        if i < 0:
            i += self.nargs()
            if i < 0:
                raise KeyError(
                    "Invalid index for expression argument: %d" % i - self.nargs()
                )
        elif i >= self.nargs():
            raise KeyError("Invalid index for expression argument: %d" % i)
        return self._args_[i]

    @property
    def args(self):
        """Return the child nodes

        Returns
        -------
        list or tuple:
            Sequence containing only the child nodes of this node.  The
            return type depends on the node storage model.  Users are
            not permitted to change the returned data (even for the case
            of data returned as a list), as that breaks the promise of
            tree immutability.

        """
        raise NotImplementedError(
            f"Derived expression ({self.__class__}) failed to implement args()"
        )

    def __call__(self, exception=True):
        """Evaluate the value of the expression tree.

        Parameters
        ----------
        exception: bool
            If :const:`False`, then an exception raised while evaluating
            is captured, and the value returned is :const:`None`.
            Default is :const:`True`.

        Returns
        -------
        The value of the expression or :const:`None`.

        """
        return visitor.evaluate_expression(self, exception)

    def __str__(self):
        """Returns a string description of the expression.

        Note:

        The value of ``pyomo.core.expr.expr_common.TO_STRING_VERBOSE``
        is used to configure the execution of this method.  If this
        value is :const:`True`, then the string representation is a
        nested function description of the expression.  The default is
        :const:`False`, which returns an algebraic (infix notation)
        description of the expression.

        Returns
        -------
        str
        """
        return visitor.expression_to_string(self)

    def to_string(self, verbose=None, labeler=None, smap=None, compute_values=False):
        """Return a string representation of the expression tree.

        Parameters
        ----------
        verbose: bool
            If :const:`True`, then the string representation
            consists of nested functions.  Otherwise, the string
            representation is an algebraic (infix notation) equation.
            Defaults to :const:`False`.

        labeler:
            An object that generates string labels for variables in the
            expression tree.  Defaults to :const:`None`.

        smap:
            If specified, this
            :class:`SymbolMap <pyomo.core.expr.symbol_map.SymbolMap>`
            is used to cache labels for variables.

        compute_values (bool):
            If :const:`True`, then parameters and fixed variables are
            evaluated before the expression string is generated.
            Default is :const:`False`.

        Returns:
            A string representation for the expression tree.

        """
        return visitor.expression_to_string(
            self,
            verbose=verbose,
            labeler=labeler,
            smap=smap,
            compute_values=compute_values,
        )

    def _to_string(self, values, verbose, smap):
        """
        Construct a string representation for this node, using the string
        representations of its children.

        This method is called by the :class:`_ToStringVisitor
        <pyomo.core.expr.current._ToStringVisitor>` class.  It must
        must be defined in subclasses.

        Args:
            values (list): The string representations of the children of this
                node.
            verbose (bool): If :const:`True`, then the string
                representation consists of nested functions.  Otherwise,
                the string representation is an algebraic equation.
            smap:  If specified, this :class:`SymbolMap
                <pyomo.core.expr.symbol_map.SymbolMap>` is
                used to cache labels for variables.

        Returns:
            A string representation for this node.
        """
        raise NotImplementedError(
            f"Derived expression ({self.__class__}) failed to implement _to_string()"
        )

    def getname(self, *args, **kwds):
        """Return the text name of a function associated with this expression
        object.

        In general, no arguments are passed to this function.

        Args:
            *arg: a variable length list of arguments
            **kwds: keyword arguments

        Returns:
            A string name for the function.

        """
        raise NotImplementedError(
            f"Derived expression ({self.__class__}) failed to implement getname()"
        )

    def clone(self, substitute=None):
        """
        Return a clone of the expression tree.

        Note:
            This method does not clone the leaves of the
            tree, which are numeric constants and variables.
            It only clones the interior nodes, and
            expression leaf nodes like
            :class:`_MutableLinearExpression<pyomo.core.expr.current._MutableLinearExpression>`.
            However, named expressions are treated like
            leaves, and they are not cloned.

        Args:
            substitute (dict): a dictionary that maps object ids to clone
                objects generated earlier during the cloning process.

        Returns:
            A new expression tree.
        """
        return visitor.clone_expression(self, substitute=substitute)

    def create_node_with_local_data(self, args, classtype=None):
        """
        Construct a node using given arguments.

        This method provides a consistent interface for constructing a
        node, which is used in tree visitor scripts.  In the simplest
        case, this returns::

            self.__class__(args)

        But in general this creates an expression object using local
        data as well as arguments that represent the child nodes.

        Args:
            args (list): A list of child nodes for the new expression
                object

        Returns:
            A new expression object with the same type as the current
            class.
        """
        if classtype is None:
            classtype = self.__class__
        return classtype(args)

    def is_constant(self):
        """Return True if this expression is an atomic constant

        This method contrasts with the is_fixed() method.  This method
        returns True if the expression is an atomic constant, that is it
        is composed exclusively of constants and immutable parameters.
        NumericValue objects returning is_constant() == True may be
        simplified to their numeric value at any point without warning.

        Note:  This defaults to False, but gets redefined in sub-classes.
        """
        return False

    def is_fixed(self):
        """
        Return :const:`True` if this expression contains no free variables.

        Returns:
            A boolean.
        """
        return visitor._expression_is_fixed(self)

    def _is_fixed(self, values):
        """
        Compute whether this expression is fixed given
        the fixed values of its children.

        This method is called by the :class:`_IsFixedVisitor
        <pyomo.core.expr.current._IsFixedVisitor>` class.  It can
        be over-written by expression classes to customize this
        logic.

        Args:
            values (list): A list of boolean values that indicate whether
                the children of this expression are fixed

        Returns:
            A boolean that is :const:`True` if the fixed values of the
            children are all :const:`True`.
        """
        return all(values)

    def is_potentially_variable(self):
        """
        Return :const:`True` if this expression might represent
        a variable expression.

        This method returns :const:`True` when (a) the expression
        tree contains one or more variables, or (b) the expression
        tree contains a named expression. In both cases, the
        expression cannot be treated as constant since (a) the variables
        may not be fixed, or (b) the named expressions may be changed
        at a later time to include non-fixed variables.

        Returns:
            A boolean.  Defaults to :const:`True` for expressions.
        """
        return True

    def is_named_expression_type(self):
        """
        Return :const:`True` if this object is a named expression.

        This method returns :const:`False` for this class, and it
        is included in other classes within Pyomo that are not named
        expressions, which allows for a check for named expressions
        without evaluating the class type.

        Returns:
            A boolean.
        """
        return False

    def is_expression_type(self, expression_system=None):
        """
        Return :const:`True` if this object is an expression.

        This method obviously returns :const:`True` for this class, but it
        is included in other classes within Pyomo that are not expressions,
        which allows for a check for expressions without
        evaluating the class type.

        Returns:
            A boolean.
        """
        return expression_system is None or expression_system == self.EXPRESSION_SYSTEM

    def size(self):
        """
        Return the number of nodes in the expression tree.

        Returns:
            A nonnegative integer that is the number of interior and leaf
            nodes in the expression tree.
        """
        return visitor.sizeof_expression(self)

    def _apply_operation(self, result):  # pragma: no cover
        """
        Compute the values of this node given the values of its children.

        This method is called by the :class:`_EvaluationVisitor
        <pyomo.core.expr.current._EvaluationVisitor>` class.  It must
        be over-written by expression classes to customize this logic.

        Note:
            This method applies the logical operation of the
            operator to the arguments.  It does *not* evaluate
            the arguments in the process, but assumes that they
            have been previously evaluated.  But note that if
            this class contains auxiliary data (e.g. like the
            numeric coefficients in the :class:`LinearExpression
            <pyomo.core.expr.current.LinearExpression>` class) then
            those values *must* be evaluated as part of this
            function call.  An uninitialized parameter value
            encountered during the execution of this method is
            considered an error.

        Args:
            values (list): A list of values that indicate the value
                of the children expressions.

        Returns:
            A floating point value for this expression.
        """
        raise NotImplementedError(
            f"Derived expression ({self.__class__}) failed to "
            "implement _apply_operation()"
        )


class NPV_Mixin(object):
    __slots__ = ()

    def is_potentially_variable(self):
        return False

    def create_node_with_local_data(self, args, classtype=None):
        assert classtype is None
        try:
            npv_args = all(
                type(arg) in native_types or not arg.is_potentially_variable()
                for arg in args
            )
        except AttributeError:
            # We can hit this during expression replacement when the new
            # type is not a PyomoObject type, but is not in the
            # native_types set.  We will play it safe and clear the NPV flag
            npv_args = False
        if npv_args:
            return super().create_node_with_local_data(args, None)
        else:
            return super().create_node_with_local_data(
                args, self.potentially_variable_base_class()
            )

    def potentially_variable_base_class(self):
        cls = list(self.__class__.__bases__)
        cls.remove(NPV_Mixin)
        assert len(cls) == 1
        return cls[0]


class ExpressionArgs_Mixin(object):
    __slots__ = ('_args_',)

    def __init__(self, args):
        self._args_ = args

    def nargs(self):
        return len(self._args_)

    @property
    def args(self):
        """
        Return the child nodes

        Returns
        -------
        list or tuple:
            Sequence containing only the child nodes of this node.  The
            return type depends on the node storage model.  Users are
            not permitted to change the returned data (even for the case
            of data returned as a list), as that breaks the promise of
            tree immutability.
        """
        return self._args_
