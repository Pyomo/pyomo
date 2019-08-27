 def nargs(self):
        """
        Returns the number of child nodes.
        By default, logical expression represents binary expression.
        #0-0 should we make this 1 or like in numexpr 2?
        """
        return 2

    def arg(self, i):

        """
        Return the i-th child node.

        Args:
            i (int): Nonnegative index of the child that is returned.

        Returns:
            The i-th child node.
        """
        if i >= self.nargs():
            raise KeyError("Invalid index for expression argument: %d" % i)
        if i < 0:
            return self._args_[self.nargs()+i]
            #0-0 send a warning?
        return self._args_[i]

    @property
    def args(self):
        """
        Return the child nodes

        Returns: Either a list or tuple (depending on the node storage
            model) containing only the child nodes of this node
        """
        return self._args_[:self.nargs()]


    def __getstate__(self):
        """
        Pickle the expression object

        Returns:
            The pickled state.
        """
        state = super(ExpressionBase, self).__getstate__()
        for i in ExpressionBase.__slots__:
           state[i] = getattr(self,i)
        return state

    def __call__(self, exception=True):
        """
        Evaluate the value of the expression tree.
        #0-0 leave it for now
        Args:
            exception (bool): If :const:`False`, then
                an exception raised while evaluating
                is captured, and the value returned is
                :const:`None`.  Default is :const:`True`.

        Returns:
            The value of the expression or :const:`None`.
        """
        return evaluate_expression(self, exception)

    def __str__(self):
        """
        Returns a string description of the expression.
        #leave it for now
        Note:
            The value of ``pyomo.core.expr.expr_common.TO_STRING_VERBOSE``
            is used to configure the execution of this method.
            If this value is :const:`True`, then the string
            representation is a nested function description of the expression.
            The default is :const:`False`, which is an algebraic
            description of the expression.

        Returns:
            A string.
        """
        return expression_to_string(self)

    def to_string(self, verbose=None, labeler=None, smap=None, compute_values=False):
        """
        Return a string representation of the expression tree.
        #leave it for now
        Args:
            verbose (bool): If :const:`True`, then the the string
                representation consists of nested functions.  Otherwise,
                the string representation is an algebraic equation.
                Defaults to :const:`False`.
            labeler: An object that generates string labels for
                variables in the expression tree.  Defaults to :const:`None`.
            smap:  If specified, this :class:`SymbolMap <pyomo.core.expr.symbol_map.SymbolMap>` is
                used to cache labels for variables.
            compute_values (bool): If :const:`True`, then
                parameters and fixed variables are evaluated before the
                expression string is generated.  Default is :const:`False`.

        Returns:
            A string representation for the expression tree.
        """
        return expression_to_string(self, verbose=verbose, labeler=labeler, smap=smap, compute_values=compute_values)

    def _precedence(self):
        return ExpressionBase.PRECEDENCE

    def _associativity(self):
        """Return the associativity of this operator.

        Returns 1 if this operator is left-to-right associative or -1 if
        it is right-to-left associative.  Any other return value will be
        interpreted as "not associative" (implying any arguments that
        are at this operator's _precedence() will be enclosed in parens).
        """
        #0-0 do we need this?
        return 1

    def _to_string(self, values, verbose, smap, compute_values):            #pragma: no cover
        """
        #0-0 pass for now
        Construct a string representation for this node, using the string
        representations of its children.

        This method is called by the :class:`_ToStringVisitor
        <pyomo.core.expr.current._ToStringVisitor>` class.  It must
        must be defined in subclasses.

        Args:
            values (list): The string representations of the children of this
                node.
            verbose (bool): If :const:`True`, then the the string
                representation consists of nested functions.  Otherwise,
                the string representation is an algebraic equation.
            smap:  If specified, this :class:`SymbolMap
                <pyomo.core.expr.symbol_map.SymbolMap>` is
                used to cache labels for variables.
            compute_values (bool): If :const:`True`, then
                parameters and fixed variables are evaluated before the
                expression string is generated.

        Returns:
            A string representation for this node.
        """
        pass

    def getname(self, *args, **kwds):                       #pragma: no cover
        """
        Return the text name of a function associated with this expression object.

        In general, no arguments are passed to this function.

        Args:
            *arg: a variable length list of arguments
            **kwds: keyword arguments

        Returns:
            A string name for the function.
        """
        raise NotImplementedError("Derived expression (%s) failed to "\
            "implement getname()" % ( str(self.__class__), ))

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
        return clone_expression(self, substitute=substitute)

    def create_node_with_local_data(self, args):
        """
        Construct a node using given arguments.

        This method provides a consistent interface for constructing a
        node, which is used in tree visitor scripts.  In the simplest
        case, this simply returns::

            self.__class__(args)

        But in general this creates an expression object using local
        data as well as arguments that represent the child nodes.

        Args:
            args (list): A list of child nodes for the new expression
                object
            memo (dict): A dictionary that maps object ids to clone
                objects generated earlier during a cloning process.
                This argument is needed to clone objects that are
                owned by a model, and it can be safely ignored for
                most expression classes.

        Returns:
            A new expression object with the same type as the current
            class.
        """
        return self.__class__(args)

    def create_potentially_variable_object(self):
        """
        Create a potentially variable version of this object.

        This method returns an object that is a potentially variable
        version of the current object.  In the simplest
        case, this simply sets the value of `__class__`:

            self.__class__ = self.__class__.__mro__[1]

        Note that this method is allowed to modify the current object
        and return it.  But in some cases it may create a new
        potentially variable object.

        Returns:
            An object that is potentially variable.
        """
        self.__class__ = self.__class__.__mro__[1]
        return self

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
        return _expression_is_fixed(self)

    def _is_fixed(self, values):
        """

        # 0-0 leave it for now?

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

    def is_expression_type(self):
        """
        Return :const:`True` if this object is an expression.

        This method obviously returns :const:`True` for this class, but it
        is included in other classes within Pyomo that are not expressions,
        which allows for a check for expressions without
        evaluating the class type.

        Returns:
            A boolean.
        """
        return True

    def size(self):
        """
        Return the number of nodes in the expression tree.

        Returns:
            A nonnegative integer that is the number of interior and leaf
            nodes in the expression tree.
        """
        return sizeof_expression(self)

    #def polynomial_degree(self):
    #not needed for logical expresion
    #def _compute_polynomial_degree(self, values):                          #pragma: no cover


    def _apply_operation(self, result):     #pragma: no cover
        """
        Compute the values of this node given the values of its children.

        This method is called by the :class:`_EvaluationVisitor
        <pyomo.core.expr.current._EvaluationVisitor>` class.  It must
        be over-written by expression classes to customize this logic.

        Note:
            This method applies the logical operation of the
            operator to the arguments.  It does *not* evaluate
            the arguments in the process, but assumes that they
            have been previously evaluated.  But noted that if
            this class contains auxilliary data (e.g. like the
            numeric coefficients in the :class:`LinearExpression
            <pyomo.core.expr.current.LinearExpression>` class, then
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
            raise NotImplementedError("Derived expression (%s) failed to "\
                "implement _apply_operation()" % ( str(self.__class__), ))