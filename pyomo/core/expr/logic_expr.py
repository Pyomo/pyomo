#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


"""
Notes. Delete later
"""

from __future__ import division

import logging
import traceback

logger = logging.getLogger('pyomo.core')

from .logicalvalue import (
    LogicalValue,
    LogicalConstant,
    as_logical,
    value,
)

from .expr_common import (
    _add, _sub, _mul, _div,
    _pow, _neg, _abs, _inplace,
    _unary, _radd, _rsub, _rmul,
    _rdiv, _rpow, _iadd, _isub,
    _imul, _idiv, _ipow, _lt, _le,
    _eq,
)


class LogicalExpressionBase(LogicalValue):
    """
    Logical expressions base expression.

    This class is used to define nodes in an expression
    tree.
    
    Abstract

    argss:
        argss (list or tuple): Children of this node.
    """

    # 0-0 do we need this and used this new base for the expressions above?
    __slots__ =  ('_args_',)
    PRECEDENCE = 0

    def __init__(self, args):
        self._args_ = args


    def nargs(self):
        """
        Returns the number of child nodes.
        By default, logical expression represents binary expression.
        #0-0 should we make this 1 or like in numexpr 2?
        """
        return 2

    def args(self, i):
        #0-0
        """
        Return the i-th child node.

        argss:
            i (int): Nonnegative index of the child that is returned.

        Returns:
            The i-th child node.
        """
        if i >= self.nargs():
            raise KeyError("Invalid index for expression argsument: %d" % i)
        if i < 0:
            return self._args_[self.nargss()+i]
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
        argss:
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
        argss:
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
        interpreted as "not associative" (implying any argsuments that
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

        argss:
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

        In general, no argsuments are passed to this function.

        argss:
            *arg: a variable length list of argsuments
            **kwds: keyword argsuments

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

        argss:
            substitute (dict): a dictionary that maps object ids to clone
                objects generated earlier during the cloning process.

        Returns:
            A new expression tree.
        """
        return clone_expression(self, substitute=substitute)

    def create_node_with_local_data(self, argss):
        """
        Construct a node using given argsuments.

        This method provides a consistent interface for constructing a
        node, which is used in tree visitor scripts.  In the simplest
        case, this simply returns::

            self.__class__(argss)

        But in general this creates an expression object using local
        data as well as argsuments that represent the child nodes.

        argss:
            argss (list): A list of child nodes for the new expression
                object
            memo (dict): A dictionary that maps object ids to clone
                objects generated earlier during a cloning process.
                This argsument is needed to clone objects that are
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

        argss:
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
            operator to the argsuments.  It does *not* evaluate
            the argsuments in the process, but assumes that they
            have been previously evaluated.  But noted that if
            this class contains auxilliary data (e.g. like the
            numeric coefficients in the :class:`LinearExpression
            <pyomo.core.expr.current.LinearExpression>` class, then
            those values *must* be evaluated as part of this
            function call.  An uninitialized parameter value
            encountered during the execution of this method is
            considered an error.

        argss:
            values (list): A list of values that indicate the value
                of the children expressions.

        Returns:
            A floating point value for this expression.
        """
        raise NotImplementedError("Derived expression (%s) failed to "\
            "implement _apply_operation()" % ( str(self.__class__), ))
    
    """
    ---------------------**********************--------------------
    The following are nodes creators that should be used to create
    new nodes properly.
    """
        
    #NotExpression Creator
    def __invert__(self):
        return NotExpression(self)


    #EquivalanceExpression Creator
    def __eq__(self, other):
        return EquivalanceExpression(self, other)

    def equals(self, other):
        return EquivalanceExpression(self, other)

    #XorExpression Creator
    def __xor__(self, other):
        return XorExpression(self, other)

    def Xor(self, other):
        return XorExpression(self, other)

    #AndExpressionCreator
    #Create a new node iff neither node is an AndNode
    #If we have an "AndNode" already, safe_add new node to the exisiting one.
    def __and__(self, other):
        if (self.getname() != "AndExpression"):  
            if (other.getname() != "AndExpression"):
                #return AndExpression(set([self, other])) #set version
                return AndExpression(list([self, other]))
            else :
                other._add(self)
                self = other
                return self
        else :
            self._add(other)
        return self

    #class method for AndExpression,basically the same logic as above
    '''
    #This section is documented just in case the class method is needed
    #in the future
    def LogicalAnd(self, other):
        if (self.getname() != "AndExpression"):  
            if (other.getname() != "AndExpression"):
                #return AndExpression(set([self, other])) #set version
                return AndExpression(list([self, other]))
            else :
                other._add(self)
                # 0-0 This step is a safety consideration, we can also 
                # use the python add and access the list to make things 
                # faster(not by much I guess)
                self = other
                return self
        else:
            self._add(other)
        return self
    '''
    
    #OrExpressionCreator
    #Create a new node iff neither node is an OrNode
    def __or__(self, other):
        if (self.getname() != "OrExpression"):  
            if (other.getname() != "OrExpression"):
                #return OrExpression(set([self, other])) #set version
                return OrExpression(list([self, other]))
            else :
                other._add(self)
                self = other
                return self
        else :
            self._add(other)
        return self

    '''
        This section is documented just in case the class method is needed
        in the future
    #class method for OrExpression,basically the same logic as above
    def LogicalOr(self, other):
        if (self.getname() != "OrExpression"):  
            if (other.getname() != "OrExpression"):
                #return OrExpression(set([self, other])) #set version
                return OrExpression(list([self, other]))
            else :
                other._add(self)
                # 0-0 This step is a safety consideration, we can also 
                # use the python add and access the list to make things 
                # faster(not by much I guess)
                self = other
                return self
        else:
            self._add(other)
        return self
    '''

"""
---------------------------******************--------------------
The following methods are static methods for nodes creator. Those should
do the exact same thing as the class methods as well as overloaded operators.
"""

# static method for NotExpression creator 
def Not(self):
    return NotExpression(self)

# static method for EquivalenceExpression creator 
def Equivalence(arg1, arg2):
    return EquivalenceExpression(arg1, arg2) 

# static method for XorExpression creator
def Xor(arg1, arg2):
    return XorExpression(arg1, arg2)

# 0-0 add a static method for impies>
def Implies(arg1, arg2):
    return Implication(arg1, arg2)

# static method for AndExpression creator
# create a new node iff neither node is an AndNode

#combine 2 function and name it And()
def And(*argv):
    # 0-0 Do we need to take care of the safety of this method?
    # argsList is a set of LogicalValues, RaiseError if not?
    # checking requires a full loop from my understanding
    # Do we need to take care of empty set of set with length 1?
    argsList = list(argv)
    parent = argsList[0]
    for tmp in argsList:
        if isinstance(tmp, AndExpression):
            parent = tmp
            argList.remove(tmp)
            for target in argsList:
                parent._add(target)
            return parent

    res = AndExpression(list([parent]))
    argsList.remove(parent)
    while (len(argsList) != 0):
        res._add(argsList.pop())
    return res

# static method for OrExpression creator
# create a new node iff neither node is an OrNode, same logic

def Or(*argv):
    argsList = list(argv)
    parent = argsList[0]
    for tmp in argsList:
        if isinstance(tmp, OrExpression):
            parent = tmp
            argList.remove(tmp)
            for target in argsList:
                parent._add(target)
            return parent

    res = OrExpression(list([parent]))
    argsList.remove(parent)   
    while (len(argsList) != 0):
        res._add(argsList.pop())
    return res



# static Method for ExactlyExpression, AtMostExpression and AtLeastExpression
# make it support tuples?
def Exactly(req, argsList):
    result = ExactlyExpression(list(argsList))
    result._args_.insert(0, req)
    return result

def AtMost(req, argsList):
    result = AtMostExpression(list(argsList))
    result._args_.insert(0, req)
    return result

def AtLeast(req, argsList):
    result = AtLeastExpression(list(argsList))
    result._args_.insert(0, req)
    return result


#-------------------------*************------------------------------



class UnaryExpression(LogicalExpressionBase):
    """ 
    An abstract class for NotExpression
    There should only be one child under this kind of nodes
    This class should never be created directly. 
    #0-0
    This is the tempting naming. "args" should be one subclass of UnaryExpression,
    BinaryExperssion or MultiNodesExpression.
    """

    __slots__ = ("_args_",)

    """
    #0-0
    The precedence of an abstract class should not be a concern here, so it will be set
    to zero for now.
    """
    def __init__(self, args):
        self._args_ = args
        #print("The variable is initialized using UnaryExpression")
        #for tracing purpose only, delete later.
        #0-0 

    PRECEDENCE = 0

    def nargs(self):
        return 1

    def getname(self, *arg, **kwd):
        return 'UnaryExpression'

    def _precedence(self):
        return UnaryExpression.PRECEDENCE

    def _to_string(self, values, verbose, smap, compute_values):
        #question: how should this one work in general, though this function should not
        #be called ever imo. 
        #fine to raise this error?
        raise NotImplementedError("Derived expression (%s) failed to "\
                "implement _apply_operation()" % ( str(self.__class__), ))

    def _apply_operation(self):
        raise TypeError("Please use Notexpression instead.")
        #0-0 ok to do this?
        


class NotExpression(UnaryExpression):
        """
        This is the node for a NotExpression, this node should have exactly one child
        """

        __slots__ = ()

        PRECEDENCE = 10
        #This operation should have the highest precedence among all, for now 
        #use 10 and adjust that later 0-0

        def getname(self, *arg, **kwd):
            return 'NotExpression'

        def _precendence(self):
            return Notexpression.PRECEDENCE

        def _to_string(self, values, verbose, smap, compute_values):
            #pass this one for now 0-0
            pass

        def _apply_operetion(self, result):
            """
            result should be a tuple in general
            """
            return not result

class BinaryExpression(LogicalExpressionBase):
    """
    The abstract class for binary expression. This class should never be initialized.
    with __init__ .  largs and rargs are tempting names for its child nodes.
    """
    __slots__ = ("_largs_", "_rargs_",)

    def __init__(self, largs, rargs):
        self._largs_ = largs
        self._rargs_ = rargs
        #
        #print("The variable is initialized using BinaryExpression")
        #for tracing purpose only, delete later.
        #0-0 

    PRECEDENCE = 0
    #As this class should never be used in practice.

    def nargs(self):
        return 2

    def getname(self, *arg, **kwd):
        return 'BinaryExpression'

    def _precedence(self):
        return BinaryExpression.PRECEDENCE

    def _to_string(self, values, verbose, smap, compute_values):
        #question: how should this one work in general, though this function should not
        #be called ever imo. 
        #fine to raise this error?
        raise NotImplementedError("Derived expression (%s) failed to "\
                "implement _apply_operation()" % ( str(self.__class__), ))

    def _apply_operation(self):
        raise TypeError("Please use the approriate binary expression instead.")
        #0-0 fine like this?




class EquivalenceExpression(BinaryExpression):
        """
        This is the node for EquivalanceExpression, this node should have exactly two children
        """

        __slots__ = ()

        PRECEDENCE = 9
        #0-0 not really sure... Is there a reference I can use?

        def getname(self, *arg, **kwd):
            return 'EquivalanceExpression'

        def _precendence(self):
            return EquivalanceExpression.PRECEDENCE

        def _to_string(self, values, verbose, smap, compute_values):
            #pass this one for now 0-0
            pass

        #change it to (self, result):
        def _apply_opeartion(self,res1, res2):
            """
            #0-0 
            """
            return (res1 == res2)


class XorExpression(BinaryExpression):
        """
        This is the node for XorExpression, this node should have exactly two children
        """

        __slots__ = ()

        PRECEDENCE = 9
        #0-0 same as above

        def getname(self, *arg, **kwd):
            return 'XorExpression'

        def _precendence(self):
            return XorExpression.PRECEDENCE

        def _to_string(self, values, verbose, smap, compute_values):
            #pass this one for now 0-0
            pass

        def _apply_operation(self,res1, res2):
            """
            #0-0 
            """
            #return (res1 + res2 == 1)
            return operator.xor(res1, res2)


class Implication(BinaryExpression):
        """
        This is the node for Implication, this node should have exactly two children
        """

        __slots__ = ()

        PRECEDENCE = 9
        #0-0 same as above

        def getname(self, *arg, **kwd):
            return 'Implication'

        def _precendence(self):
            return XorExpression.PRECEDENCE

        def _to_string(self, values, verbose, smap, compute_values):
            #pass this one for now 0-0
            pass

        def _apply_opeartion(self,res1, res2):
            """
            # 0-0 Use Not(a) or B?
            """
            #we may need a get_args function for below 
            #try OrExpression(Not(self._largs_), self._rargs_) 
            return ((not res1) or (res2))


class MultiargsExpression(LogicalExpressionBase):
    """
    The abstract class for MultiargsExpression. This class should never be initialized.
    with __init__ .  argss is a tempting name.
    """

    #args should be a set from Romeo's prototype

    __slots__ = ("_args_")

    def __init__(self, ChildList):
        #self._argss_ =  list([v for v in ChildList]) #if we want the set version
        self._args_ =  list([v for v in ChildList])
        #Not really sure
        #print("The variable is initialized using MultiargsExpression")
        #for tracing purpose only, delete later.
        #0-0 

    PRECEDENCE = 0
    #As this class should never be used in practice.

    def nargs(self):
        return len(self._args_)

    def getname(self, *arg, **kwd):
        return 'MultiNodeExpression'

    def _add(self, other):
        #0-0 a private method that adds another logicalexpression into this node
        #add elements into the list,while not creating a node if they share the same type
        #Always use this safe_add to add elements into a multinode
        '''
        #set version
        if (other.getname() != self.getname()):
            self._argss_.add(other)
        else:
            self._argss_.update(other._argss_)
            #should we remove other in some way here?
        '''
        #list version
        if (type(other) != type(self)):
            self._args_.append(other)
        else:
            self._args_.extend(other._args_) 

    def _precedence(self):
        return MultiargsExpression.PRECEDENCE

    def _to_string(self, values, verbose, smap, compute_values):
        #question: how should this one work in general, though this function should not
        #be called ever imo. 
        #fine to raise this error?
        raise NotImplementedError("Derived expression (%s) failed to "\
                "implement _apply_operation()" % ( str(self.__class__), ))
    #0-0 for debug currently
    def __str__(self):
        res = ""
        for tmp in self._args_ :
            res += (tmp.getname() + " ")
        return res

    def _apply_operation(self):
        raise TypeError("Please use the approriate MultiargsExpression instead.")
        #0-0 fine like this?


class AndExpression(MultiargsExpression):
    """
    This is the node for AndExpression.
    For coding only, given that AndExpression should have & as the
    overloaded operator, it is necessary to perform a check. 
    """

    __slots__ = ()

    PRECEDENCE = 9
    #0-0 same as above

    def getname(self, *arg, **kwd):
        return 'AndExpression'

    def _precendence(self):
        return AndExpression.PRECEDENCE

    def _to_string(self, values, verbose, smap, compute_values):
        #pass this one for now 0-0
        pass

    def _apply_opeartion(self, result):
        """
        #0-0 to be implemented
        """
        if (len(self._args_) != len(res_list)):
            KeyError("Make sure number of truth values matches number"\
             "of children for this node")
        return all(result)


class OrExpression(MultiargsExpression):
    __slots__ = ()

    PRECEDENCE = 9
    #0-0 same as above

    def getname(self, *arg, **kwd):
        return 'OrExpression'

    def _precendence(self):
        return OrExpression.PRECEDENCE

    def _to_string(self, values, verbose, smap, compute_values):
        #pass this one for now 0-0
        pass

    def _apply_operation(self, result):
        """
        #0-0 to be implemented
        """
        if (len(self._args_) != len(res_list)):
            raise KeyError("Make sure number of truth values matches number"\
             "of children for this node")
        return any(result)



'''for Exactly, ...
'''
class Exactly(MultiargsExpression):
    __slots__ = ()

    PRECEDENCE = 9
    #0-0 same as above

    def getname(self, *arg, **kwd):
        return 'ExactlyExpression'

    def _precendence(self):
        return ExactlyExpression.PRECEDENCE

    def _to_string(self, values, verbose, smap, compute_values):
        #pass this one for now 0-0
        pass

    def _apply_operation(self,result):
        """
        #0-0 to be implemented
        """
        if (len(self._args_)-1 != len(res_list)):
            KeyError("Make sure number of truth values matches number"\
             "of children for this node")
        '''
        counter = 0
        for tmp in res_list:
            if(tmp == True):
                counter += 1
        return (counter == self._args_[0])
        '''
        return sum(result) == self._args_[0]


class AtMostExpression(MultiargsExpression):
    __slots__ = ()

    PRECEDENCE = 9
    #0-0 same as above

    def getname(self, *arg, **kwd):
        return 'AtMostExpression'

    def _precendence(self):
        return AtMostExpression.PRECEDENCE

    def _to_string(self, values, verbose, smap, compute_values):
        #pass this one for now 0-0
        pass

    def _apply_operation(self, res_list):
        if (len(self._args_)-1 != len(res_list)):
            KeyError("Make sure number of truth values matches number"\
             "of children for this node")
        counter = 0
        for tmp in res_list:
            if(tmp == True):
                counter += 1
        return (counter >= self._args_[0])

        

class AtLeastExpression(MultiargsExpression):
    __slots__ = ()

    PRECEDENCE = 9
    #0-0 same as above

    def getname(self, *arg, **kwd):
        return 'AtLeastExpression'

    def _precendence(self):
        return AtLeastExpression.PRECEDENCE

    def _to_string(self, values, verbose, smap, compute_values):
        #pass this one for now 0-0
        pass

    def _apply_operation(self, res_list):
        if (len(self._args_)-1 != len(res_list)):
            KeyError("Make sure number of truth values matches number"\
             "of children for this node")
        counter = 0
        for tmp in res_list:
            if(tmp == True):
                counter += 1
        return (counter <= self._args_[0])









