#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from __future__ import division

import inspect
import logging
import six
from copy import deepcopy
from collections import deque

if six.PY2:
    getargspec = inspect.getargspec
else:
    # For our needs, getfullargspec is a drop-in replacement for
    # getargspec (which was removed in Python 3.x)
    getargspec = inspect.getfullargspec

logger = logging.getLogger('pyomo.core')

from .symbol_map import SymbolMap
from . import expr_common as common
from .expr_errors import TemplateExpressionError
from pyomo.common.deprecation import deprecation_warning

from pyomo.core.expr.boolean_value import (
    BooleanValue,)


from pyomo.core.expr.numvalue import (
    nonpyomo_leaf_types,
    native_numeric_types,
    value,)


# NOTE: This module also has dependencies on numeric_expr; however, to
# avoid circular dependencies, we will NOT import them here.  Instead,
# until we can resolve the circular dependencies, they will be injected
# into this module by the .current module (which must be imported
# *after* numeric_expr, logocal_expr, and this module.


#-------------------------------------------------------
#
# Visitor Logic
#
#-------------------------------------------------------

class StreamBasedExpressionVisitor(object):
    """This class implements a generic stream-based expression walker.

    This visitor walks an expression tree using a depth-first strategy
    and generates a full event stream similar to other tree visitors
    (e.g., the expat XML parser).  The following events are triggered
    through callback functions as the traversal enters and leaves nodes
    in the tree:

      initializeWalker(expr) -> walk, result
      enterNode(N1) -> args, data
      {for N2 in args:}
        beforeChild(N1, N2) -> descend, child_result
          enterNode(N2) -> N2_args, N2_data
          [...]
          exitNode(N2, n2_data) -> child_result
        acceptChildResult(N1, data, child_result) -> data
        afterChild(N1, N2) -> None
      exitNode(N1, data) -> N1_result
      finalizeWalker(result) -> result

    Individual event callbacks match the following signatures:

   walk, result = initializeWalker(self, expr):

        initializeWalker() is called to set the walker up and perform
        any preliminary processing on the root node.  The method returns
        a flag indicating if the tree should be walked and a result.  If
        `walk` is True, then result is ignored.  If `walk` is False,
        then `result` is returned as the final result from the walker,
        bypassing all other callbacks (including finalizeResult).

   args, data = enterNode(self, node):

        enterNode() is called when the walker first enters a node (from
        above), and is passed the node being entered.  It is expected to
        return a tuple of child `args` (as either a tuple or list) and a
        user-specified data structure for collecting results.  If None
        is returned for args, the node's args attribute is used for
        expression types and the empty tuple for leaf nodes.  Returning
        None is equivalent to returning (None,None).  If the callback is
        not defined, the default behavior is equivalent to returning
        (None, []).

    node_result = exitNode(self, node, data):

        exitNode() is called after the node is completely processed (as
        the walker returns up the tree to the parent node).  It is
        passed the node and the results data structure (defined by
        enterNode() and possibly further modified by
        acceptChildResult()), and is expected to return the "result" for
        this node.  If not specified, the default action is to return
        the data object from enterNode().

    descend, child_result = beforeChild(self, node, child, child_idx):

        beforeChild() is called by a node for every child before
        entering the child node.  The node, child node, and child index
        (position in the args list from enterNode()) are passed as
        arguments.  beforeChild should return a tuple (descend,
        child_result).  If descend is False, the child node will not be
        entered and the value returned to child_result will be passed to
        the node's acceptChildResult callback.  Returning None is
        equivalent to (True, None).  The default behavior if not
        specified is equivalent to (True, None).

    data = acceptChildResult(self, node, data, child_result, child_idx):

        acceptChildResult() is called for each child result being
        returned to a node.  This callback is responsible for recording
        the result for later processing or passing up the tree.  It is
        passed the node, result data structure (see enterNode()), child
        result, and the child index (position in args from enterNode()).
        The data structure (possibly modified or replaced) must be
        returned.  If acceptChildResult is not specified, it does
        nothing if data is None, otherwise it calls data.append(result).

    afterChild(self, node, child, child_idx):

        afterChild() is called by a node for every child node
        immediately after processing the node is complete before control
        moves to the next child or up to the parent node.  The node,
        child node, an child index (position in args from enterNode())
        are passed, and nothing is returned.  If afterChild is not
        specified, no action takes place.

    finalizeResult(self, result):

        finalizeResult() is called once after the entire expression tree
        has been walked.  It is passed the result returned by the root
        node exitNode() callback.  If finalizeResult is not specified,
        the walker returns the result obtained from the exitNode
        callback on the root node.

    Clients interact with this class by either deriving from it and
    implementing the necessary callbacks (see above), assigning callable
    functions to an instance of this class, or passing the callback
    functions as arguments to this class' constructor.

    """

    # The list of event methods that can either be implemented by
    # derived classes or specified as callback functions to the class
    # constructor:
    client_methods = ('enterNode','exitNode','beforeChild','afterChild',
                      'acceptChildResult','initializeWalker','finalizeResult')
    def __init__(self, **kwds):
        # This is slightly tricky: We want derived classes to be able to
        # override the "None" defaults here, and for keyword arguments
        # to override both.  The hasattr check prevents the "None"
        # defaults from overriding attributes or methods defined on
        # derived classes.
        for field in self.client_methods:
            if field in kwds:
                setattr(self, field, kwds.pop(field))
            elif not hasattr(self, field):
                setattr(self, field, None)
        if kwds:
            raise RuntimeError("Unrecognized keyword arguments: %s" % (kwds,))

        # Handle deprecated APIs
        _fcns = (('beforeChild',2), ('acceptChildResult',3), ('afterChild',2))
        for name, nargs in _fcns:
            fcn = getattr(self, name)
            if fcn is None:
                continue
            _args = getargspec(fcn)
            _self_arg = 1 if inspect.ismethod(fcn) else 0
            if len(_args.args) == nargs + _self_arg and _args.varargs is None:
                deprecation_warning(
                    "Note that the API for the StreamBasedExpressionVisitor "
                    "has changed to include the child index for the %s() "
                    "method.  Please update your walker callbacks." % (name,))
                def wrap(fcn, nargs):
                    def wrapper(*args):
                        return fcn(*args[:nargs])
                    return wrapper
                setattr(self, name, wrap(fcn, nargs))


    def walk_expression(self, expr):
        """Walk an expression, calling registered callbacks.
        """
        #
        # This walker uses a linked list to store the stack (instead of
        # an array).  The nodes of the linked list are 6-member tuples:
        #
        #    ( pointer to parent,
        #      expression node,
        #      tuple/list of child nodes (arguments),
        #      number of child nodes (arguments),
        #      data object to aggregate results from child nodes,
        #      current child node index )
        #
        # The walker only needs a single pointer to the end of the list
        # (ptr).  The beginning of the list is indicated by a None
        # parent pointer.
        #
        if self.initializeWalker is not None:
            walk, result = self.initializeWalker(expr)
            if not walk:
                return result
        if self.enterNode is not None:
            tmp = self.enterNode(expr)
            if tmp is None:
                args = data = None
            else:
                args, data = tmp
        else:
            args = None
            data = []
        if args is None:
            if type(expr) in nonpyomo_leaf_types \
                    or not expr.is_expression_type():
                args = ()
            else:
                args = expr.args
        if hasattr(args, '__enter__'):
            args.__enter__()
        node = expr
        # Note that because we increment child_idx just before fetching
        # the child node, it must be initialized to -1, and ptr[3] must
        # always be *one less than* the number of arguments
        child_idx = -1
        ptr = (None, node, args, len(args)-1, data, child_idx)

        try:
            while 1:
                if child_idx < ptr[3]:
                    # Increment the child index pointer here for
                    # consistency.  Note that this means that for the bulk
                    # of the time, 'child_idx' will not match the value of
                    # ptr[5].  This provides a modest performance
                    # improvement, as we only have to recreate the ptr tuple
                    # just before we descend further into the tree (i.e., we
                    # avoid recreating the tuples for the special case where
                    # beforeChild indicates that we should not descend
                    # further).
                    child_idx += 1
                    # This node still has children to process
                    child = ptr[2][child_idx]

                    # Notify this node that we are about to descend into a
                    # child.
                    if self.beforeChild is not None:
                        tmp = self.beforeChild(node, child, child_idx)
                        if tmp is None:
                            descend = True
                            child_result = None
                        else:
                            descend, child_result = tmp
                        if not descend:
                            # We are aborting processing of this child node.
                            # Tell this node to accept the child result and
                            # we will move along
                            if self.acceptChildResult is not None:
                                data = self.acceptChildResult(
                                    node, data, child_result, child_idx)
                            elif data is not None:
                                data.append(child_result)
                            # And let the node know that we are done with a
                            # child node
                            if self.afterChild is not None:
                                self.afterChild(node, child, child_idx)
                            # Jump to the top to continue processing the
                            # next child node
                            continue

                    # Update the child argument counter in the stack.
                    # Because we are using tuples, we need to recreate the
                    # "ptr" object (linked list node)
                    ptr = ptr[:4] + (data, child_idx,)

                    # We are now going to actually enter this node.  The
                    # node will tell us the list of its child nodes that we
                    # need to process
                    if self.enterNode is not None:
                        tmp = self.enterNode(child)
                        if tmp is None:
                            args = data = None
                        else:
                            args, data = tmp
                    else:
                        args = None
                        data = []
                    if args is None:
                        if type(child) in nonpyomo_leaf_types \
                           or not child.is_expression_type():
                            # Leaves (either non-pyomo types or
                            # non-Expressions) have no child arguments, so
                            # are just put on the stack
                            args = ()
                        else:
                            args = child.args
                    if hasattr(args, '__enter__'):
                        args.__enter__()
                    node = child
                    child_idx = -1
                    ptr = (ptr, node, args, len(args)-1, data, child_idx)

                else: # child_idx == ptr[3]:
                    # We are done with this node.  Call exitNode to compute
                    # any result
                    if hasattr(ptr[2], '__exit__'):
                        ptr[2].__exit__(None, None, None)
                    if self.exitNode is not None:
                        node_result = self.exitNode(node, data)
                    else:
                        node_result = data

                    # Pop the node off the linked list
                    ptr = ptr[0]
                    # If we have returned to the beginning, return the final
                    # answer
                    if ptr is None:
                        if self.finalizeResult is not None:
                            return self.finalizeResult(node_result)
                        else:
                            return node_result
                    # Not done yet, update node to point to the new active
                    # node
                    node, child = ptr[1], node
                    data = ptr[4]
                    child_idx = ptr[5]

                    # We need to alert the node to accept the child's result:
                    if self.acceptChildResult is not None:
                        data = self.acceptChildResult(
                            node, data, node_result, child_idx)
                    elif data is not None:
                        data.append(node_result)

                    # And let the node know that we are done with a child node
                    if self.afterChild is not None:
                        self.afterChild(node, child, child_idx)

        finally:
            while ptr is not None:
                if hasattr(ptr[2], '__exit__'):
                        ptr[2].__exit__(None, None, None)
                ptr = ptr[0]

class SimpleExpressionVisitor(object):

    """
    Note:
        This class is a customization of the PyUtilib :class:`SimpleVisitor
        <pyutilib.misc.visitor.SimpleVisitor>` class that is tailored
        to efficiently walk Pyomo expression trees.  However, this class
        is not a subclass of the PyUtilib :class:`SimpleVisitor
        <pyutilib.misc.visitor.SimpleVisitor>` class because all key methods
        are reimplemented.
    """

    def visit(self, node):  #pragma: no cover
        """
        Visit a node in an expression tree and perform some operation on
        it.

        This method should be over-written by a user
        that is creating a sub-class.

        Args:
            node: a node in an expression tree

        Returns:
            nothing
        """
        pass

    def finalize(self):     #pragma: no cover
        """
        Return the "final value" of the search.

        The default implementation returns :const:`None`, because
        the traditional visitor pattern does not return a value.

        Returns:
            The final value after the search.  Default is :const:`None`.
        """
        pass

    def xbfs(self, node):
        """
        Breadth-first search of an expression tree,
        except that leaf nodes are immediately visited.

        Note:
            This method has the same functionality as the
            PyUtilib :class:`SimpleVisitor.xbfs <pyutilib.misc.visitor.SimpleVisitor.xbfs>`
            method.  The difference is that this method
            is tailored to efficiently walk Pyomo expression trees.

        Args:
            node: The root node of the expression tree that is searched.

        Returns:
            The return value is determined by the :func:`finalize` function,
            which may be defined by the user.  Defaults to :const:`None`.
        """
        dq = deque([node])
        while dq:
            current = dq.popleft()
            self.visit(current)
            #for c in self.children(current):
            for c in current.args:
                #if self.is_leaf(c):
                if c.__class__ in nonpyomo_leaf_types or not c.is_expression_type() or c.nargs() == 0:
                    self.visit(c)
                else:
                    dq.append(c)
        return self.finalize()

    def xbfs_yield_leaves(self, node):
        """
        Breadth-first search of an expression tree, except that
        leaf nodes are immediately visited.

        Note:
            This method has the same functionality as the
            PyUtilib :class:`SimpleVisitor.xbfs_yield_leaves <pyutilib.misc.visitor.SimpleVisitor.xbfs_yield_leaves>`
            method.  The difference is that this method
            is tailored to efficiently walk Pyomo expression trees.

        Args:
            node: The root node of the expression tree
                that is searched.

        Returns:
            The return value is determined by the :func:`finalize` function,
            which may be defined by the user.  Defaults to :const:`None`.
        """
        #
        # If we start with a leaf, then yield it and stop iteration
        #
        if node.__class__ in nonpyomo_leaf_types or not node.is_expression_type() or node.nargs() == 0:
            ans = self.visit(node)
            if not ans is None:
                yield ans
            return
        #
        # Iterate through the tree.
        #
        dq = deque([node])
        while dq:
            current = dq.popleft()
            #self.visit(current)
            #for c in self.children(current):
            for c in current.args:
                #if self.is_leaf(c):
                if c.__class__ in nonpyomo_leaf_types or not c.is_expression_type() or c.nargs() == 0:
                    ans = self.visit(c)
                    if not ans is None:
                        yield ans
                else:
                    dq.append(c)


class ExpressionValueVisitor(object):
    """
    Note:
        This class is a customization of the PyUtilib :class:`ValueVisitor
        <pyutilib.misc.visitor.ValueVisitor>` class that is tailored
        to efficiently walk Pyomo expression trees.  However, this class
        is not a subclass of the PyUtilib :class:`ValueVisitor
        <pyutilib.misc.visitor.ValueVisitor>` class because all key methods
        are reimplemented.
    """

    def visit(self, node, values):  #pragma: no cover
        """
        Visit a node in a tree and compute its value using
        the values of its children.

        This method should be over-written by a user
        that is creating a sub-class.

        Args:
            node: a node in a tree
            values: a list of values of this node's children

        Returns:
            The *value* for this node, which is computed using :attr:`values`
        """
        pass

    def visiting_potential_leaf(self, node):    #pragma: no cover
        """
        Visit a node and return its value if it is a leaf.

        Note:
            This method needs to be over-written for a specific
            visitor application.

        Args:
            node: a node in a tree

        Returns:
            A tuple: ``(flag, value)``.   If ``flag`` is False,
            then the node is not a leaf and ``value`` is :const:`None`.
            Otherwise, ``value`` is the computed value for this node.
        """
        raise RuntimeError("The visiting_potential_leaf method needs to be defined.")

    def finalize(self, ans):    #pragma: no cover
        """
        This method defines the return value for the search methods
        in this class.

        The default implementation returns the value of the
        initial node (aka the root node), because
        this visitor pattern computes and returns value for each
        node to enable the computation of this value.

        Args:
            ans: The final value computed by the search method.

        Returns:
            The final value after the search. Defaults to simply
            returning :attr:`ans`.
        """
        return ans

    def dfs_postorder_stack(self, node):
        """
        Perform a depth-first search in postorder using a stack
        implementation.

        Note:
            This method has the same functionality as the
            PyUtilib :class:`ValueVisitor.dfs_postorder_stack <pyutilib.misc.visitor.ValueVisitor.dfs_postorder_stack>`
            method.  The difference is that this method
            is tailored to efficiently walk Pyomo expression trees.

        Args:
            node: The root node of the expression tree
                that is searched.

        Returns:
            The return value is determined by the :func:`finalize` function,
            which may be defined by the user.
        """
        flag, value = self.visiting_potential_leaf(node)
        if flag:
            return value
        #_stack = [ (node, self.children(node), 0, len(self.children(node)), [])]
        _stack = [ (node, node._args_, 0, node.nargs(), [])]
        #
        # Iterate until the stack is empty
        #
        # Note: 1 is faster than True for Python 2.x
        #
        while 1:
            #
            # Get the top of the stack
            #   _obj        Current expression object
            #   _argList    The arguments for this expression objet
            #   _idx        The current argument being considered
            #   _len        The number of arguments
            #   _result     The return values
            #
            _obj, _argList, _idx, _len, _result = _stack.pop()
            #
            # Iterate through the arguments
            #
            while _idx < _len:
                _sub = _argList[_idx]
                _idx += 1
                flag, value = self.visiting_potential_leaf(_sub)
                if flag:
                    _result.append( value )
                else:
                    #
                    # Push an expression onto the stack
                    #
                    _stack.append( (_obj, _argList, _idx, _len, _result) )
                    _obj                    = _sub
                    #_argList                = self.children(_sub)
                    _argList                = _sub._args_
                    _idx                    = 0
                    _len                    = _sub.nargs()
                    _result                 = []
            #
            # Process the current node
            #
            ans = self.visit(_obj, _result)
            if _stack:
                #
                # "return" the recursion by putting the return value on the end of the results stack
                #
                _stack[-1][-1].append( ans )
            else:
                return self.finalize(ans)

def replace_expressions(expr,
                        substitution_map,
                        descend_into_named_expressions=True,
                        remove_named_expressions=False):
    """

    Parameters
    ----------
    expr : Pyomo expression
       The source expression
    substitution_map : dict
       A dictionary mapping object ids in the source to the replacement objects.
    descend_into_named_expressions : bool
       True if replacement should go into named expression objects, False to halt at
       a named expression
    remove_named_expressions : bool
       True if the named expressions should be replaced with a standard expression,
       and False if the named expression should be left in place

    Returns
    -------
       Pyomo expression : returns the new expression object
    """
    new_expr = ExpressionReplacementVisitor(
            substitute=substitution_map,
            descend_into_named_expressions=descend_into_named_expressions,
            remove_named_expressions=remove_named_expressions
            ).dfs_postorder_stack(expr)
    return new_expr


class ExpressionReplacementVisitor(object):
    """
    Note:
        This class is a customization of the PyUtilib :class:`ValueVisitor
        <pyutilib.misc.visitor.ValueVisitor>` class that is tailored
        to support replacement of sub-trees in a Pyomo expression
        tree.  However, this class is not a subclass of the PyUtilib
        :class:`ValueVisitor <pyutilib.misc.visitor.ValueVisitor>`
        class because all key methods are reimplemented.
    """

    def __init__(self,
                 substitute=None,
                 descend_into_named_expressions=True,
                 remove_named_expressions=False):
        """
        Contruct a visitor that is tailored to support the
        replacement of sub-trees in a pyomo expression tree.

        Args:
            memo (dict): A dictionary mapping object ids to
                objects.  This dictionary has the same semantics as
                the memo object used with ``copy.deepcopy``.  Defaults
                to None, which indicates that no user-defined
                dictionary is used.
        """
        self.enter_named_expr = descend_into_named_expressions
        self.rm_named_expr = remove_named_expressions
        if substitute is None:
            self.substitute = {}
        else:
            self.substitute = substitute

    def visit(self, node, values):
        """
        Visit and clone nodes that have been expanded.

        Note:
            This method normally does not need to be re-defined
            by a user.

        Args:
            node: The node that will be cloned.
            values (list): The list of child nodes that have been
                cloned.  These values are used to define the
                cloned node.

        Returns:
            The cloned node.  Default is to simply return the node.
        """
        return node

    def visiting_potential_leaf(self, node):    #pragma: no cover
        """
        Visit a node and return a cloned node if it is a leaf.

        Note:
            This method needs to be over-written for a specific
            visitor application.

        Args:
            node: a node in a tree

        Returns:
            A tuple: ``(flag, value)``.   If ``flag`` is False,
            then the node is not a leaf and ``value`` is :const:`None`.
            Otherwise, ``value`` is a cloned node.
        """
        _id = id(node)
        if _id in self.substitute:
            return True, self.substitute[_id]
        elif type(node) in nonpyomo_leaf_types or not node.is_expression_type():
            return True, node
        elif not self.enter_named_expr and node.is_named_expression_type():
            return True, node
        else:
            return False, None

    def finalize(self, ans):
        """
        This method defines the return value for the search methods
        in this class.

        The default implementation returns the value of the
        initial node (aka the root node), because
        this visitor pattern computes and returns value for each
        node to enable the computation of this value.

        Args:
            ans: The final value computed by the search method.

        Returns:
            The final value after the search. Defaults to simply
            returning :attr:`ans`.
        """
        return ans

    def construct_node(self, node, values):
        """
        Call the expression create_node_with_local_data() method.
        """
        return node.create_node_with_local_data( tuple(values) )

    def dfs_postorder_stack(self, node):
        """
        Perform a depth-first search in postorder using a stack
        implementation.

        This method replaces subtrees.  This method detects if the
        :func:`visit` method returns a different object.  If so, then
        the node has been replaced and search process is adapted
        to replace all subsequent parent nodes in the tree.

        Note:
            This method has the same functionality as the
            PyUtilib :class:`ValueVisitor.dfs_postorder_stack <pyutilib.misc.visitor.ValueVisitor.dfs_postorder_stack>`
            method that is tailored to support the
            replacement of sub-trees in a Pyomo expression tree.

        Args:
            node: The root node of the expression tree
                that is searched.

        Returns:
            The return value is determined by the :func:`finalize` function,
            which may be defined by the user.
        """
        if node.__class__ is LinearExpression:
            _argList = [node.constant] + node.linear_coefs + node.linear_vars
            _len = len(_argList)
            _stack = [ (node, _argList, 0, _len, [False])]
        else:
            flag, value = self.visiting_potential_leaf(node)
            if flag:
                return value
            _stack = [ (node, node._args_, 0, node.nargs(), [False])]
        #
        # Iterate until the stack is empty
        #
        # Note: 1 is faster than True for Python 2.x
        #
        while 1:
            #
            # Get the top of the stack
            #   _obj        Current expression object
            #   _argList    The arguments for this expression objet
            #   _idx        The current argument being considered
            #   _len        The number of arguments
            #   _result     The 'dirty' flag followed by return values
            #
            _obj, _argList, _idx, _len, _result = _stack.pop()
            #
            # Iterate through the arguments, entering each one
            #
            while _idx < _len:
                _sub = _argList[_idx]
                _idx += 1
                flag, value = self.visiting_potential_leaf(_sub)
                if flag:
                    if id(value) != id(_sub):
                        _result[0] = True
                    _result.append( value )
                else:
                    #
                    # Push an expression onto the stack
                    #
                    _stack.append( (_obj, _argList, _idx, _len, _result) )
                    _obj = _sub
                    _idx = 0
                    _result = [False]
                    if _sub.__class__ is LinearExpression:
                        _argList = [_sub.constant] + _sub.linear_coefs \
                                   + _sub.linear_vars
                        _len = len(_argList)
                    else:
                        _argList = _sub._args_
                        _len = _sub.nargs()
            #
            # Finalize (exit) the current node
            #
            # If the user has defined a visit() function in a
            # subclass, then call that function.  But if the user
            # hasn't created a new class and we need to, then
            # call the ExpressionReplacementVisitor.visit() function.
            #
            ans = self.visit(_obj, _result[1:])
            if ans.is_named_expression_type():
                if self.rm_named_expr:
                    ans = _result[1]
                    _result[0] = True
                else:
                    _result[0] = False
                    assert(len(_result) == 2)
                    ans.expr = _result[1]
            elif _result[0]:
                if ans.__class__ is LinearExpression:
                    ans = _result[1]
                    nterms = (len(_result)-2)//2
                    for i in range(nterms):
                        ans += _result[2+i]*_result[2+i+nterms]
                if id(ans) == id(_obj):
                    ans = self.construct_node(_obj, _result[1:])
                if ans.__class__ is MonomialTermExpression:
                    # CDL This code wass trying to determine if we needed to change the MonomialTermExpression
                    # to a ProductExpression, but it fails for the case of a MonomialExpression
                    # that has its rhs Var replaced with another MonomialExpression (and might
                    # fail for other cases as well.
                    # Rather than trying to update the logic to catch all cases, I am choosing
                    # to execute the actual product operator code instead to ensure things are
                    # consistent
                    # See WalkerTests.test_replace_expressions_with_monomial_term  in test_expr_pyomo5.py
                    # to see the behavior
                    # if ( ( ans._args_[0].__class__ not in native_numeric_types
                    #        and ans._args_[0].is_potentially_variable )
                    #      or
                    #      ( ans._args_[1].__class__ in native_numeric_types
                    #        or not ans._args_[1].is_potentially_variable() ) ):
                    #     ans.__class__ = ProductExpression
                    ans = ans._args_[0] * ans._args_[1]
                elif ans.__class__ in NPV_expression_types:
                    # For simplicity, not-potentially-variable expressions are
                    # replaced with their potentially variable counterparts.
                    ans = ans.create_potentially_variable_object()
            elif id(ans) != id(_obj):
                _result[0] = True

            if _stack:
                if _result[0]:
                    _stack[-1][-1][0] = True
                #
                # "return" the recursion by putting the return value on
                # the end of the results stack
                #
                _stack[-1][-1].append( ans )
            else:
                return self.finalize(ans)


#-------------------------------------------------------
#
# Functions used to process expression trees
#
#-------------------------------------------------------

# =====================================================
#  clone_expression
# =====================================================

def clone_expression(expr, substitute=None):
    """A function that is used to clone an expression.

    Cloning is equivalent to calling ``copy.deepcopy`` with no Block
    scope.  That is, the expression tree is duplicated, but no Pyomo
    components (leaf nodes *or* named Expressions) are duplicated.

    Args:
        expr: The expression that will be cloned.
        substitute (dict): A dictionary mapping object ids to
            objects. This dictionary has the same semantics as
            the memo object used with ``copy.deepcopy``. Defaults
            to None, which indicates that no user-defined
            dictionary is used.

    Returns:
        The cloned expression.

    """
    clone_counter._count += 1
    memo = {'__block_scope__': {id(None): False}}
    if substitute:
        memo.update(substitute)
    return deepcopy(expr, memo)


# =====================================================
#  sizeof_expression
# =====================================================

def sizeof_expression(expr):
    """
    Return the number of nodes in the expression tree.

    Args:
        expr: The root node of an expression tree.

    Returns:
        A non-negative integer that is the number of
        interior and leaf nodes in the expression tree.
    """
    def enter(node):
        return None, 1
    def accept(node, data, child_result, child_idx):
        return data + child_result
    return StreamBasedExpressionVisitor(
        enterNode=enter,
        acceptChildResult=accept,
    ).walk_expression(expr)

# =====================================================
#  evaluate_expression
# =====================================================

class _EvaluationVisitor(ExpressionValueVisitor):

    def visit(self, node, values):
        """ Visit nodes that have been expanded """
        return node._apply_operation(values)

    def visiting_potential_leaf(self, node):
        """
        Visiting a potential leaf.

        Return True if the node is not expanded.
        """
        if node.__class__ in nonpyomo_leaf_types:
            return True, node

        if node.is_expression_type():
            return False, None

        if node.is_numeric_type():
            return True, value(node)
        elif node.is_logical_type():
            return True, value(node)
        else:
            return True, node




class FixedExpressionError(Exception):

    def __init__(self, *args, **kwds):
        super(FixedExpressionError, self).__init__(*args, **kwds)


class NonConstantExpressionError(Exception):

    def __init__(self, *args, **kwds):
        super(NonConstantExpressionError, self).__init__(*args, **kwds)


class _EvaluateConstantExpressionVisitor(ExpressionValueVisitor):

    def visit(self, node, values):
        """ Visit nodes that have been expanded """
        return node._apply_operation(values)

    def visiting_potential_leaf(self, node):
        """
        Visiting a potential leaf.

        Return True if the node is not expanded.
        """
        if node.__class__ in nonpyomo_leaf_types:
            return True, node

        if node.is_expression_type():
            return False, None

        if node.is_numeric_type():
            # Get the object value.  This will also cause templates to
            # raise TemplateExpressionErrors
            try:
                val = value(node)
            except TemplateExpressionError:
                raise
            except:
                # Uninitialized Var/Param objects should be given the
                # opportunity to map the error to a NonConstant / Fixed
                # expression error
                if not node.is_fixed():
                    raise NonConstantExpressionError()
                if not node.is_constant():
                    raise FixedExpressionError()
                raise

            if not node.is_fixed():
                raise NonConstantExpressionError()
            if not node.is_constant():
                raise FixedExpressionError()
            return True, val

        return True, node


def evaluate_expression(exp, exception=True, constant=False):
    """Evaluate the value of the expression.

    Args:
        expr: The root node of an expression tree.
        exception (bool): A flag that indicates whether
            exceptions are raised.  If this flag is
            :const:`False`, then an exception that
            occurs while evaluating the expression
            is caught and the return value is :const:`None`.
            Default is :const:`True`.
        constant (bool): If True, constant expressions are
            evaluated and returned but nonconstant expressions
            raise either FixedExpressionError or
            NonconstantExpressionError (default=False).

    Returns:
        A floating point value if the expression evaluates
        normally, or :const:`None` if an exception occurs
        and is caught.

    """
    if constant:
        visitor = _EvaluateConstantExpressionVisitor()
    else:
        visitor = _EvaluationVisitor()
    try:
        return visitor.dfs_postorder_stack(exp)

    except ( TemplateExpressionError, ValueError, TypeError,
             NonConstantExpressionError, FixedExpressionError ):
        # Errors that we want to be able to suppress:
        #
        #   TemplateExpressionError: raised when generating expression
        #      templates
        #   FixedExpressionError, NonConstantExpressionError: raised
        #      when processing expressions that are expected to be fixed
        #      (e.g., indices)
        #   ValueError: "standard" expression value errors
        #   TypeError: This can be raised in Python3 when evaluating a
        #      operation returns a complex number (e.g., sqrt(-1))
        if exception:
            raise
        return None


# =====================================================
#  identify_components
# =====================================================

class _ComponentVisitor(SimpleExpressionVisitor):

    def __init__(self, types):
        self.seen = set()
        if types.__class__ is set:
            self.types = types
        else:
            self.types = set(types)

    def visit(self, node):
        if node.__class__ in self.types:
            if id(node) in self.seen:
                return
            self.seen.add(id(node))
            return node


def identify_components(expr, component_types):
    """
    A generator that yields a sequence of nodes
    in an expression tree that belong to a specified set.

    Args:
        expr: The root node of an expression tree.
        component_types (set or list): A set of class
            types that will be matched during the search.

    Yields:
        Each node that is found.
    """
    #
    # OPTIONS:
    # component_types - set (or list) if class types to find
    # in the expression.
    #
    visitor = _ComponentVisitor(component_types)
    for v in visitor.xbfs_yield_leaves(expr):
        yield v


# =====================================================
#  identify_variables
# =====================================================

class _VariableVisitor(SimpleExpressionVisitor):

    def __init__(self):
        self.seen = set()

    def visit(self, node):
        if node.__class__ in nonpyomo_leaf_types:
            return

        if node.is_variable_type():
            if id(node) in self.seen:
                return
            self.seen.add(id(node))
            return node

        if node.is_expression_type() and isinstance(node, LinearExpression):
            if id(node) in self.seen:
                return
            self.seen.add(id(node))

            def unique_vars_generator():
                for var in node.linear_vars:
                    if id(var) in self.seen:
                        continue
                    self.seen.add(id(var))
                    yield var
            return tuple(v for v in unique_vars_generator())


def identify_variables(expr, include_fixed=True):
    """
    A generator that yields a sequence of variables
    in an expression tree.

    Args:
        expr: The root node of an expression tree.
        include_fixed (bool): If :const:`True`, then
            this generator will yield variables whose
            value is fixed.  Defaults to :const:`True`.

    Yields:
        Each variable that is found.
    """
    visitor = _VariableVisitor()
    if include_fixed:
        for v in visitor.xbfs_yield_leaves(expr):
            if isinstance(v, tuple):
                for v_i in v:
                    yield v_i
            else:
                yield v
    else:
        for v in visitor.xbfs_yield_leaves(expr):
            if isinstance(v, tuple):
                for v_i in v:
                    if not v_i.is_fixed():
                        yield v_i
            else:
                if not v.is_fixed():
                    yield v


# =====================================================
#  identify_mutable_parameters
# =====================================================

class _MutableParamVisitor(SimpleExpressionVisitor):

    def __init__(self):
        self.seen = set()

    def visit(self, node):
        if node.__class__ in nonpyomo_leaf_types:
            return

        # TODO: Confirm that this has the right semantics
        if not node.is_variable_type() and node.is_fixed():
            if id(node) in self.seen:
                return
            self.seen.add(id(node))
            return node


def identify_mutable_parameters(expr):
    """
    A generator that yields a sequence of mutable
    parameters in an expression tree.

    Args:
        expr: The root node of an expression tree.

    Yields:
        Each mutable parameter that is found.
    """
    visitor = _MutableParamVisitor()
    for v in visitor.xbfs_yield_leaves(expr):
        yield v


# =====================================================
#  polynomial_degree
# =====================================================

class _PolynomialDegreeVisitor(ExpressionValueVisitor):

    def visit(self, node, values):
        """ Visit nodes that have been expanded """
        return node._compute_polynomial_degree(values)

    def visiting_potential_leaf(self, node):
        """
        Visiting a potential leaf.

        Return True if the node is not expanded.
        """
        if node.__class__ in nonpyomo_leaf_types:
            return True, 0

        if node.is_expression_type():
            return False, None

        if node.is_numeric_type():
            return True, 0 if node.is_fixed() else 1
        else:
            return True, node


def polynomial_degree(node):
    """
    Return the polynomial degree of the expression.

    Args:
        node: The root node of an expression tree.

    Returns:
        A non-negative integer that is the polynomial
        degree if the expression is polynomial, or :const:`None` otherwise.
    """
    visitor = _PolynomialDegreeVisitor()
    return visitor.dfs_postorder_stack(node)


# =====================================================
#  _expression_is_fixed
# =====================================================

class _IsFixedVisitor(ExpressionValueVisitor):
    """
    NOTE: This doesn't check if combiner logic is
    all or any and short-circuit the test.  It's
    not clear that that is an important optimization.
    """

    def visit(self, node, values):
        """ Visit nodes that have been expanded """
        return node._is_fixed(values)

    def visiting_potential_leaf(self, node):
        """
        Visiting a potential leaf.

        Return True if the node is not expanded.
        """
        if node.__class__ in nonpyomo_leaf_types:
            return True, True

        elif node.is_expression_type():
            return False, None

        elif node.is_numeric_type():
            return True, node.is_fixed()

        return True, node


def _expression_is_fixed(node):
    """
    Return the polynomial degree of the expression.

    Args:
        node: The root node of an expression tree.

    Returns:
        A non-negative integer that is the polynomial
        degree if the expression is polynomial, or :const:`None` otherwise.
    """
    visitor = _IsFixedVisitor()
    return visitor.dfs_postorder_stack(node)


# =====================================================
#  expression_to_string
# =====================================================

class _ToStringVisitor(ExpressionValueVisitor):

    def __init__(self, verbose, smap, compute_values):
        super(_ToStringVisitor, self).__init__()
        self.verbose = verbose
        self.smap = smap
        self.compute_values = compute_values

    def visit(self, node, values):
        """ Visit nodes that have been expanded """
        tmp = []
        for i,val in enumerate(values):
            arg = node._args_[i]

            if arg is None:
                tmp.append('Undefined')                 # TODO: coverage
            elif arg.__class__ in native_numeric_types:
                tmp.append(val)
            elif arg.__class__ in nonpyomo_leaf_types:
                tmp.append("'{0}'".format(val))
            else:
                parens = False
                if not self.verbose and arg.is_expression_type():
                    if node._precedence() < arg._precedence():
                        parens = True
                    elif node._precedence() == arg._precedence():
                        if i == 0:
                            parens = node._associativity() != 1
                        elif i == len(node._args_)-1:
                            parens = node._associativity() != -1
                        else:
                            parens = True
                if parens:
                    tmp.append("({0})".format(val))
                else:
                    tmp.append(val)

        return node._to_string(tmp, self.verbose, self.smap, self.compute_values)

    def visiting_potential_leaf(self, node):
        """
        Visiting a potential leaf.

        Return True if the node is not expanded.
        """
        if node is None:
            return True, None                           # TODO: coverage

        if node.__class__ in nonpyomo_leaf_types:
            return True, str(node)

        if node.is_expression_type():
            return False, None

        if node.is_variable_type():
            if not node.fixed:
                return True, node.to_string(verbose=self.verbose, smap=self.smap, compute_values=False)
            return True, node.to_string(verbose=self.verbose, smap=self.smap, compute_values=self.compute_values)

        if hasattr(node, 'to_string'):
            return True, node.to_string(verbose=self.verbose, smap=self.smap, compute_values=self.compute_values)
        else:
            return True, str(node)


def expression_to_string(expr, verbose=None, labeler=None, smap=None, compute_values=False):
    """
    Return a string representation of an expression.

    Args:
        expr: The root node of an expression tree.
        verbose (bool): If :const:`True`, then the output is
            a nested functional form.  Otherwise, the output
            is an algebraic expression.  Default is :const:`False`.
        labeler:  If specified, this labeler is used to label
            variables in the expression.
        smap:  If specified, this :class:`SymbolMap <pyomo.core.expr.symbol_map.SymbolMap>` is
            used to cache labels.
        compute_values (bool): If :const:`True`, then
            parameters and fixed variables are evaluated before the
            expression string is generated.  Default is :const:`False`.

    Returns:
        A string representation for the expression.
    """
    verbose = common.TO_STRING_VERBOSE if verbose is None else verbose
    #
    # Setup the symbol map
    #
    if labeler is not None:
        if smap is None:
            smap = SymbolMap()
        smap.default_labeler = labeler
    #
    # Create and execute the visitor pattern
    #
    visitor = _ToStringVisitor(verbose, smap, compute_values)
    return visitor.dfs_postorder_stack(expr)
