#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


import inspect
import logging
import sys
from copy import deepcopy
from collections import deque

logger = logging.getLogger('pyomo.core')

from pyomo.common.deprecation import deprecated, deprecation_warning
from pyomo.common.errors import DeveloperError, TemplateExpressionError
from pyomo.common.numeric_types import (
    nonpyomo_leaf_types,
    native_types,
    native_numeric_types,
    value,
)
import pyomo.core.expr.expr_common as common
from pyomo.core.expr.symbol_map import SymbolMap

try:
    # sys._getframe is slightly faster than inspect's currentframe, but
    # is not guaranteed to exist everywhere
    currentframe = sys._getframe
except AttributeError:
    currentframe = inspect.currentframe


def get_stack_depth():
    n = -1  # skip *this* frame in the count
    f = currentframe()
    while f is not None:
        n += 1
        f = f.f_back
    return n


# For efficiency, we want to run recursively, but don't want to hit
# Python's recursion limit (because that would be difficult to recover
# from cleanly).  However, there is a non-trivial cost to determine the
# current stack depth - and we don't want to hit that for every call.
# Instead, we will assume that the walker is always called with at
# least RECURSION_LIMIT frames available on the stack.  When we hit the
# end of that limit, we will actually check how much space is left on
# the stack and run recursively until only 2*RECURSION_LIMIT frames are
# left.  For the vast majority of well-formed expressions this approach
# avoids a somewhat costly call to get_stack_depth, but still catches
# the vast majority of cases that could generate a recursion error.
RECURSION_LIMIT = 50


class RevertToNonrecursive(Exception):
    pass


# NOTE: This module also has dependencies on numeric_expr; however, to
# avoid circular dependencies, we will NOT import them here.  Instead,
# until we can resolve the circular dependencies, they will be injected
# into this module by the .current module (which must be imported
# *after* numeric_expr, logocal_expr, and this module.


# -------------------------------------------------------
#
# Visitor Logic
#
# -------------------------------------------------------


class StreamBasedExpressionVisitor(object):
    """This class implements a generic stream-based expression walker.

    This visitor walks an expression tree using a depth-first strategy
    and generates a full event stream similar to other tree visitors
    (e.g., the expat XML parser).  The following events are triggered
    through callback functions as the traversal enters and leaves nodes
    in the tree:

    ::

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
    # constructor.
    #
    # This is a dict mapping the callback name to a single character
    # that we can use to classify the set of callbacks used by a
    # particular Visitor (we define special-purpose node processors for
    # certain common combinations).  For example, a 'bex' visitor is one
    # that supports beforeChild, enterNode, and exitNode, but NOT
    # afterChild or acceptChildResult.
    client_methods = {
        'enterNode': 'e',
        'exitNode': 'x',
        'beforeChild': 'b',
        'afterChild': 'a',
        'acceptChildResult': 'c',
        'initializeWalker': '',
        'finalizeResult': '',
    }

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
        _fcns = (('beforeChild', 2), ('acceptChildResult', 3), ('afterChild', 2))
        for name, nargs in _fcns:
            fcn = getattr(self, name)
            if fcn is None:
                continue
            _args = inspect.getfullargspec(fcn)
            _self_arg = 1 if inspect.ismethod(fcn) else 0
            if len(_args.args) == nargs + _self_arg and _args.varargs is None:
                deprecation_warning(
                    "Note that the API for the StreamBasedExpressionVisitor "
                    "has changed to include the child index for the %s() "
                    "method.  Please update your walker callbacks." % (name,),
                    version='5.7.0',
                )

                def wrap(fcn, nargs):
                    def wrapper(*args):
                        return fcn(*args[:nargs])

                    return wrapper

                setattr(self, name, wrap(fcn, nargs))

        self.recursion_stack = None

        # Set up the custom recursive node handler function (customized
        # for the specific set of callbacks that are defined for this
        # class instance).
        recursive_node_handler = '_process_node_' + ''.join(
            sorted(
                '' if getattr(self, f[0]) is None else f[1]
                for f in self.client_methods.items()
            )
        )
        self._process_node = getattr(
            self, recursive_node_handler, self._process_node_general
        )

    def walk_expression(self, expr):
        """Walk an expression, calling registered callbacks.

        This is the standard interface for running the visitor.  It
        defaults to using an efficient recursive implementation of the
        visitor, falling back on :py:meth:`walk_expression_nonrecursive`
        if the recursion stack gets too deep.

        """
        if self.initializeWalker is not None:
            walk, root = self.initializeWalker(expr)
            if not walk:
                return root
            elif root is None:
                root = expr
        else:
            root = expr

        try:
            result = self._process_node(root, RECURSION_LIMIT)
            _nonrecursive = None
        except RevertToNonrecursive:
            ptr = (None,) + self.recursion_stack.pop()
            while self.recursion_stack:
                ptr = (ptr,) + self.recursion_stack.pop()
            self.recursion_stack = None
            _nonrecursive = self._nonrecursive_walker_loop, ptr
        except RecursionError:
            logger.warning(
                'Unexpected RecursionError walking an expression tree.',
                extra={'id': 'W1003'},
            )
            _nonrecursive = self.walk_expression_nonrecursive, expr

        if _nonrecursive is not None:
            return _nonrecursive[0](_nonrecursive[1])

        if self.finalizeResult is not None:
            return self.finalizeResult(result)
        else:
            return result

    def _compute_actual_recursion_limit(self):
        recursion_limit = (
            sys.getrecursionlimit() - get_stack_depth() - 2 * RECURSION_LIMIT
        )
        if recursion_limit <= RECURSION_LIMIT:
            self.recursion_stack = []
            raise RevertToNonrecursive()
        return recursion_limit

    def _process_node_general(self, node, recursion_limit):
        """Recursive routine for processing nodes with general callbacks

        This is the "general" implementation of the
        StreamBasedExpressionVisitor node processor that can handle any
        combination of registered callback functions.

        """
        if not recursion_limit:
            recursion_limit = self._compute_actual_recursion_limit()
        else:
            recursion_limit -= 1

        if self.enterNode is not None:
            tmp = self.enterNode(node)
            if tmp is None:
                args = data = None
            else:
                args, data = tmp
        else:
            args = None
            data = []
        if args is None:
            if type(node) in nonpyomo_leaf_types or not node.is_expression_type():
                args = ()
            else:
                args = node.args

        # Because we do not require the args to be a context manager, we
        # will mock up the "with args" using a try-finally.
        context_manager = hasattr(args, '__enter__')
        if context_manager:
            args.__enter__()

        try:
            descend = True
            child_idx = -1
            # Note: this relies on iter(iterator) returning the
            # iterator.  This seems to hold for all common iterators
            # (list, tuple, generator, etc)
            arg_iter = iter(args)
            for child in arg_iter:
                child_idx += 1
                if self.beforeChild is not None:
                    tmp = self.beforeChild(node, child, child_idx)
                    if tmp is None:
                        descend = True
                    else:
                        descend, child_result = tmp

                if descend:
                    child_result = self._process_node(child, recursion_limit)

                if self.acceptChildResult is not None:
                    data = self.acceptChildResult(node, data, child_result, child_idx)
                elif data is not None:
                    data.append(child_result)

                if self.afterChild is not None:
                    self.afterChild(node, child, child_idx)
        except RevertToNonrecursive:
            self._recursive_frame_to_nonrecursive_stack(locals())
            context_manager = False
            raise
        finally:
            if context_manager:
                args.__exit__(None, None, None)

        # We are done with this node.  Call exitNode to compute
        # any result
        if self.exitNode is not None:
            return self.exitNode(node, data)
        else:
            return data

    def _process_node_bex(self, node, recursion_limit):
        """Recursive routine for processing nodes with only 'bex' callbacks

        This is a special-case implementation of the "general"
        StreamBasedExpressionVisitor node processor for the case that
        only beforeChild, enterNode, and exitNode are defined (see
        also the definition of the client_methods dict).

        """
        if not recursion_limit:
            recursion_limit = self._compute_actual_recursion_limit()
        else:
            recursion_limit -= 1

        tmp = self.enterNode(node)
        if tmp is None:
            args = data = None
        else:
            args, data = tmp
        if args is None:
            if type(node) in nonpyomo_leaf_types or not node.is_expression_type():
                args = ()
            else:
                args = node.args

        # Because we do not require the args to be a context manager, we
        # will mock up the "with args" using a try-finally.
        context_manager = hasattr(args, '__enter__')
        if context_manager:
            args.__enter__()

        try:
            child_idx = -1
            # Note: this relies on iter(iterator) returning the
            # iterator.  This seems to hold for all common iterators
            # (list, tuple, generator, etc)
            arg_iter = iter(args)
            for child in arg_iter:
                child_idx += 1
                tmp = self.beforeChild(node, child, child_idx)
                if tmp is None:
                    descend = True
                else:
                    descend, child_result = tmp

                if descend:
                    data.append(self._process_node(child, recursion_limit))
                else:
                    data.append(child_result)
        except RevertToNonrecursive:
            self._recursive_frame_to_nonrecursive_stack(locals())
            context_manager = False
            raise
        finally:
            if context_manager:
                args.__exit__(None, None, None)

        # We are done with this node.  Call exitNode to compute
        # any result
        return self.exitNode(node, data)

    def _process_node_bx(self, node, recursion_limit):
        """Recursive routine for processing nodes with only 'bx' callbacks

        This is a special-case implementation of the "general"
        StreamBasedExpressionVisitor node processor for the case that
        only beforeChild and exitNode are defined (see also the
        definition of the client_methods dict).

        """
        if not recursion_limit:
            recursion_limit = self._compute_actual_recursion_limit()
        else:
            recursion_limit -= 1

        if type(node) in nonpyomo_leaf_types or not node.is_expression_type():
            args = ()
        else:
            args = node.args
        data = []

        try:
            child_idx = -1
            # Note: this relies on iter(iterator) returning the
            # iterator.  This seems to hold for all common iterators
            # (list, tuple, generator, etc)
            arg_iter = iter(args)
            for child in arg_iter:
                child_idx += 1
                tmp = self.beforeChild(node, child, child_idx)
                if tmp is None:
                    descend = True
                else:
                    descend, child_result = tmp
                if descend:
                    data.append(self._process_node(child, recursion_limit))
                else:
                    data.append(child_result)
        except RevertToNonrecursive:
            self._recursive_frame_to_nonrecursive_stack(locals())
            raise
        finally:
            pass

        # We are done with this node.  Call exitNode to compute
        # any result
        return self.exitNode(node, data)

    def _recursive_frame_to_nonrecursive_stack(self, local):
        child_idx = local['child_idx']
        _arg_list = [None] * child_idx
        _arg_list.append(local['child'])
        _arg_list.extend(local['arg_iter'])
        if not self.recursion_stack:
            # For the deepest stack frame, the recursion limit hit
            # as we started to enter the child.  As we haven't
            # started processing it yet, we need to decrement
            # child_idx so that it is revisited
            child_idx -= 1
        self.recursion_stack.append(
            (local['node'], _arg_list, len(_arg_list) - 1, local['data'], child_idx)
        )

    def walk_expression_nonrecursive(self, expr):
        """Nonrecursively walk an expression, calling registered callbacks.

        This routine is safer than the recursive walkers for deep (or
        unbalanced) trees.  It is, however, slightly slower than the
        recursive implementations.

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
            elif result is not None:
                expr = result
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
            if type(expr) in nonpyomo_leaf_types or not expr.is_expression_type():
                args = ()
            else:
                args = expr.args
        if hasattr(args, '__enter__'):
            args.__enter__()
        node = expr
        # Note that because we increment child_idx just before fetching
        # the child node, it must be initialized to -1, and ptr[3] must
        # always be *one less than* the number of arguments
        return self._nonrecursive_walker_loop(
            (None, node, args, len(args) - 1, data, -1)
        )

    def _nonrecursive_walker_loop(self, ptr):
        _, node, args, _, data, child_idx = ptr
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
                                    node, data, child_result, child_idx
                                )
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
                    ptr = ptr[:4] + (data, child_idx)

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
                        if (
                            type(child) in nonpyomo_leaf_types
                            or not child.is_expression_type()
                        ):
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
                    ptr = (ptr, node, args, len(args) - 1, data, child_idx)

                else:  # child_idx == ptr[3]:
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
                            node, data, node_result, child_idx
                        )
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


@deprecated(
    "The SimpleExpressionVisitor is deprecated.  "
    "Please use the StreamBasedExpressionVisitor instead.",
    version='6.9.0',
)
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

    def visit(self, node):  # pragma: no cover
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

    def finalize(self):  # pragma: no cover
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
        if (
            node.__class__ in nonpyomo_leaf_types
            or not node.is_expression_type()
            or node.nargs() == 0
        ):
            self.visit(node)
            return self.finalize()

        dq = deque([node])
        while dq:
            current = dq.popleft()
            self.visit(current)
            # for c in self.children(current):
            for c in current.args:
                # if self.is_leaf(c):
                if (
                    c.__class__ in nonpyomo_leaf_types
                    or not c.is_expression_type()
                    or c.nargs() == 0
                ):
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
        if (
            node.__class__ in nonpyomo_leaf_types
            or not node.is_expression_type()
            or node.nargs() == 0
        ):
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
            # self.visit(current)
            # for c in self.children(current):
            for c in current.args:
                # if self.is_leaf(c):
                if (
                    c.__class__ in nonpyomo_leaf_types
                    or not c.is_expression_type()
                    or c.nargs() == 0
                ):
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

    def visit(self, node, values):  # pragma: no cover
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

    def visiting_potential_leaf(self, node):  # pragma: no cover
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

    def finalize(self, ans):  # pragma: no cover
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
            return self.finalize(value)
        # _stack = [ (node, self.children(node), 0, len(self.children(node)), [])]
        _stack = [(node, node.args, 0, node.nargs(), [])]
        #
        # Iterate until the stack is empty
        #
        # Note: 1 is faster than True for Python 2.x
        #
        while 1:
            #
            # Get the top of the stack
            #   _obj        Current expression object
            #   _argList    The arguments for this expression object
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
                    _result.append(value)
                else:
                    #
                    # Push an expression onto the stack
                    #
                    _stack.append((_obj, _argList, _idx, _len, _result))
                    _obj = _sub
                    # _argList                = self.children(_sub)
                    _argList = _sub.args
                    _idx = 0
                    _len = _sub.nargs()
                    _result = []
            #
            # Process the current node
            #
            ans = self.visit(_obj, _result)
            if _stack:
                #
                # "return" the recursion by putting the return value on
                # the end of the results stack
                #
                _stack[-1][-1].append(ans)
            else:
                return self.finalize(ans)


def replace_expressions(
    expr,
    substitution_map,
    descend_into_named_expressions=True,
    remove_named_expressions=True,
):
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
    return ExpressionReplacementVisitor(
        substitute=substitution_map,
        descend_into_named_expressions=descend_into_named_expressions,
        remove_named_expressions=remove_named_expressions,
    ).walk_expression(expr)


class ExpressionReplacementVisitor(StreamBasedExpressionVisitor):
    def __init__(
        self,
        substitute=None,
        descend_into_named_expressions=True,
        remove_named_expressions=True,
    ):
        if substitute is None:
            substitute = {}
        # Note: preserving the attribute names from the previous
        # implementation of the expression walker.
        self.substitute = substitute
        self.enter_named_expr = descend_into_named_expressions
        self.rm_named_expr = remove_named_expressions

        kwds = {}
        if hasattr(self, 'visiting_potential_leaf'):
            deprecation_warning(
                "ExpressionReplacementVisitor: this walker has been ported "
                "to derive from StreamBasedExpressionVisitor.  "
                "visiting_potential_leaf() has been replaced by beforeChild()"
                "(note to implementers: the sense of the bool return value "
                "has been inverted).",
                version='6.2',
            )

            def beforeChild(node, child, child_idx):
                is_leaf, ans = self.visiting_potential_leaf(child)
                return not is_leaf, ans

            kwds['beforeChild'] = beforeChild

        if hasattr(self, 'visit'):
            raise DeveloperError(
                "ExpressionReplacementVisitor: this walker has been ported "
                "to derive from StreamBasedExpressionVisitor.  "
                "overriding visit() has no effect (and is likely to generate "
                "invalid expression trees)"
            )
        super().__init__(**kwds)

    def initializeWalker(self, expr):
        walk, result = self.beforeChild(None, expr, 0)
        if not walk:
            return False, result
        return True, expr

    def beforeChild(self, node, child, child_idx):
        if id(child) in self.substitute:
            return False, self.substitute[id(child)]
        elif type(child) in native_types:
            return False, child
        elif not child.is_expression_type():
            return False, child
        elif child.is_named_expression_type():
            if not self.enter_named_expr:
                return False, child
        return True, None

    def enterNode(self, node):
        args = list(node.args)
        # [bool:args_have_changed, list:original_args, bool:node_is_constant]
        return args, [False, args, True]

    def acceptChildResult(self, node, data, child_result, child_idx):
        if data[1][child_idx] is not child_result:
            data[1][child_idx] = child_result
            data[0] = True
        if (
            child_result.__class__ not in native_types
            and not child_result.is_constant()
        ):
            data[2] = False
        return data

    def exitNode(self, node, data):
        if node.is_named_expression_type():
            assert len(data[1]) == 1
            if self.rm_named_expr:
                return data[1][0]
            elif data[0]:
                node.set_value(data[1][0])
                return node
        elif data[0]:
            if data[2]:
                return node._apply_operation(data[1])
            else:
                return node.create_node_with_local_data(data[1])
        return node

    @deprecated(
        "ExpressionReplacementVisitor: this walker has been ported "
        "to derive from StreamBasedExpressionVisitor.  "
        "dfs_postorder_stack() has been replaced with walk_expression()",
        version='6.2',
    )
    def dfs_postorder_stack(self, expr):
        return self.walk_expression(expr)


def evaluate_fixed_subexpressions(
    expr, descend_into_named_expressions=True, remove_named_expressions=True
):
    return EvaluateFixedSubexpressionVisitor(
        descend_into_named_expressions=descend_into_named_expressions,
        remove_named_expressions=remove_named_expressions,
    ).walk_expression(expr)


class EvaluateFixedSubexpressionVisitor(ExpressionReplacementVisitor):
    def __init__(
        self, descend_into_named_expressions=False, remove_named_expressions=False
    ):
        super().__init__(
            descend_into_named_expressions=descend_into_named_expressions,
            remove_named_expressions=remove_named_expressions,
        )

    def beforeChild(self, node, child, child_idx):
        if type(child) in native_types:
            return False, child
        elif not child.is_expression_type():
            if child.is_fixed():
                return False, child()
            else:
                return False, child
        elif child.is_named_expression_type():
            if not self.enter_named_expr:
                return False, child
        return True, None


# -------------------------------------------------------
#
# Functions used to process expression trees
#
# -------------------------------------------------------

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
    common.clone_counter._count += 1
    memo = {'__block_scope__': {id(None): False}}
    if substitute:
        expr = replace_expressions(expr, substitute)
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
        enterNode=enter, acceptChildResult=accept
    ).walk_expression(expr)


# =====================================================
#  evaluate_expression
# =====================================================


class _EvaluationVisitor(ExpressionValueVisitor):
    def __init__(self, exception):
        self.exception = exception

    def visit(self, node, values):
        """Visit nodes that have been expanded"""
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
            return True, value(node, exception=self.exception)
        elif node.is_logical_type():
            return True, value(node, exception=self.exception)
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
        """Visit nodes that have been expanded"""
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
                    raise NonConstantExpressionError(
                        f"{node} ({type(node).__name__}) is not fixed"
                    )
                if not node.is_constant():
                    raise FixedExpressionError(
                        f"{node} ({type(node).__name__}) is not constant"
                    )
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
    clear_active = False
    if constant:
        visitor = _EvaluateConstantExpressionVisitor()
    else:
        if evaluate_expression.visitor_active:
            visitor = _EvaluationVisitor(exception=exception)
        else:
            visitor = evaluate_expression.visitor_cache
            visitor.exception = exception
            evaluate_expression.visitor_active = True
            clear_active = True

    try:
        ans = visitor.dfs_postorder_stack(exp)
    except (
        TemplateExpressionError,
        ValueError,
        TypeError,
        NonConstantExpressionError,
        FixedExpressionError,
    ):
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
    finally:
        if clear_active:
            evaluate_expression.visitor_active = False
    if ans.__class__ not in native_types and ans.is_numeric_type() is True:
        return value(ans)
    return ans


evaluate_expression.visitor_cache = _EvaluationVisitor(True)
evaluate_expression.visitor_active = False

# =====================================================
#  identify_components
# =====================================================


class _ComponentVisitor(StreamBasedExpressionVisitor):
    def __init__(self, types):
        super().__init__()
        if types.__class__ is not set:
            types = set(types)
        self._types = types

    def initializeWalker(self, expr):
        self._objs = []
        self._seen = set()
        return True, None

    def finalizeResult(self, result):
        return self._objs

    def exitNode(self, node, data):
        if node.__class__ in self._types and id(node) not in self._seen:
            self._seen.add(id(node))
            self._objs.append(node)


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
    yield from visitor.walk_expression(expr)


# =====================================================
#  identify_variables
# =====================================================


class IdentifyVariableVisitor(StreamBasedExpressionVisitor):
    def __init__(self, include_fixed=False, named_expression_cache=None):
        """Visitor that collects all unique variables participating in an
        expression

        Args:
            include_fixed (bool): Whether to include fixed variables
            named_expression_cache (optional, dict): Dict mapping ids of named
                expressions to a tuple of the list of all variables and the
                set of all variable ids contained in the named expression.

        """
        super().__init__()
        self._include_fixed = include_fixed
        self._cache = named_expression_cache
        # Stack of named expressions. This holds the tuple
        #     (eid, _seen, _exprs)
        # where eid is the id() of the subexpression we are currently
        # processing, and _seen and _exprs are from the parent context.
        self._expr_stack = []
        # The following attributes will be added by initializeWalker:
        # self._seen: dict(eid: obj)
        # self._exprs: list of (e, e.expr) for any (nested) named expressions

    def initializeWalker(self, expr):
        assert not self._expr_stack
        self._seen = {}
        self._exprs = None
        if not self.beforeChild(None, expr, 0)[0]:
            return False, self.finalizeResult(None)
        return True, expr

    def beforeChild(self, parent, child, index):
        if child.__class__ in native_types:
            return False, None
        elif child.is_expression_type():
            if child.is_named_expression_type():
                return self._process_named_expr(child)
            else:
                return True, None
        elif child.is_variable_type() and (self._include_fixed or not child.fixed):
            if id(child) not in self._seen:
                self._seen[id(child)] = child
        return False, None

    def exitNode(self, node, data):
        if node.is_named_expression_type() and self._cache is not None:
            # If we are returning from a named expression, we must make
            # sure that we properly restore the "outer" context and then
            # merge the objects from the named expression we just exited
            # into the list for the parent expression context.
            _seen = self._seen
            _exprs = self._exprs
            eid, self._seen, self._exprs = self._expr_stack.pop()
            assert eid == id(node)
            self._merge_obj_lists(_seen, _exprs)

    def finalizeResult(self, result):
        assert not self._expr_stack
        return self._seen.values()

    def _merge_obj_lists(self, _seen, _exprs):
        self._seen.update(_seen)
        if self._exprs is not None:
            self._exprs.update(_exprs)

    def _process_named_expr(self, child):
        if self._cache is None:
            return True, None
        eid = id(child)
        if eid in self._cache:
            _seen, _exprs = self._cache[eid]
            if all(c.expr is e for c, e in _exprs.values()):
                # We have already encountered this named expression. We just add
                # the cached objects to our list and don't descend.
                #
                # Note that a cache hit requires not only that we have seen
                # this expression before, but also that none of the named
                # expressions have changed.  If they have, then the cache
                # miss will fall over to the else clause below and descend
                # into the expression, (implicitly) rebuilding the cache.
                self._merge_obj_lists(_seen, _exprs)
                return False, None
        # If we are descending into a new named expression or a cached
        # named expression where the cache is now invalid.  Initialize a
        # cache to store the expression's local objects.
        self._expr_stack.append((eid, self._seen, self._exprs))
        self._seen = {}
        self._exprs = {eid: (child, child.expr)}
        self._cache[eid] = (self._seen, self._exprs)
        return True, None


def identify_variables(expr, include_fixed=True, named_expression_cache=None):
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
    v = identify_variables.visitor
    save = v._include_fixed, v._cache
    try:
        v._include_fixed = include_fixed
        v._cache = named_expression_cache
        yield from v.walk_expression(expr)
    finally:
        v._include_fixed, v._cache = save


identify_variables.visitor = IdentifyVariableVisitor()


# =====================================================
#  identify_mutable_parameters
# =====================================================


class IdentifyMutableParamVisitor(IdentifyVariableVisitor):
    def __init__(self):
        # Hide the IdentifyVariableVisitor API (not relevant here)
        super().__init__()

    def beforeChild(self, parent, child, index):
        if child.__class__ in native_types:
            return False, None
        elif child.is_expression_type():
            if child.is_named_expression_type():
                return self._process_named_expr(child)
            else:
                return True, None
        if (
            not child.is_variable_type()
            and child.is_fixed()
            and not child.is_constant()
        ):
            if id(child) not in self._seen:
                self._seen[id(child)] = child
        return False, None


def identify_mutable_parameters(expr):
    """
    A generator that yields a sequence of mutable
    parameters in an expression tree.

    Args:
        expr: The root node of an expression tree.

    Yields:
        Each mutable parameter that is found.
    """
    yield from identify_mutable_parameters.visitor.walk_expression(expr)


identify_mutable_parameters.visitor = IdentifyMutableParamVisitor()

# =====================================================
#  polynomial_degree
# =====================================================


class _PolynomialDegreeVisitor(ExpressionValueVisitor):
    def visit(self, node, values):
        """Visit nodes that have been expanded"""
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
        """Visit nodes that have been expanded"""
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
    """Return bool indicating if this expression is fixed (non-variable)

    Args:
        node: The root node of an expression tree.

    Returns: bool

    """
    visitor = _IsFixedVisitor()
    return visitor.dfs_postorder_stack(node)


# =====================================================
#  expression_to_string
# =====================================================

LEFT_TO_RIGHT = common.OperatorAssociativity.LEFT_TO_RIGHT
RIGHT_TO_LEFT = common.OperatorAssociativity.RIGHT_TO_LEFT


class _ToStringVisitor(ExpressionValueVisitor):
    _expression_handlers = None
    _leaf_node_types = set()

    def __init__(self, verbose, smap):
        super(_ToStringVisitor, self).__init__()
        self.verbose = verbose
        self.smap = smap

    def visit(self, node, values):
        """Visit nodes that have been expanded"""
        node_prec = node.PRECEDENCE
        if node_prec is not None and not self.verbose:
            for i, (val, arg) in enumerate(zip(values, node.args)):
                arg_prec = getattr(arg, 'PRECEDENCE', None)
                if arg_prec is None:
                    # This embedded constant (4) is evil, but to actually
                    # import the NegationExpression.PRECEDENCE from
                    # numeric_expr would create a circular dependency.
                    #
                    # FIXME: rework the dependencies between
                    # numeric_expr and visitor
                    if val[0] == '-' and node_prec < 4:
                        values[i] = f"({val})"
                else:
                    if node_prec < arg_prec:
                        parens = True
                    elif node_prec == arg_prec:
                        if i == 0:
                            parens = node.ASSOCIATIVITY != LEFT_TO_RIGHT
                        elif i == node.nargs() - 1:
                            parens = node.ASSOCIATIVITY != RIGHT_TO_LEFT
                        else:
                            parens = True
                    else:
                        parens = False
                    if parens:
                        values[i] = f"({val})"

        if self._expression_handlers and node.__class__ in self._expression_handlers:
            return self._expression_handlers[node.__class__](self, node, values)

        return node._to_string(values, self.verbose, self.smap)

    def visiting_potential_leaf(self, node):
        """
        Visiting a potential leaf.

        Return True if the node is not expanded.
        """
        if node is None:
            return True, 'Undefined'

        if node.__class__ in native_numeric_types:
            return True, str(node)

        if node.__class__ in nonpyomo_leaf_types:
            return True, repr(node)

        if node.is_expression_type() and node.__class__ not in self._leaf_node_types:
            return False, None

        if hasattr(node, 'to_string'):
            return True, node.to_string(verbose=self.verbose, smap=self.smap)
        elif self.smap is not None:
            return True, self.smap.getSymbol(node)
        else:
            return True, str(node)


def expression_to_string(
    expr, verbose=None, labeler=None, smap=None, compute_values=False
):
    """Return a string representation of an expression.

    Parameters
    ----------
    expr: ExpressionBase
        The root node of an expression tree.

    verbose: bool
        If :const:`True`, then the output is a nested functional form.
        Otherwise, the output is an algebraic expression.  Default is
        retrieved from :py:attr:`common.TO_STRING_VERBOSE`

    labeler: Callable
        If specified, this labeler is used to generate the string
        representation for leaves (Var / Param objects) in the
        expression.

    smap:  SymbolMap
        If specified, this :class:`SymbolMap
        <pyomo.core.expr.symbol_map.SymbolMap>` is used to cache labels.

    compute_values: bool
        If :const:`True`, then parameters and fixed variables are
        evaluated before the expression string is generated.  Default is
        :const:`False`.

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
    # TODO: should we deprecate the compute_values option?
    #
    if compute_values:
        expr = evaluate_fixed_subexpressions(expr)
    #
    # Create and execute the visitor pattern
    #
    visitor = _ToStringVisitor(verbose, smap)
    return visitor.dfs_postorder_stack(expr)
