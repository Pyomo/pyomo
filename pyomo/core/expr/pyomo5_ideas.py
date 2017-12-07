#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

##
## This is code that I've removed from Pyomo5, but which we might want to use later ...
##

def _clear_expression_pool():
    pass

sum = builtins.sum

UNREFERENCED_EXPR_COUNT = 10
UNREFERENCED_INTRINSIC_EXPR_COUNT = 8 
UNREFERENCED_EXPR_IF_COUNT = 10
if sys.version_info[:2] >= (3, 6):
    UNREFERENCED_EXPR_COUNT -= 1
    UNREFERENCED_INTRINSIC_EXPR_COUNT += 1
    UNREFERENCED_EXPR_IF_COUNT -= 1
elif sys.version_info[:2] < (2, 7):
    UNREFERENCED_EXPR_IF_COUNT = -4

class EntangledExpressionError(Exception):
    def __init__(self, sub_expr):
        msg = \
"""Attempting to form an expression with a
subexpression that is already part of another expression of component.
This would create two expressions that share common subexpressions,
which is not allowed in Pyomo.  Either clone the subexpression using
'clone_expression' before creating the new expression, or if you want
the two expressions to share a common subexpression, use an Expression
component to store the subexpression and use the subexpression in each
expression.  Common subexpression:\n\t%s""" % (str(sub_expr),)
        super(EntangledExpressionError, self).__init__(msg)

#-------------------------------------------------------
#
# Global Data
#
#-------------------------------------------------------

class ignore_entangled_context(object):
    detangle = [True]

    def __enter__(self):
        ignore_entangled_expressions.detangle.append(False)

    def __exit__(self, *args):
        ignore_entangled_expressions.detangle.pop()

ignore_entangled_expressions = ignore_entangled_context()


class mutable_quadratic_context(object):

    def __enter__(self):
        self.e = _QuadraticExpression()
        return self.e

    def __exit__(self, *args):
        if self.e.__class__ == _QuadraticExpression:
            self.e.__class__ = _StaticQuadraticExpression

#quadratic_expression = mutable_quadratic_context()
quadratic_expression = nonlinear_expression


def _orig_clone_expression(expr, memo=None, verbose=False, clone_leaves=True):
    clone_counter_context._count += 1
    if not memo:
        memo = {'__block_scope__': { id(None): False }}
    #
    if expr.__class__ in native_numeric_types:
        return expr
    if not expr.is_expression_type():
        return deepcopy(expr, memo)
    #
    # The stack starts with the current expression
    #
    _stack = [ (expr, expr._args, 0, expr.nargs(), [])]
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
        #
        _obj, _argList, _idx, _len, _result = _stack.pop()
        if verbose: #pragma:nocover
            print("*"*10 + " POP  " + "*"*10)
        #
        # Iterate through the arguments
        #
        while _idx < _len:
            if verbose: #pragma:nocover
                print("-"*30)
                print(type(_obj))
                print(_obj)
                print(_argList)
                print(_idx)
                print(_len)
                print(_result)

            _sub = _argList[_idx]
            _idx += 1
            if _sub.__class__ in native_numeric_types:
                #
                # Store a native or numeric object
                #
                _result.append( deepcopy(_sub, memo) )
            elif _sub.__class__ not in pyomo5_expression_types:
                #
                # Store a kernel object that is cloned
                #
                if clone_leaves:
                    _result.append( deepcopy(_sub, memo) )
                else:
                    _result.append( _sub )
            else:
                #
                # Push an expression onto the stack
                #
                if verbose: #pragma:nocover
                    print("*"*10 + " PUSH " + "*"*10)
                _stack.append( (_obj, _argList, _idx, _len, _result) )
                _obj     = _sub
                _argList = _sub._args
                _idx     = 0
                _len     = _sub.nargs()
                _result  = []
    
        if verbose: #pragma:nocover
            print("="*30)
            print(type(_obj))
            print(_obj)
            print(_argList)
            print(_idx)
            print(_len)
            print(_result)
        #
        # Now replace the current expression object
        #
        ans = _obj._clone( tuple(_result), memo )
        if verbose: #pragma:nocover
            print("STACK LEN %d" % len(_stack))
        if _stack:
            #
            # "return" the recursion by putting the return value on the end of the reults stack
            #
            _stack[-1][-1].append( ans )
        else:
            return ans


def _orig_sizeof_expression(expr, verbose=False):
    #
    # Note: This does not try to optimize the compression to recognize
    #   subgraphs.
    #
    if expr.__class__ in native_numeric_types or not expr.is_expression_type():
        return 1
    #
    # The stack starts with the current expression
    #
    _stack = [ (expr, expr._args, 0, expr.nargs(), [])]
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
        #
        _obj, _argList, _idx, _len, _result = _stack.pop()
        if verbose: #pragma:nocover
            print("*"*10 + " POP  " + "*"*10)
        #
        # Iterate through the arguments
        #
        while _idx < _len:
            if verbose: #pragma:nocover
                print("-"*30)
                print(type(_obj))
                print(_obj)
                print(_argList)
                print(_idx)
                print(_len)
                print(_result)

            _sub = _argList[_idx]
            _idx += 1
            if _sub.__class__ in native_numeric_types or not _sub.is_expression_type():
                #
                # Store a native or numeric object
                #
                _result.append( 1 )
            else:
                #
                # Push an expression onto the stack
                #
                if verbose: #pragma:nocover
                    print("*"*10 + " PUSH " + "*"*10)
                _stack.append( (_obj, _argList, _idx, _len, _result) )
                _obj     = _sub
                _argList = _sub._args
                _idx     = 0
                _len     = _sub.nargs()
                _result  = []
    
        if verbose: #pragma:nocover
            print("="*30)
            print(type(_obj))
            print(_obj)
            print(_argList)
            print(_idx)
            print(_len)
            print(_result)
            print("STACK LEN %d" % len(_stack))

        ans = sum(_result)+1
        if _stack:
            #
            # "return" the recursion by putting the return value on the end of the reults stack
            #
            _stack[-1][-1].append( ans )
        else:
            return ans


def _orig_evaluate_expression(exp, exception=True, only_fixed_vars=False):
    try:
        if exp.__class__ in pyomo5_variable_types:
            if not only_fixed_vars or exp.fixed:
                return exp.value
            else:
                raise ValueError("Cannot evaluate an unfixed variable with only_fixed_vars=True")
        elif exp.__class__ in native_numeric_types:
            return exp
        elif not exp.is_expression_type():
            return exp()

        _stack = [ (exp, exp._args, 0, exp.nargs(), []) ]
        while 1:  # Note: 1 is faster than True for Python 2.x
            _obj, _argList, _idx, _len, _result = _stack.pop()
            while _idx < _len:
                _sub = _argList[_idx]
                _idx += 1
                if _sub.__class__ in native_numeric_types:
                    _result.append( _sub )
                elif _sub.is_expression_type():
                    _stack.append( (_obj, _argList, _idx, _len, _result) )
                    _obj     = _sub
                    _argList = _sub._args
                    _idx     = 0
                    _len     = _sub.nargs()
                    _result  = []
                elif _sub.__class__ in pyomo5_variable_types:
                    if only_fixed_vars:
                        if _sub.fixed:
                            _result.append( _sub.value )
                        else:
                            raise ValueError("Cannot evaluate an unfixed variable with only_fixed_vars=True")
                    else:
                        _result.append( value(_sub) )
                else:
                    _result.append( value(_sub) )
            ans = _obj._apply_operation(_result)
            if _stack:
                _stack[-1][-1].append( ans )
            else:
                return ans
    except TemplateExpressionError:
        if exception:
            raise
        return None
    except ValueError:
        if exception:
            raise
        return None


def _orig_identify_variables(expr,
                       include_fixed=True,
                       allow_duplicates=False,
                       include_potentially_variable=False):
    if not allow_duplicates:
        _seen = set()
    _stack = [ ([expr], 0, 1) ]
    while _stack:
        _argList, _idx, _len = _stack.pop()
        while _idx < _len:
            _sub = _argList[_idx]
            _idx += 1
            if _sub.__class__ in native_types:
                pass
            elif _sub.is_expression_type():
                _stack.append(( _argList, _idx, _len ))
                _argList = _sub._args
                _idx = 0
                _len = _sub.nargs()
            elif _sub.__class__ in pyomo5_variable_types:
                if ( include_fixed
                     or not _sub.is_fixed()
                     or include_potentially_variable ):
                    if not allow_duplicates:
                        if id(_sub) in _seen:
                            continue
                        _seen.add(id(_sub))
                    yield _sub
            elif include_potentially_variable and _sub._potentially_variable():
                if not allow_duplicates:
                    if id(_sub) in _seen:
                        continue
                    _seen.add(id(_sub))
                yield _sub


def _orig_polynomial_degree(node):
    # TODO: Confirm whether this check works
    #if not node._potentially_variable():
    #    return 0
    _stack = [ (node, node._args, 0, node.nargs(), []) ]
    while 1:  # Note: 1 is faster than True for Python 2.x
        _obj, _argList, _idx, _len, _result = _stack.pop()
        while _idx < _len:
            _sub = _argList[_idx]
            _idx += 1
            if _sub.__class__ in native_numeric_types or not _sub._potentially_variable():
                _result.append( 0 )
            elif _sub.is_expression_type():
                _stack.append( (_obj, _argList, _idx, _len, _result) )
                _obj     = _sub
                _argList = _sub._args
                _idx     = 0
                _len     = _sub.nargs()
                _result  = []
            else:
                _result.append( 0 if _sub.is_fixed() else 1 )
        ans = _obj._polynomial_degree(_result)
        if _stack:
            _stack[-1][-1].append( ans )
        else:
            return ans


def _orig_expression_is_fixed(node):
    if not node._potentially_variable():
        return True
    _stack = [ (node, node._args, 0, node.nargs(), []) ]
    while 1:  # Note: 1 is faster than True for Python 2.x
        _obj, _argList, _idx, _len, _result = _stack.pop()
        while _idx < _len:
            _sub = _argList[_idx]
            _idx += 1
            if _sub.__class__ in native_numeric_types or not _sub._potentially_variable():
                _result.append( True )
            elif not _sub.__class__ in pyomo5_expression_types:
                _result.append( _sub.is_fixed() ) 
            else:
                _stack.append( (_obj, _argList, _idx, _len, _result) )
                _obj     = _sub
                _argList = _sub._args
                _idx     = 0
                _len     = _sub.nargs()
                _result  = []

        ans = _obj._is_fixed(_result)
        if _stack:
            _stack[-1][-1].append( ans )
            #if _obj._is_fixed is all:
            #    if not _result[-1]:
            #        _idx = _len
            #elif _obj._is_fixed is any:
            #    if _result[-1]:
            #        _idx = _len
        else:
            return ans


# =====================================================
#  compress_expression
# =====================================================

class CompressVisitor(ValueExpressionVisitor):

    def __init__(self, multiprod=False):
        self._clone = [None, False]
        self.multiprod = multiprod

    def visit(self, node, values):
        """ Visit nodes that have been expanded """
        clone = self._clone.pop()
        #
        # Now replace the current expression object if it's a sum
        #
        if node.__class__ is _SumExpression or node.__class__ is _NPV_SumExpression or node.__class__ is _Constant_SumExpression:
            ans = _SumExpression._combine_expr(*values)
            #
            # We've replaced a node, so set the context for the parent's search to
            # ensure that it is cloned.
            #
            self._clone[-1] = True
        #
        # Now replace the current expression object if it's a product
        #
        elif self.multiprod and node.__class__ in pyomo5_product_types:
            ans = _ProductExpression._combine_expr(*values)
            #
            # We've replaced a node, so set the context for the parent's search to
            # ensure that it is cloned.
            #
            self._clone[-1] = True
        #
        # Now replace the current expression object if it's a reciprocal
        #
        elif self.multiprod and node.__class__ in pyomo5_reciprocal_types:
            ans = _ReciprocalExpression._combine_expr(*values)
            #
            # We've replaced a node, so set the context for the parent's search to
            # ensure that it is cloned.
            #
            self._clone[-1] = True

        elif clone:
            ans = node._clone( tuple(values), None )
            self._clone[-1] = True

        else:
            ans = node
        return ans

    def visiting_potential_leaf(self, node, _values):
        """ 
        Visiting a potential leaf.

        Return True if the node is not expanded.
        """
        if node.__class__ in native_numeric_types or \
               node.__class__ not in pyomo5_expression_types or \
               node.__class__ in pyomo5_multisum_types or \
               node.__class__ is _MultiProdExpression or \
               not node._potentially_variable():
            _values.append( node )
            return True
        #
        # This node is expanded, so set its cloning flag.
        #
        self._clone.append(False)
        return False

    def finalize(self, ans):
        #if ans.__class__ is _MutableMultiSumExpression:
        #    ans.__class__ = _CompressedSumExpression
        return ans


def NEW_compress_expression(expr, verbose=False, dive=False, multiprod=False):
    #
    # Only compress a true expression DAG
    #
    # Note: This does not try to optimize the compression to recognize
    #   subgraphs.
    #
    # Note: This uses a two-part stack.  The boolean indicates whether the
    #   parent should be cloned (because a child has been replaced), and the
    #   tuple represents the current context during the tree search.
    #
    if expr.__class__ in native_numeric_types or not expr.is_expression_type() or not expr._potentially_variable():
        return expr
    if expr.__class__ is _MutableMultiSumExpression:
        expr.__class__ = _CompressedSumExpression
        return expr
    if expr.__class__ in pyomo5_multisum_types:
        return expr
    #
    # Only compress trees whose root is _SumExpression
    #
    # Note: This tacitly avoids compressing all trees
    # that are not potentially variable, since they have a
    # different class.
    #
    if not dive and \
       not (expr.__class__ is _SumExpression or expr.__class__ is _NPV_SumExpression or expr.__class__ is _Constant_SumExpression):
        return expr
    visitor = CompressVisitor(multiprod=multiprod)
    return visitor.dfs_postorder_stack(expr)


def compress_expression(expr, verbose=False, dive=False, multiprod=False):
    return expr

def Xcompress_expression(expr, verbose=False, dive=False, multiprod=False):
    #
    # Only compress a true expression DAG
    #
    # Note: This does not try to optimize the compression to recognize
    #   subgraphs.
    #
    # Note: This uses a two-part stack.  The boolean indicates whether the
    #   parent should be cloned (because a child has been replaced), and the
    #   tuple represents the current context during the tree search.
    #
    if expr.__class__ in native_numeric_types or not expr.is_expression_type() or not expr._potentially_variable():
        return expr
    #if expr.__class__ is _MutableMultiSumExpression:
    #    expr.__class__ = _CompressedSumExpression
    #    return expr
    if expr.__class__ in pyomo5_multisum_types:
        return expr
    #
    # Only compress trees whose root is _SumExpression
    #
    # Note: This tacitly avoids compressing all trees
    # that are not potentially variable, since they have a
    # different class.
    #
    if not dive and \
       not (expr.__class__ is _SumExpression or expr.__class__ is _NPV_SumExpression or expr.__class__ is _Constant_SumExpression):
        return expr
    #
    # The stack starts with the current expression
    #
    _stack = [ False, (expr, expr._args, 0, expr.nargs(), [])]
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
        #
        _obj, _argList, _idx, _len, _result = _stack.pop()
        _clone = _stack.pop()
        if _clone and _stack:
            _stack[-2] = True
        if verbose: #pragma:nocover
            print("*"*10 + " POP  " + "*"*10)
        #
        # Iterate through the arguments
        #
        while _idx < _len:
            if verbose: #pragma:nocover
                print("-"*30)
                print(type(_obj))
                print(_obj)
                print(_argList)
                print(_idx)
                print(_len)
                print(_result)
                print(_clone)
                print("-"*30)

            _sub = _argList[_idx]
            _idx += 1
            if _sub.__class__ in native_numeric_types:
                #
                # Store a native or numeric object
                #
                _result.append( _sub )
            elif _sub.__class__ not in pyomo5_expression_types or \
                 _sub.__class__ in pyomo5_multisum_types or \
                 _sub.__class__ is _MultiProdExpression or \
                 not _sub._potentially_variable():
                _result.append( _sub )
            else:
                #
                # Push an expression onto the stack
                #
                if verbose: #pragma:nocover
                    print("*"*10 + " PUSH " + "*"*10)
                _stack.append( False )
                _stack.append( (_obj, _argList, _idx, _len, _result) )
                _obj                    = _sub
                _argList                = _sub._args
                _idx                    = 0
                _len                    = _sub.nargs()
                _result                 = []
                _clone                  = False
    
        if verbose: #pragma:nocover
            print("="*30)
            print(type(_obj))
            print(_obj)
            print(_argList)
            print(_idx)
            print(_len)
            print(_result)
            print(_clone)
            print("="*30)
        #
        # Now replace the current expression object if it's a sum
        #
        if _obj.__class__ is _SumExpression or _obj.__class__ is _NPV_SumExpression or _obj.__class__ is _Constant_SumExpression:
            ans = _SumExpression._combine_expr(*_result)
            if _stack:
                #
                # We've replaced a node, so set the context for the parent's search to
                # ensure that it is cloned.
                #
                _stack[-2] = True
        #
        # Now replace the current expression object if it's a product
        #
        elif multiprod and _obj.__class__ in pyomo5_product_types:
            ans = _ProductExpression._combine_expr(*_result)
            if _stack:
                #
                # We've replaced a node, so set the context for the parent's search to
                # ensure that it is cloned.
                #
                _stack[-2] = True
        #
        # Now replace the current expression object if it's a reciprocal
        #
        elif multiprod and _obj.__class__ in pyomo5_reciprocal_types:
            ans = _ReciprocalExpression._combine_expr(*_result)
            if _stack:
                #
                # We've replaced a node, so set the context for the parent's search to
                # ensure that it is cloned.
                #
                _stack[-2] = True

        elif _clone:
            ans = _obj._clone( tuple(_result), None )
            if _stack:
                _stack[-2] = True

        else:
            ans = _obj

        #print(ans)
        #print(ans._args)
        if verbose: #pragma:nocover
            print("STACK LEN %d" % len(_stack))
        if _stack:
            #
            # "return" the recursion by putting the return value on the end of the results stack
            #
            _stack[-1][-1].append( ans )
        else:
            #if ans.__class__ is _MutableMultiSumExpression:
            #    ans.__class__ = _CompressedSumExpression
            return ans


class _MultiProdExpression(_ProductExpression):
    """An object that defines a product with 1 or more terms, including denominators."""

    __slots__ = ('_nnum',)
    PRECEDENCE = 4

    def __init__(self, args, nnum=None):
        self._args = args
        self._nnum = nnum
        if _getrefcount_available:
            self._is_owned = UNREFERENCED_EXPR_COUNT
        else:
            self._is_owned = False
            for arg in args:
                if arg.__class__ in pyomo5_expression_types:
                    arg._is_owned = True

    def nargs(self):
        return len(self._args)

    def _clone(self, args, memo):
        return self.__class__(args, self._nnum)

    def _precedence(self):
        return _MultiProdExpression.PRECEDENCE

    def _apply_operation(self, result):
        return prod(result)

    def getname(self, *args, **kwds):
        return 'multiprod'

    def _potentially_variable(self):
        return len(self._args) > 1

    def _apply_operation(self, result):
        ans = 1
        i = 0
        n_ = len(self._args)
        for j in xargs(0,nnum):
            ans *= result[i]
            i += 1
        while i < n_:
            ans /= result[i]
            i += 1


class _LinearViewSumExpression(_ViewSumExpression):

    __slots__ = ()

    def __init__(self, args):
        if args.__class__ is tuple:
            self._args = []
            linear_terms = []
            nonlinear_terms = []
            for arg in args:
                # NOTE - Change this to use the decompose_term() function
                self.decompose(arg, linear_terms, nonlinear_terms)
            self._args.extend(linear_terms)
            if len(nonlinear_terms) > 0:
                # We add nonlinear terms, but change the class type.
                self._args.extend(nonlinear_terms)
                self.__class__ = _ViewSumExpression
        else:
            self._args = args
        self._is_owned = False
        self._nargs = len(self._args)

    def add(self, new_arg):
        if new_arg.__class__ in native_numeric_types and isclose(new_arg,0):
            return self
        # Clone 'self', because _LinearViewSumExpression are immutable
        self._is_owned = True
        self = self.__class__(self._args)
        #
        if new_arg.__class__ is _LinearViewSumExpression:
            self._args.extend(islice(new_arg._args, new_arg._nargs))
        else:
            linear_terms = []
            nonlinear_terms = []
            if new_arg.__class__ is _ViewSumExpression or new_arg.__class__ is _MutableViewSumExpression:
                for arg in islice(new_arg._args, new_arg._nargs):
                    self.decompose(arg, linear_terms, nonlinear_terms)
            elif not new_arg is None:
                self.decompose(new_arg, linear_terms, nonlinear_terms)
            self._args.extend(linear_terms)
            if len(nonlinear_terms) > 0:
                #
                # We add nonlinear terms, but change the class type.  Note that
                # this doesn't change the type of _LinearViewSumExpression
                # objects that share a prefix of the underlying list.
                #
                self._args.extend(nonlinear_terms)
                self.__class__ = _ViewSumExpression
        #
        self._nargs = len(self._args)
        if not new_arg is None and new_arg.__class__ in pyomo5_expression_types:
            new_arg._is_owned = True
        return self

    def is_constant(self):
        for arg in islice(self._args, self._nargs):
            if not arg[1] is None:
                return False
        return True

    def _potentially_variable(self):
        global pyomo5_variable_types
        if pyomo5_variable_types is None:
            from pyomo.core.base import _VarData, _GeneralVarData, SimpleVar
            from pyomo.core.kernel.component_variable import IVariable, variable
            pyomo5_variable_types = set([_VarData, _GeneralVarData, IVariable, variable, SimpleVar])
            _LinearExpression.vtypes = pyomo5_variable_types

        for arg in islice(self._args, self._nargs):
            if not arg[1] is None:
                return True
        return False

    def _to_string_skip(self, _idx):
        return  _idx == 0 and \
                self._args[0][1] is None and \
                self._args[0][0].__class__ in native_numeric_types and \
                isclose(self._args[0][0], 0)


class _MutableMultiSumExpression(_SumExpression):
    """An object that defines a summation with 1 or more terms and a constant term."""

    __slots__ = ()
    PRECEDENCE = 6

    def __init__(self, args):
        self._args = list(args)
        if _getrefcount_available:
            self._is_owned = UNREFERENCED_EXPR_COUNT
        else:
            self._is_owned = False
            for arg in args:
                if arg.__class__ in pyomo5_expression_types:
                    arg._is_owned = True

    def nargs(self):
        return len(self._args)

    def _precedence(self):
        return _MutableMultiSumExpression.PRECEDENCE

    def _apply_operation(self, result):
        return sum(result)

    def getname(self, *args, **kwds):
        return 'multisum'

    def is_constant(self):
        return len(self._args) <= 1

    def _potentially_variable(self):
        return len(self._args) > 1

    def _to_string_skip(self, _idx):
        return  _idx == 0 and \
                self._args[0].__class__ in native_numeric_types and \
                isclose(self._args[0], 0)


class _MultiSumExpression(_MutableMultiSumExpression):
    """A temporary object that defines a summation with 1 or more terms and a constant term."""
    
    __slots__ = ()


class _CompressedSumExpression(_MutableMultiSumExpression):
    """A temporary object that defines a summation with 1 or more terms and a constant term."""
    
    __slots__ = ()


class _QuadraticExpression(_ExpressionBase):
    __slots__ = ('constant',          # The constant term
                 'linear_coefs',      # Linear coefficients
                 'linear_vars',       # Linear variables
                 'quadratic_coefs',   # Quadratic coefficients
                 'quadratic_vars')    # Quadratic variables

    PRECEDENCE = 6

    def __init__(self):
        self.constant = 0
        self.linear_coefs = []
        self.linear_vars = []
        self.quadratic_coefs = []
        self.quadratic_vars = []
        self._args = tuple()
        if _getrefcount_available:
            self._is_owned = UNREFERENCED_EXPR_COUNT
        else:
            self._is_owned = False

    def nargs(self):
        return 0

    def _clone(self, args=None):
        repn = self.__class__()
        repn.constant = deepcopy(self.constant)
        repn.linear_coefs = deepcopy(self.linear_coefs)
        repn.linear_vars = deepcopy(self.linear_vars)
        repn.quadratic_coefs = deepcopy(self.quadratic_coefs)
        repn.quadratic_vars = deepcopy(self.quadratic_vars)
        return repn

    def getname(self, *args, **kwds):
        return 'sum'

    def _polynomial_degree(self, result):
        if len(self.quadratic_vars) > 0:
            return 2
        elif len(self.linear_vars) > 0:
            return 1
        return 0

    def is_constant(self):
        return len(self.quadratic_vars) == 0 and len(self.linear_vars) == 0

    def is_fixed(self):
        if len(self.linear_vars) == 0 and len(self.quadratic_vars) == 0:
            return True
        for v,w in self.quadratic_vars:
            if not (v.fixed or w.fixed):
                return False
        for v in self.linear_vars:
            if not v.fixed:
                return False
        return True

    def _potentially_variable(self):
        return len(self.quadratic_vars) > 0 or len(self.linear_vars) > 0


class _StaticQuadraticExpression(_QuadraticExpression):
    __slots__ = ()


pyomo5_multisum_types = set([
        _ViewSumExpression,
        _MutableViewSumExpression,
        #_MutableMultiSumExpression,
        #_MultiSumExpression,
        #_CompressedSumExpression
        ])
pyomo5_sum_types = set([
        _SumExpression,
        _Constant_SumExpression,
        _NPV_SumExpression
        ])

