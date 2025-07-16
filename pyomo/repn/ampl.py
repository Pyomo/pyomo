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

import ctypes
import math
import operator

from collections import deque
from operator import itemgetter

from pyomo.common.deprecation import deprecation_warning
from pyomo.common.errors import DeveloperError, InfeasibleConstraintException, MouseTrap
from pyomo.common.numeric_types import (
    native_complex_types,
    native_numeric_types,
    native_types,
    value,
)


from pyomo.core.base import Expression
from pyomo.core.expr import (
    NegationExpression,
    ProductExpression,
    DivisionExpression,
    PowExpression,
    AbsExpression,
    UnaryFunctionExpression,
    MonomialTermExpression,
    LinearExpression,
    SumExpression,
    EqualityExpression,
    InequalityExpression,
    RangedExpression,
    Expr_ifExpression,
    ExternalFunctionExpression,
)
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor, _EvaluationVisitor
from pyomo.repn.util import (
    BeforeChildDispatcher,
    ExitNodeDispatcher,
    ExprType,
    InvalidNumber,
    apply_node_operation,
    complex_number_error,
    nan,
    sum_like_expression_types,
)


_CONSTANT = ExprType.CONSTANT
_MONOMIAL = ExprType.MONOMIAL
_GENERAL = ExprType.GENERAL

# Feasibility tolerance for trivial (fixed) constraints
TOL = 1e-8


def _create_strict_inequality_map(vars_):
    vars_['strict_inequality_map'] = {
        True: vars_['less_than'],
        False: vars_['less_equal'],
        (True, True): (vars_['less_than'], vars_['less_than']),
        (True, False): (vars_['less_than'], vars_['less_equal']),
        (False, True): (vars_['less_equal'], vars_['less_than']),
        (False, False): (vars_['less_equal'], vars_['less_equal']),
    }


class TextNLDebugTemplate(object):
    unary = {
        'log': 'o43\t#log\n',
        'log10': 'o42\t#log10\n',
        'sin': 'o41\t#sin\n',
        'cos': 'o46\t#cos\n',
        'tan': 'o38\t#tan\n',
        'sinh': 'o40\t#sinh\n',
        'cosh': 'o45\t#cosh\n',
        'tanh': 'o37\t#tanh\n',
        'asin': 'o51\t#asin\n',
        'acos': 'o53\t#acos\n',
        'atan': 'o49\t#atan\n',
        'exp': 'o44\t#exp\n',
        'sqrt': 'o39\t#sqrt\n',
        'asinh': 'o50\t#asinh\n',
        'acosh': 'o52\t#acosh\n',
        'atanh': 'o47\t#atanh\n',
        'ceil': 'o14\t#ceil\n',
        'floor': 'o13\t#floor\n',
    }

    binary_sum = 'o0\t#+\n'
    product = 'o2\t#*\n'
    division = 'o3\t# /\n'
    pow = 'o5\t#^\n'
    abs = 'o15\t# abs\n'
    negation = 'o16\t#-\n'
    nary_sum = 'o54\t# sumlist\n%d\t# (n)\n'
    exprif = 'o35\t# if\n'
    and_expr = 'o21\t# and\n'
    less_than = 'o22\t# lt\n'
    less_equal = 'o23\t# le\n'
    equality = 'o24\t# eq\n'
    external_fcn = 'f%d %d%s\n'
    # NOTE: to support scaling and substitutions, we do NOT include the
    # 'v' or the EOL here:
    var = '%s'
    const = 'n%s\n'
    string = 'h%d:%s\n'
    monomial = product + const + var.replace('%', '%%')
    multiplier = product + const

    _create_strict_inequality_map(vars())


nl_operators = {
    0: (2, operator.add),
    2: (2, operator.mul),
    3: (2, operator.truediv),
    5: (2, operator.pow),
    15: (1, operator.abs),
    16: (1, operator.neg),
    54: (None, lambda *x: sum(x)),
    35: (3, lambda a, b, c: b if a else c),
    21: (2, operator.and_),
    22: (2, operator.lt),
    23: (2, operator.le),
    24: (2, operator.eq),
    43: (1, math.log),
    42: (1, math.log10),
    41: (1, math.sin),
    46: (1, math.cos),
    38: (1, math.tan),
    40: (1, math.sinh),
    45: (1, math.cosh),
    37: (1, math.tanh),
    51: (1, math.asin),
    53: (1, math.acos),
    49: (1, math.atan),
    44: (1, math.exp),
    39: (1, math.sqrt),
    50: (1, math.asinh),
    52: (1, math.acosh),
    47: (1, math.atanh),
    14: (1, math.ceil),
    13: (1, math.floor),
}


def _strip_template_comments(vars_, base_):
    vars_['unary'] = {
        k: v[: v.find('\t#')] + '\n' if v[-1] == '\n' else ''
        for k, v in base_.unary.items()
    }
    for k, v in base_.__dict__.items():
        if type(v) is str and '\t#' in v:
            v_lines = v.split('\n')
            for i, l in enumerate(v_lines):
                comment_start = l.find('\t#')
                if comment_start >= 0:
                    v_lines[i] = l[:comment_start]
            vars_[k] = '\n'.join(v_lines)


def _inv2str(val):
    return f"{val._str() if hasattr(val, '_str') else val}"


# The "standard" text mode template is the debugging template with the
# comments removed
class TextNLTemplate(TextNLDebugTemplate):
    _strip_template_comments(vars(), TextNLDebugTemplate)
    _create_strict_inequality_map(vars())


class NLFragment(object):
    """This is a mock "component" for the nl portion of a named Expression.

    It is used internally in the writer when requesting symbolic solver
    labels so that we can generate meaningful names for the nonlinear
    portion of an Expression component.

    """

    __slots__ = ('_repn', '_node')

    def __init__(self, repn, node):
        self._repn = repn
        self._node = node

    @property
    def name(self):
        return 'nl(' + self._node.name + ')'


class AMPLRepn(object):
    """The "compiled" representation of an expression in AMPL NL format.

    This stores a compiled form of an expression in the AMPL "NL"
    format.  The data structure contains 6 fields:

    Attributes
    ----------
    mult : float

        A constant multiplier applied to this expression.  The
        :py:class`AMPLRepn` returned by the :py:class`AMPLRepnVisitor`
        should always have `mult` == 1.

    const : float

        The constant portion of this expression

    linear : Dict[int, float] or None

        Mapping of `id(VarData)` to linear coefficient

    nonlinear : Tuple[str, List[int]] or List[Tuple[str, List[int]]] or None

        The general nonlinear portion of the compiled expression as a
        tuple of two parts:

           - the nl template string: this is the NL string with
             placeholders (``%s``) for all the variables that appear in
             the expression.

           - an iterable if the :class:`VarData` IDs that correspond to the
             placeholders in the nl template string

        This is `None` if there is no general nonlinear part of the
        expression.  Note that this can be a list of tuple fragments
        within AMPLRepnVisitor, but that list is concatenated to a
        single tuple when exiting the `AMPLRepnVisitor`.

    named_exprs : Set[int]

        A set of IDs point to named expressions (:py:class:`Expression`)
        objects appearing in this expression.

    nl : Tuple[str, Iterable[int]]

        This holds the complete compiled representation of this
        expression (including multiplier, constant, linear terms, and
        nonlinear fragment) using the same format as the `nonlinear`
        attribute.  This field (if not None) should be considered
        authoritative, as there are NL fragments that are not
        representable by {mult, const, linear, nonlinear} (e.g., string
        arguments).

    """

    __slots__ = ('nl', 'mult', 'const', 'linear', 'nonlinear', 'named_exprs')

    template = TextNLTemplate

    def __init__(self, const, linear, nonlinear):
        self.nl = None
        self.mult = 1
        self.const = const
        self.linear = linear
        if nonlinear is None:
            self.nonlinear = self.named_exprs = None
        else:
            nl, nl_args, self.named_exprs = nonlinear
            self.nonlinear = nl, nl_args

    def __str__(self):
        return (
            f'AMPLRepn(mult={self.mult}, const={self.const}, '
            f'linear={self.linear}, nonlinear={self.nonlinear}, '
            f'nl={self.nl}, named_exprs={self.named_exprs})'
        )

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return isinstance(other.__class__, AMPLRepn) and (
            self.nl == other.nl
            and self.mult == other.mult
            and self.const == other.const
            and self.linear == other.linear
            and self.nonlinear == other.nonlinear
            and self.named_exprs == other.named_exprs
        )

    def __hash__(self):
        # Approximation of the Python default object hash
        # (4 LSB are rolled to the MSB to reduce hash collisions)
        return id(self) // 16 + (
            (id(self) & 15) << 8 * ctypes.sizeof(ctypes.c_void_p) - 4
        )

    def duplicate(self):
        ans = self.__class__.__new__(self.__class__)
        ans.nl = self.nl
        ans.mult = self.mult
        ans.const = self.const
        ans.linear = None if self.linear is None else dict(self.linear)
        ans.nonlinear = self.nonlinear
        ans.named_exprs = None if self.named_exprs is None else set(self.named_exprs)
        return ans

    def compile_repn(self, prefix='', args=None, named_exprs=None):
        template = self.template
        if self.mult != 1:
            if self.mult == -1:
                prefix += template.negation
            else:
                prefix += template.multiplier % self.mult
            self.mult = 1
        if self.named_exprs is not None:
            if named_exprs is None:
                named_exprs = set(self.named_exprs)
            else:
                named_exprs.update(self.named_exprs)
        if self.nl is not None:
            # This handles both named subexpressions and embedded
            # non-numeric (e.g., string) arguments.
            nl, nl_args = self.nl
            if prefix:
                nl = prefix + nl
            if args is not None:
                assert args is not nl_args
                args.extend(nl_args)
            else:
                args = list(nl_args)
            if nl_args:
                # For string arguments, nl_args is an empty tuple and
                # self.named_exprs is None.  For named subexpressions,
                # we are guaranteed that named_exprs is NOT None.  We
                # need to ensure that the named subexpression that we
                # are returning is added to the named_exprs set.
                named_exprs.update(nl_args)
            return nl, args, named_exprs

        if args is None:
            args = []
        if self.linear:
            nterms = -len(args)
            _v_template = template.var
            _m_template = template.monomial
            # Because we are compiling this expression (into a NL
            # expression), we will go ahead and filter the 0*x terms
            # from the expression.  Note that the args are accumulated
            # by side-effect, which prevents iterating over the linear
            # terms twice.
            nl_sum = ''.join(
                args.append(v) or (_v_template if c == 1 else _m_template % c)
                for v, c in self.linear.items()
                if c
            )
            nterms += len(args)
        else:
            nterms = 0
            nl_sum = ''
        if self.nonlinear:
            if self.nonlinear.__class__ is list:
                nterms += len(self.nonlinear)
                nl_sum += ''.join(map(itemgetter(0), self.nonlinear))
                deque(map(args.extend, map(itemgetter(1), self.nonlinear)), maxlen=0)
            else:
                nterms += 1
                nl_sum += self.nonlinear[0]
                args.extend(self.nonlinear[1])
        if self.const:
            nterms += 1
            nl_sum += template.const % self.const

        if nterms > 2:
            return (prefix + (template.nary_sum % nterms) + nl_sum, args, named_exprs)
        elif nterms == 2:
            return prefix + template.binary_sum + nl_sum, args, named_exprs
        elif nterms == 1:
            return prefix + nl_sum, args, named_exprs
        else:  # nterms == 0
            return prefix + (template.const % 0), args, named_exprs

    def compile_nonlinear_fragment(self):
        if not self.nonlinear:
            self.nonlinear = None
            return
        args = []
        nterms = len(self.nonlinear)
        nl_sum = ''.join(map(itemgetter(0), self.nonlinear))
        deque(map(args.extend, map(itemgetter(1), self.nonlinear)), maxlen=0)

        if nterms > 2:
            self.nonlinear = (self.template.nary_sum % nterms) + nl_sum, args
        elif nterms == 2:
            self.nonlinear = self.template.binary_sum + nl_sum, args
        else:  # nterms == 1:
            self.nonlinear = nl_sum, args

    def append(self, other):
        """Append a child result from acceptChildResult

        Notes
        -----
        This method assumes that the operator was "+". It is implemented
        so that we can directly use an AMPLRepn() as a data object in
        the expression walker (thereby avoiding the function call for a
        custom callback)

        """
        # Note that self.mult will always be 1 (we only call append()
        # within a sum, so there is no opportunity for self.mult to
        # change). Omitting the assertion for efficiency.
        # assert self.mult == 1
        _type = other[0]
        if _type is _MONOMIAL:
            _, v, c = other
            if v in self.linear:
                self.linear[v] += c
            else:
                self.linear[v] = c
        elif _type is _GENERAL:
            _, other = other
            if other.nl is not None and other.nl[1]:
                if other.linear:
                    # This is a named expression with both a linear and
                    # nonlinear component.  We want to merge it with
                    # this AMPLRepn, preserving the named expression for
                    # only the nonlinear component (merging the linear
                    # component with this AMPLRepn).
                    pass
                else:
                    # This is a nonlinear-only named expression,
                    # possibly with a multiplier that is not 1.  Compile
                    # it and append it (this both resolves the
                    # multiplier, and marks the named expression as
                    # having been used)
                    other = other.compile_repn('', None, self.named_exprs)
                    nl, nl_args, self.named_exprs = other
                    self.nonlinear.append((nl, nl_args))
                    return
            if other.named_exprs is not None:
                if self.named_exprs is None:
                    self.named_exprs = set(other.named_exprs)
                else:
                    self.named_exprs.update(other.named_exprs)
            if other.mult != 1:
                mult = other.mult
                self.const += mult * other.const
                if other.linear:
                    linear = self.linear
                    for v, c in other.linear.items():
                        if v in linear:
                            linear[v] += c * mult
                        else:
                            linear[v] = c * mult
                if other.nonlinear:
                    if other.nonlinear.__class__ is list:
                        other.compile_nonlinear_fragment()
                    if mult == -1:
                        prefix = self.template.negation
                    else:
                        prefix = self.template.multiplier % mult
                    self.nonlinear.append(
                        (prefix + other.nonlinear[0], other.nonlinear[1])
                    )
            else:
                self.const += other.const
                if other.linear:
                    linear = self.linear
                    for v, c in other.linear.items():
                        if v in linear:
                            linear[v] += c
                        else:
                            linear[v] = c
                if other.nonlinear:
                    if other.nonlinear.__class__ is list:
                        self.nonlinear.extend(other.nonlinear)
                    else:
                        self.nonlinear.append(other.nonlinear)
        elif _type is _CONSTANT:
            self.const += other[1]

    def to_expr(self, var_map):
        if self.nl is not None or self.nonlinear is not None:
            # TODO: support converting general nonlinear expressions
            # back to Pyomo expressions.  This will require an AMPL
            # parser.
            raise MouseTrap("Cannot convert nonlinear AMPLRepn to Pyomo Expression")
        if self.linear:
            # Explicitly generate the LinearExpression.  At time of
            # writing, this is about 40% faster than standard operator
            # overloading for O(1000) element sums
            ans = LinearExpression(
                [coef * var_map[vid] for vid, coef in self.linear.items()]
            )
            ans += self.const
        else:
            ans = self.const
        return ans * self.mult


class DebugAMPLRepn(AMPLRepn):
    """An `AMPLRepn` that uses the "debug" (annotated) NL format

    This is identical to the :py:class:`AMPLRepn` class, except it is
    built using the `TextNLDebugTemplate` formatting template.  This
    format includes descriptions of the operators and variable /
    expression names in the NL text.

    """

    __slots__ = ()
    template = TextNLDebugTemplate


def handle_negation_node(visitor, node, arg1):
    if arg1[0] is _MONOMIAL:
        return (_MONOMIAL, arg1[1], -1 * arg1[2])
    elif arg1[0] is _GENERAL:
        arg1[1].mult *= -1
        return arg1
    elif arg1[0] is _CONSTANT:
        return (_CONSTANT, -1 * arg1[1])
    else:
        raise RuntimeError("%s: %s" % (type(arg1[0]), arg1))


def handle_product_node(visitor, node, arg1, arg2):
    if arg2[0] is _CONSTANT:
        arg2, arg1 = arg1, arg2
    if arg1[0] is _CONSTANT:
        mult = arg1[1]
        if not mult:
            # simplify multiplication by 0 (if arg2 is zero, the
            # simplification happens when we evaluate the constant
            # below).  Note that this is not IEEE-754 compliant, and
            # will map 0*inf and 0*nan to 0 (and not to nan).  We are
            # including this for backwards compatibility with the NLv1
            # writer, but arguably we should deprecate/remove this
            # "feature" in the future.
            if arg2[0] is _CONSTANT:
                _prod = mult * arg2[1]
                if _prod:
                    deprecation_warning(
                        f"Encountered {mult}*{_inv2str(arg2[1])} in expression tree.  "
                        "Mapping the NaN result to 0 for compatibility "
                        "with the nl_v1 writer.  In the future, this NaN "
                        "will be preserved/emitted to comply with IEEE-754.",
                        version='6.4.3',
                    )
                    _prod = 0
                return (_CONSTANT, _prod)
            return arg1
        if mult == 1:
            return arg2
        elif arg2[0] is _MONOMIAL:
            if mult != mult:
                # This catches mult (i.e., arg1) == nan
                return arg1
            return (_MONOMIAL, arg2[1], mult * arg2[2])
        elif arg2[0] is _GENERAL:
            if mult != mult:
                # This catches mult (i.e., arg1) == nan
                return arg1
            arg2[1].mult *= mult
            return arg2
        elif arg2[0] is _CONSTANT:
            if not arg2[1]:
                # Simplify multiplication by 0; see note above about
                # IEEE-754 incompatibility.
                _prod = mult * arg2[1]
                if _prod:
                    deprecation_warning(
                        f"Encountered {_inv2str(mult)}*{arg2[1]} in expression tree.  "
                        "Mapping the NaN result to 0 for compatibility "
                        "with the nl_v1 writer.  In the future, this NaN "
                        "will be preserved/emitted to comply with IEEE-754.",
                        version='6.4.3',
                    )
                    _prod = 0
                return (_CONSTANT, _prod)
            return (_CONSTANT, mult * arg2[1])
    nonlin = visitor.node_result_to_amplrepn(arg1).compile_repn(
        visitor.template.product
    )
    nonlin = visitor.node_result_to_amplrepn(arg2).compile_repn(*nonlin)
    return (_GENERAL, visitor.Result(0, None, nonlin))


def handle_division_node(visitor, node, arg1, arg2):
    if arg2[0] is _CONSTANT:
        div = arg2[1]
        if div == 1:
            return arg1
        if arg1[0] is _MONOMIAL:
            tmp = apply_node_operation(node, (arg1[2], div))
            if tmp != tmp:
                # This catches if the coefficient division results in nan
                return _CONSTANT, tmp
            return (_MONOMIAL, arg1[1], tmp)
        elif arg1[0] is _GENERAL:
            tmp = apply_node_operation(node, (arg1[1].mult, div))
            if tmp != tmp:
                # This catches if the multiplier division results in nan
                return _CONSTANT, tmp
            arg1[1].mult = tmp
            return arg1
        elif arg1[0] is _CONSTANT:
            return _CONSTANT, apply_node_operation(node, (arg1[1], div))
    elif arg1[0] is _CONSTANT and not arg1[1]:
        return _CONSTANT, 0
    nonlin = visitor.node_result_to_amplrepn(arg1).compile_repn(
        visitor.template.division
    )
    nonlin = visitor.node_result_to_amplrepn(arg2).compile_repn(*nonlin)
    return (_GENERAL, visitor.Result(0, None, nonlin))


def handle_pow_node(visitor, node, arg1, arg2):
    if arg2[0] is _CONSTANT:
        if arg1[0] is _CONSTANT:
            ans = apply_node_operation(node, (arg1[1], arg2[1]))
            if ans.__class__ in native_complex_types:
                ans = complex_number_error(ans, visitor, node)
            return _CONSTANT, ans
        elif not arg2[1]:
            return _CONSTANT, 1
        elif arg2[1] == 1:
            return arg1
    nonlin = visitor.node_result_to_amplrepn(arg1).compile_repn(visitor.template.pow)
    nonlin = visitor.node_result_to_amplrepn(arg2).compile_repn(*nonlin)
    return (_GENERAL, visitor.Result(0, None, nonlin))


def handle_abs_node(visitor, node, arg1):
    if arg1[0] is _CONSTANT:
        return (_CONSTANT, abs(arg1[1]))
    nonlin = visitor.node_result_to_amplrepn(arg1).compile_repn(visitor.template.abs)
    return (_GENERAL, visitor.Result(0, None, nonlin))


def handle_unary_node(visitor, node, arg1):
    if arg1[0] is _CONSTANT:
        return _CONSTANT, apply_node_operation(node, (arg1[1],))
    nonlin = visitor.node_result_to_amplrepn(arg1).compile_repn(
        visitor.template.unary[node.name]
    )
    return (_GENERAL, visitor.Result(0, None, nonlin))


def handle_exprif_node(visitor, node, arg1, arg2, arg3):
    if arg1[0] is _CONSTANT:
        if arg1[1]:
            return arg2
        else:
            return arg3
    nonlin = visitor.node_result_to_amplrepn(arg1).compile_repn(visitor.template.exprif)
    nonlin = visitor.node_result_to_amplrepn(arg2).compile_repn(*nonlin)
    nonlin = visitor.node_result_to_amplrepn(arg3).compile_repn(*nonlin)
    return (_GENERAL, visitor.Result(0, None, nonlin))


def handle_equality_node(visitor, node, arg1, arg2):
    if arg1[0] is _CONSTANT and arg2[0] is _CONSTANT:
        return (_CONSTANT, arg1[1] == arg2[1])
    nonlin = visitor.node_result_to_amplrepn(arg1).compile_repn(
        visitor.template.equality
    )
    nonlin = visitor.node_result_to_amplrepn(arg2).compile_repn(*nonlin)
    return (_GENERAL, visitor.Result(0, None, nonlin))


def handle_inequality_node(visitor, node, arg1, arg2):
    if arg1[0] is _CONSTANT and arg2[0] is _CONSTANT:
        return (_CONSTANT, node._apply_operation((arg1[1], arg2[1])))
    nonlin = visitor.node_result_to_amplrepn(arg1).compile_repn(
        visitor.template.strict_inequality_map[node.strict]
    )
    nonlin = visitor.node_result_to_amplrepn(arg2).compile_repn(*nonlin)
    return (_GENERAL, visitor.Result(0, None, nonlin))


def handle_ranged_inequality_node(visitor, node, arg1, arg2, arg3):
    if arg1[0] is _CONSTANT and arg2[0] is _CONSTANT and arg3[0] is _CONSTANT:
        return (_CONSTANT, node._apply_operation((arg1[1], arg2[1], arg3[1])))
    op = visitor.template.strict_inequality_map[node.strict]
    nl, args, named = visitor.node_result_to_amplrepn(arg1).compile_repn(
        visitor.template.and_expr + op[0]
    )
    nl2, args2, named = visitor.node_result_to_amplrepn(arg2).compile_repn(
        '', None, named
    )
    nl += nl2 + op[1] + nl2
    args.extend(args2)
    args.extend(args2)
    nonlin = visitor.node_result_to_amplrepn(arg3).compile_repn(nl, args, named)
    return (_GENERAL, visitor.Result(0, None, nonlin))


def handle_named_expression_node(visitor, node, arg1):
    _id = id(node)
    # Note that while named subexpressions ('defined variables' in the
    # ASL NL file vernacular) look like variables, they are not allowed
    # to appear in the 'linear' portion of a constraint / objective
    # definition.  We will return this as a "var" template, but
    # wrapped in the nonlinear portion of the expression tree.
    repn = visitor.node_result_to_amplrepn(arg1)

    # A local copy of the expression source list.  This will be updated
    # later if the same Expression node is encountered in another
    # expression tree.
    #
    # This is a 3-tuple [con_id, obj_id, substitute_expression].  If the
    # expression is used by more than 1 constraint / objective, then the
    # id is set to 0.  If it is not used by any, then it is None.
    # substitute_expression is a bool indicating if this named
    # subexpression tree should be directly substituted into any
    # expression tree that references this node (i.e., do NOT emit the V
    # line).
    expression_source = [None, None, False]
    # Record this common expression
    visitor.subexpression_cache[_id] = (
        # 0: the "component" that generated this expression ID
        node,
        # 1: the common subexpression (to be written out)
        repn,
        # 2: the source usage information for this subexpression:
        #    [(con_id, obj_id, substitute); see above]
        expression_source,
    )

    # As we will eventually need the compiled form of any nonlinear
    # expression, we will go ahead and compile it here.  We do not
    # do the same for the linear component as we will only need the
    # linear component compiled to a dict if we are emitting the
    # original (linear + nonlinear) V line (which will not happen if
    # the V line is part of a larger linear operator).
    if repn.nonlinear.__class__ is list:
        repn.compile_nonlinear_fragment()

    if not visitor.use_named_exprs:
        return _GENERAL, repn.duplicate()

    mult, repn.mult = repn.mult, 1
    if repn.named_exprs is None:
        repn.named_exprs = set()

    # When converting this shared subexpression to a (nonlinear)
    # node, we want to just reference this subexpression:
    repn.nl = (visitor.template.var, (_id,))

    if repn.nonlinear:
        if repn.linear:
            # If this expression has both linear and nonlinear
            # components, we will follow the ASL convention and break
            # the named subexpression into two named subexpressions: one
            # that is only the nonlinear component and one that has the
            # const/linear component (and references the first).  This
            # will allow us to propagate linear coefficients up from
            # named subexpressions when appropriate.
            sub_node = NLFragment(repn, node)
            sub_id = id(sub_node)
            sub_repn = visitor.Result(0, None, None)
            sub_repn.nonlinear = repn.nonlinear
            sub_repn.nl = (visitor.template.var, (sub_id,))
            sub_repn.named_exprs = set(repn.named_exprs)

            repn.named_exprs.add(sub_id)
            repn.nonlinear = sub_repn.nl

            # See above for the meaning of this source information
            nl_info = list(expression_source)
            visitor.subexpression_cache[sub_id] = (sub_node, sub_repn, nl_info)
            # It is important that the NL subexpression comes before the
            # main named expression: re-insert the original named
            # expression (so that the nonlinear sub_node comes first
            # when iterating over subexpression_cache)
            visitor.subexpression_cache[_id] = visitor.subexpression_cache.pop(_id)
        else:
            nl_info = expression_source
    else:
        repn.nonlinear = None
        if repn.linear:
            if (
                not repn.const
                and len(repn.linear) == 1
                and next(iter(repn.linear.values())) == 1
            ):
                # This Expression holds only a variable (multiplied by
                # 1).  Do not emit this as a named variable and instead
                # just inject the variable where this expression is
                # used.
                repn.nl = None
                expression_source[2] = True
        else:
            # This Expression holds only a constant.  Do not emit this
            # as a named variable and instead just inject the constant
            # where this expression is used.
            repn.nl = None
            expression_source[2] = True

    if mult != 1:
        repn.const *= mult
        if repn.linear:
            _lin = repn.linear
            for v in repn.linear:
                _lin[v] *= mult
        if repn.nonlinear:
            if mult == -1:
                prefix = visitor.template.negation
            else:
                prefix = visitor.template.multiplier % mult
            repn.nonlinear = prefix + repn.nonlinear[0], repn.nonlinear[1]

    if expression_source[2]:
        if repn.linear:
            assert len(repn.linear) == 1 and not repn.const
            return (_MONOMIAL,) + next(iter(repn.linear.items()))
        else:
            return (_CONSTANT, repn.const)

    return (_GENERAL, repn.duplicate())


def handle_external_function_node(visitor, node, *args):
    func = node._fcn._function
    # There is a special case for external functions: these are the only
    # expressions that can accept string arguments. As we currently pass
    # these as 'precompiled' GENERAL AMPLRepns, the normal trap for
    # constant subexpressions will miss string arguments.  We will catch
    # that case here by looking for NL fragments with no variable
    # references.  Note that the NL fragment is NOT the raw string
    # argument that we want to evaluate: the raw string is in the
    # `const` field.
    if all(
        arg[0] is _CONSTANT or (arg[0] is _GENERAL and arg[1].nl and not arg[1].nl[1])
        for arg in args
    ):
        arg_list = [arg[1] if arg[0] is _CONSTANT else arg[1].const for arg in args]
        return _CONSTANT, apply_node_operation(node, arg_list)
    if func in visitor.external_functions:
        if node._fcn._library != visitor.external_functions[func][1]._library:
            raise RuntimeError(
                "The same external function name (%s) is associated "
                "with two different libraries (%s through %s, and %s "
                "through %s).  The ASL solver will fail to link "
                "correctly."
                % (
                    func,
                    visitor.external_functions[func]._library,
                    visitor.external_functions[func]._library.name,
                    node._fcn._library,
                    node._fcn.name,
                )
            )
    else:
        visitor.external_functions[func] = (len(visitor.external_functions), node._fcn)
    comment = f'\t#{node.local_name}' if visitor.symbolic_solver_labels else ''
    nl = visitor.template.external_fcn % (
        visitor.external_functions[func][0],
        len(args),
        comment,
    )
    arg_ids = []
    named_exprs = set()
    for arg in args:
        _id = id(arg)
        arg_ids.append(_id)
        visitor.subexpression_cache[_id] = (
            arg,
            visitor.Result(
                0,
                None,
                visitor.node_result_to_amplrepn(arg).compile_repn(
                    named_exprs=named_exprs
                ),
            ),
            (None, None, True),
        )
    if not named_exprs:
        named_exprs = None
    return (
        _GENERAL,
        visitor.Result(0, None, (nl + '%s' * len(arg_ids), arg_ids, named_exprs)),
    )


_operator_handles = ExitNodeDispatcher(
    {
        NegationExpression: handle_negation_node,
        ProductExpression: handle_product_node,
        DivisionExpression: handle_division_node,
        PowExpression: handle_pow_node,
        AbsExpression: handle_abs_node,
        UnaryFunctionExpression: handle_unary_node,
        Expr_ifExpression: handle_exprif_node,
        EqualityExpression: handle_equality_node,
        InequalityExpression: handle_inequality_node,
        RangedExpression: handle_ranged_inequality_node,
        Expression: handle_named_expression_node,
        ExternalFunctionExpression: handle_external_function_node,
        # These are handled explicitly in beforeChild():
        # LinearExpression: handle_linear_expression,
        # SumExpression: handle_sum_expression,
        #
        # Note: MonomialTermExpression is only hit when processing NPV
        # subexpressions that raise errors (e.g., log(0) * m.x), so no
        # special processing is needed [it is just a product expression]
        MonomialTermExpression: handle_product_node,
    }
)


class AMPLBeforeChildDispatcher(BeforeChildDispatcher):
    __slots__ = ()

    def __init__(self):
        # Special linear / summation expressions
        self[MonomialTermExpression] = self._before_monomial
        self[LinearExpression] = self._before_linear
        self[SumExpression] = self._before_general_expression

    @staticmethod
    def _record_var(visitor, var):
        # We always add all indices to the var_map at once so that
        # we can honor deterministic ordering of unordered sets
        # (because the user could have iterated over an unordered
        # set when constructing an expression, thereby altering the
        # order in which we would see the variables)
        vm = visitor.var_map
        try:
            _iter = var.parent_component().values(visitor.sorter)
        except AttributeError:
            # Note that this only works for the AML, as kernel does not
            # provide a parent_component()
            _iter = (var,)
        for v in _iter:
            if v.fixed:
                continue
            vm[id(v)] = v

    @staticmethod
    def _before_string(visitor, child):
        visitor.encountered_string_arguments = True
        ans = visitor.Result(child, None, None)
        ans.nl = (visitor.template.string % (len(child), child), ())
        return False, (_GENERAL, ans)

    @staticmethod
    def _before_var(visitor, child):
        _id = id(child)
        if _id not in visitor.var_map:
            if child.fixed:
                if _id not in visitor.fixed_vars:
                    visitor.cache_fixed_var(_id, child)
                return False, (_CONSTANT, visitor.fixed_vars[_id])
            _before_child_handlers._record_var(visitor, child)
        return False, (_MONOMIAL, _id, 1)

    @staticmethod
    def _before_monomial(visitor, child):
        #
        # The following are performance optimizations for common
        # situations (Monomial terms and Linear expressions)
        #
        arg1, arg2 = child._args_
        if arg1.__class__ not in native_types:
            try:
                arg1 = visitor.check_constant(visitor.evaluate(arg1), arg1)
            except (ValueError, ArithmeticError):
                return True, None

        # Trap multiplication by 0 and nan.
        if not arg1:
            if arg2.fixed:
                _id = id(arg2)
                if _id not in visitor.fixed_vars:
                    visitor.cache_fixed_var(id(arg2), arg2)
                arg2 = visitor.fixed_vars[_id]
                if arg2 != arg2:
                    deprecation_warning(
                        f"Encountered {arg1}*{_inv2str(arg2)} in expression tree.  "
                        "Mapping the NaN result to 0 for compatibility "
                        "with the nl_v1 writer.  In the future, this NaN "
                        "will be preserved/emitted to comply with IEEE-754.",
                        version='6.4.3',
                    )
            return False, (_CONSTANT, arg1)

        _id = id(arg2)
        if _id not in visitor.var_map:
            if arg2.fixed:
                if _id not in visitor.fixed_vars:
                    visitor.cache_fixed_var(_id, arg2)
                return False, (_CONSTANT, arg1 * visitor.fixed_vars[_id])
            _before_child_handlers._record_var(visitor, arg2)
        return False, (_MONOMIAL, _id, arg1)

    @staticmethod
    def _before_linear(visitor, child):
        # Because we are going to modify the LinearExpression in this
        # walker, we need to make a copy of the arg list from the original
        # expression tree.
        var_map = visitor.var_map
        const = 0
        linear = {}
        for arg in child.args:
            if arg.__class__ is MonomialTermExpression:
                arg1, arg2 = arg._args_
                if arg1.__class__ not in native_types:
                    try:
                        arg1 = visitor.check_constant(visitor.evaluate(arg1), arg1)
                    except (ValueError, ArithmeticError):
                        return True, None

                # Trap multiplication by 0 and nan.
                if not arg1:
                    if arg2.fixed:
                        arg2 = visitor.check_constant(arg2.value, arg2)
                        if arg2 != arg2:
                            deprecation_warning(
                                f"Encountered {arg1}*{_inv2str(arg2)} in expression "
                                "tree.  Mapping the NaN result to 0 for compatibility "
                                "with the nl_v1 writer.  In the future, this NaN "
                                "will be preserved/emitted to comply with IEEE-754.",
                                version='6.4.3',
                            )
                    continue

                _id = id(arg2)
                if _id not in var_map:
                    if arg2.fixed:
                        if _id not in visitor.fixed_vars:
                            visitor.cache_fixed_var(_id, arg2)
                        const += arg1 * visitor.fixed_vars[_id]
                        continue
                    _before_child_handlers._record_var(visitor, arg2)
                    linear[_id] = arg1
                elif _id in linear:
                    linear[_id] += arg1
                else:
                    linear[_id] = arg1
            elif arg.__class__ in native_types:
                const += arg
            elif arg.is_variable_type():
                _id = id(arg)
                if _id not in var_map:
                    if arg.fixed:
                        if _id not in visitor.fixed_vars:
                            visitor.cache_fixed_var(_id, arg)
                        const += visitor.fixed_vars[_id]
                        continue
                    _before_child_handlers._record_var(visitor, arg)
                    linear[_id] = 1
                elif _id in linear:
                    linear[_id] += 1
                else:
                    linear[_id] = 1
            else:
                try:
                    const += visitor.check_constant(visitor.evaluate(arg), arg)
                except (ValueError, ArithmeticError):
                    return True, None

        if linear:
            return False, (_GENERAL, visitor.Result(const, linear, None))
        else:
            return False, (_CONSTANT, const)

    @staticmethod
    def _before_named_expression(visitor, child):
        _id = id(child)
        if _id in visitor.subexpression_cache:
            obj, repn, info = visitor.subexpression_cache[_id]
            if info[2]:
                if repn.linear:
                    return False, (_MONOMIAL, next(iter(repn.linear)), 1)
                else:
                    return False, (_CONSTANT, repn.const)
            return False, (_GENERAL, repn.duplicate())
        else:
            return True, None


_before_child_handlers = AMPLBeforeChildDispatcher()


class AMPLRepnVisitor(StreamBasedExpressionVisitor):
    def __init__(
        self,
        subexpression_cache,
        external_functions,
        var_map,
        used_named_expressions,
        symbolic_solver_labels,
        use_named_exprs,
        sorter,
    ):
        super().__init__()
        self.subexpression_cache = subexpression_cache
        self.external_functions = external_functions
        self.active_expression_source = None
        self.var_map = var_map
        self.used_named_expressions = used_named_expressions
        self.symbolic_solver_labels = symbolic_solver_labels
        self.use_named_exprs = use_named_exprs
        self.encountered_string_arguments = False
        self.fixed_vars = {}
        self._eval_expr_visitor = _EvaluationVisitor(True)
        self.evaluate = self._eval_expr_visitor.dfs_postorder_stack
        self.sorter = sorter

        if symbolic_solver_labels:
            self.Result = DebugAMPLRepn
        else:
            self.Result = AMPLRepn
        self.template = self.Result.template

    def check_constant(self, ans, obj):
        if ans.__class__ not in native_numeric_types:
            # None can be returned from uninitialized Var/Param objects
            if ans is None:
                return InvalidNumber(
                    None, f"'{obj}' evaluated to a nonnumeric value '{ans}'"
                )
            if ans.__class__ is InvalidNumber:
                return ans
            elif ans.__class__ in native_complex_types:
                return complex_number_error(ans, self, obj)
            else:
                # It is possible to get other non-numeric types.  Most
                # common are bool and 1-element numpy.array().  We will
                # attempt to convert the value to a float before
                # proceeding.
                #
                # Note that as of NumPy 1.25, blindly casting a
                # 1-element ndarray to a float will generate a
                # deprecation warning.  We will explicitly test for
                # that, but want to do the test without triggering the
                # numpy import
                for cls in ans.__class__.__mro__:
                    if cls.__name__ == 'ndarray' and cls.__module__ == 'numpy':
                        if len(ans) == 1:
                            ans = ans[0]
                        break
                # TODO: we should check bool and warn/error (while bool is
                # convertible to float in Python, they have very
                # different semantic meanings in Pyomo).
                try:
                    ans = float(ans)
                except:
                    return InvalidNumber(
                        ans, f"'{obj}' evaluated to a nonnumeric value '{ans}'"
                    )
        if ans != ans:
            return InvalidNumber(
                nan, f"'{obj}' evaluated to a nonnumeric value '{ans}'"
            )
        return ans

    def cache_fixed_var(self, _id, child):
        val = self.check_constant(child.value, child)
        lb, ub = child.bounds
        if (lb is not None and lb - val > TOL) or (ub is not None and ub - val < -TOL):
            raise InfeasibleConstraintException(
                "model contains a trivially infeasible "
                f"variable '{child.name}' (fixed value "
                f"{val} outside bounds [{lb}, {ub}])."
            )
        self.fixed_vars[_id] = self.check_constant(child.value, child)

    def node_result_to_amplrepn(self, data):
        if data[0] is _GENERAL:
            return data[1]
        elif data[0] is _MONOMIAL:
            _, v, c = data
            if c:
                return self.Result(0, {v: c}, None)
            else:
                return self.Result(0, None, None)
        elif data[0] is _CONSTANT:
            return self.Result(data[1], None, None)
        else:
            raise DeveloperError("unknown result type")

    def initializeWalker(self, expr):
        expr, src, src_idx, self.expression_scaling_factor = expr
        self.active_expression_source = (src_idx, id(src))
        walk, result = self.beforeChild(None, expr, 0)
        if not walk:
            return False, self.finalizeResult(result)
        return True, expr

    def beforeChild(self, node, child, child_idx):
        return _before_child_handlers[child.__class__](self, child)

    def enterNode(self, node):
        # SumExpression are potentially large nary operators.  Directly
        # populate the result
        if node.__class__ in sum_like_expression_types:
            data = self.Result(0, {}, None)
            data.nonlinear = []
            return node.args, data
        else:
            return node.args, []

    def exitNode(self, node, data):
        if data.__class__ is self.Result:
            # If the summation resulted in a constant, return the constant
            if data.linear or data.nonlinear or data.nl:
                return (_GENERAL, data)
            else:
                return (_CONSTANT, data.const)
        #
        # General expressions...
        #
        return _operator_handles[node.__class__](self, node, *data)

    def finalizeResult(self, result):
        ans = self.node_result_to_amplrepn(result)

        # Multiply the expression by the scaling factor provided by the caller
        ans.mult *= self.expression_scaling_factor

        # If this was a nonlinear named expression, and that expression
        # has no linear portion, then we will directly use this as a
        # named expression.  We need to mark that the expression was
        # used and return it as a simple nonlinear expression pointing
        # to this named expression.  In all other cases, we will return
        # the processed representation (which will reference the
        # nonlinear-only named subexpression - if it exists - but not
        # this outer named expression).  This prevents accidentally
        # recharacterizing variables that only appear linearly as
        # nonlinear variables.
        if ans.nl is not None:
            if not ans.nl[1]:
                raise ValueError("Numeric expression resolved to a string constant")
            # This *is* a named subexpression.  If there is no linear
            # component, then replace this expression with the named
            # expression.  The mult will be handled later.  We know that
            # the const is built into the nonlinear expression, because
            # it cannot be changed "in place" (only through addition,
            # which would have "cleared" the nl attribute)
            if not ans.linear:
                ans.named_exprs.update(ans.nl[1])
                ans.nonlinear = ans.nl
                ans.const = 0
            else:
                # This named expression has both a linear and a
                # nonlinear component, and possibly a multiplier and
                # constant.  We will not include this named expression
                # and instead will expose the components so that linear
                # variables are not accidentally re-characterized as
                # nonlinear.
                pass
            ans.nl = None

        if ans.nonlinear.__class__ is list:
            ans.compile_nonlinear_fragment()

        if not ans.linear:
            ans.linear = {}
        if ans.mult != 1:
            linear = ans.linear
            mult, ans.mult = ans.mult, 1
            ans.const *= mult
            if linear:
                for k in linear:
                    linear[k] *= mult
            if ans.nonlinear:
                if mult == -1:
                    prefix = self.template.negation
                else:
                    prefix = self.template.multiplier % mult
                ans.nonlinear = prefix + ans.nonlinear[0], ans.nonlinear[1]
        #
        self.active_expression_source = None
        return ans


def evaluate_ampl_nl_expression(nl, external_functions):
    expr = nl.splitlines()
    stack = []
    while expr:
        line = expr.pop()
        tokens = line.split()
        # remove tokens after the first comment
        for i, t in enumerate(tokens):
            if t.startswith('#'):
                tokens = tokens[:i]
                break
        if len(tokens) != 1:
            # skip blank lines
            if not tokens:
                continue
            if tokens[0][0] == 'f':
                # external function
                fid, nargs = tokens
                fid = int(fid[1:])
                nargs = int(nargs)
                fcn_id, ef = external_functions[fid]
                assert fid == fcn_id
                stack.append(ef.evaluate(tuple(stack.pop() for i in range(nargs))))
                continue
            raise DeveloperError(
                f"Unsupported line format _evaluate_constant_nl() "
                f"(we expect each line to contain a single token): '{line}'"
            )
        term = tokens[0]
        # the "command" can be determined by the first character on the line
        cmd = term[0]
        # Note that we will unpack the line into the expected number of
        # explicit arguments as a form of error checking
        if cmd == 'n':
            # numeric constant
            stack.append(float(term[1:]))
        elif cmd == 'o':
            # operator
            nargs, fcn = nl_operators[int(term[1:])]
            if nargs is None:
                nargs = int(stack.pop())
            stack.append(fcn(*(stack.pop() for i in range(nargs))))
        elif cmd in '1234567890':
            # this is either a single int (e.g., the nargs in a nary
            # sum) or a string argument.  Preserve it as-is until later
            # when we know which we are expecting.
            stack.append(term)
        elif cmd == 'h':
            stack.append(term.split(':', 1)[1])
        else:
            raise DeveloperError(
                f"Unsupported NL operator in _evaluate_constant_nl(): '{line}'"
            )
    assert len(stack) == 1
    return stack[0]
