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
import logging
import os
import sys
from collections import deque
from operator import itemgetter, attrgetter, setitem

from pyomo.common.backports import nullcontext
from pyomo.common.config import (
    ConfigBlock,
    ConfigValue,
    InEnum,
    document_kwargs_from_configdict,
)
from pyomo.common.deprecation import deprecation_warning
from pyomo.common.errors import DeveloperError
from pyomo.common.gc_manager import PauseGC
from pyomo.common.timing import TicTocTimer

from pyomo.core.expr.current import (
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
    native_types,
    native_numeric_types,
    value,
)
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor
from pyomo.core.base import (
    Block,
    Objective,
    Constraint,
    Var,
    Param,
    Expression,
    ExternalFunction,
    Suffix,
    SOSConstraint,
    SymbolMap,
    NameLabeler,
    SortComponents,
    minimize,
)
from pyomo.core.base.block import SortComponents
from pyomo.core.base.component import ActiveComponent
from pyomo.core.base.expression import ScalarExpression, _GeneralExpressionData
from pyomo.core.base.objective import ScalarObjective, _GeneralObjectiveData
import pyomo.core.kernel as kernel
from pyomo.core.pyomoobject import PyomoObject
from pyomo.opt import WriterFactory

from pyomo.repn.plugins.ampl.ampl_ import set_pyomo_amplfunc_env

### FIXME: Remove the following as soon as non-active components no
### longer report active==True
from pyomo.core.base import Set, RangeSet
from pyomo.network import Port

###

logger = logging.getLogger(__name__)

# Feasibility tolerance for trivial (fixed) constraints
TOL = 1e-8
inf = float('inf')
minus_inf = -inf
nan = float('nan')

HALT_ON_EVALUATION_ERROR = False


class _CONSTANT(object):
    pass


class _MONOMIAL(object):
    pass


class _GENERAL(object):
    pass


def _apply_node_operation(node, args):
    try:
        tmp = (_CONSTANT, node._apply_operation(args))
        if tmp[1].__class__ is complex:
            raise ValueError('Pyomo does not support complex numbers')
        return tmp
    except:
        logger.warning(
            "Exception encountered evaluating expression "
            "'%s(%s)'\n\tmessage: %s\n\texpression: %s"
            % (node.name, ", ".join(map(str, args)), str(sys.exc_info()[1]), node)
        )
        if HALT_ON_EVALUATION_ERROR:
            raise
        return (_CONSTANT, nan)


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
    __slots__ = ('nl', 'mult', 'const', 'linear', 'nonlinear', 'named_exprs')

    ActiveVisitor = None

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

    def duplicate(self):
        ans = self.__class__.__new__(self.__class__)
        ans.nl = self.nl
        ans.mult = self.mult
        ans.const = self.const
        ans.linear = None if self.linear is None else dict(self.linear)
        ans.nonlinear = self.nonlinear
        ans.named_exprs = self.named_exprs
        return ans

    def compile_repn(self, visitor, prefix='', args=None, named_exprs=None):
        template = visitor.template
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

    def compile_nonlinear_fragment(self, visitor):
        if not self.nonlinear:
            self.nonlinear = None
            return
        args = []
        nterms = len(self.nonlinear)
        nl_sum = ''.join(map(itemgetter(0), self.nonlinear))
        deque(map(args.extend, map(itemgetter(1), self.nonlinear)), maxlen=0)

        if nterms > 2:
            self.nonlinear = (visitor.template.nary_sum % nterms) + nl_sum, args
        elif nterms == 2:
            self.nonlinear = visitor.template.binary_sum + nl_sum, args
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
                    other = other.compile_repn(
                        self.ActiveVisitor, '', None, self.named_exprs
                    )
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
                        other.compile_nonlinear_fragment(self.ActiveVisitor)
                    if mult == -1:
                        prefix = self.ActiveVisitor.template.negation
                    else:
                        prefix = self.ActiveVisitor.template.multiplier % mult
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
            if other[1] is None:
                self.const = None
            else:
                self.const += other[1]


def _create_strict_inequality_map(vars_):
    vars_['strict_inequality_map'] = {
        True: vars_['less_than'],
        False: vars_['less_equal'],
        (True, True): (vars_['less_than'], vars_['less_than']),
        (True, False): (vars_['less_than'], vars_['less_equal']),
        (False, True): (vars_['less_equal'], vars_['less_than']),
        (False, False): (vars_['less_equal'], vars_['less_equal']),
    }


class text_nl_debug_template(object):
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
    var = 'v%s\n'
    const = 'n%r\n'
    string = 'h%d:%s\n'
    monomial = product + const + var.replace('%', '%%')
    multiplier = product + const

    _create_strict_inequality_map(vars())


def _strip_template_comments(vars_, base_):
    vars_['unary'] = {k: v[: v.find('\t#')] + '\n' for k, v in base_.unary.items()}
    for k, v in base_.__dict__.items():
        if type(v) is str and '\t#' in v:
            v_lines = v.split('\n')
            for i, l in enumerate(v_lines):
                comment_start = l.find('\t#')
                if comment_start >= 0:
                    v_lines[i] = l[:comment_start]
            vars_[k] = '\n'.join(v_lines)


# The "standard" text mode template is the debugging template with the
# comments removed
class text_nl_template(text_nl_debug_template):
    _strip_template_comments(vars(), text_nl_debug_template)
    _create_strict_inequality_map(vars())


def node_result_to_amplrepn(data):
    if data[0] is _GENERAL:
        return data[1]
    elif data[0] is _MONOMIAL:
        _, v, c = data
        if c != 0:
            return AMPLRepn(0, {v: c}, None)
        else:
            return AMPLRepn(0, None, None)
    elif data[0] is _CONSTANT:
        return AMPLRepn(data[1], None, None)
    else:
        raise DeveloperError("unknown result type")


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
        if mult == 0: # Skip this branch if mult is None
        #if not mult:
            # simplify multiplication by 0 (if arg2 is zero, the
            # simplification happens when we evaluate the constant
            # below).  Note that this is not IEEE-754 compliant, and
            # will map 0*inf and 0*nan to 0 (and not to nan).  We are
            # including this for backwards compatibility with the NLv1
            # writer, but arguably we should deprecate/remove this
            # "feature" in the future.
            if arg2[0] is _CONSTANT:
                if arg2[1] is None:
                    _prod = None
                    return (_CONSTANT, _prod)
                else:
                    _prod = mult * arg2[1]
                    if _prod:
                        deprecation_warning(
                            f"Encountered {mult}*{arg2[1]} in expression tree.  "
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
            # Catch None and return a monomial term
            if mult is None:
                return (_MONOMIAL, arg2[1], None)
            if mult != mult:
                # This catches mult (i.e., arg1) == nan
                return arg1
            return (_MONOMIAL, arg2[1], mult * arg2[2])
        elif arg2[0] is _GENERAL:
            # Catch None and return a general expression term
            if mult is None:
                arg2[1].mult = None
                return arg2
            if mult != mult:
                # This catches mult (i.e., arg1) == nan
                return arg1
            arg2[1].mult *= mult
            return arg2
        elif arg2[0] is _CONSTANT:
            # Catch None in either argument
            if arg2[1] is None or mult is None:
                return (_CONSTANT, None)
            if not arg2[1]:
                # Simplify multiplication by 0; see note above about
                # IEEE-754 incompatibility.
                _prod = mult * arg2[1]
                if _prod:
                    deprecation_warning(
                        f"Encountered {mult}*{arg2[1]} in expression tree.  "
                        "Mapping the NaN result to 0 for compatibility "
                        "with the nl_v1 writer.  In the future, this NaN "
                        "will be preserved/emitted to comply with IEEE-754.",
                        version='6.4.3',
                    )
                    _prod = 0
                return (_CONSTANT, _prod)
            return (_CONSTANT, mult * arg2[1])
    nonlin = node_result_to_amplrepn(arg1).compile_repn(
        visitor, visitor.template.product
    )
    nonlin = node_result_to_amplrepn(arg2).compile_repn(visitor, *nonlin)
    return (_GENERAL, AMPLRepn(0, None, nonlin))


def handle_division_node(visitor, node, arg1, arg2):
    if arg2[0] is _CONSTANT:
        div = arg2[1]
        if div == 1:
            return arg1
        if arg1[0] is _MONOMIAL:
            tmp = _apply_node_operation(node, (arg1[2], div))
            if tmp[1] != tmp[1]:
                # This catches if the coefficient division results in nan
                return tmp
            return (_MONOMIAL, arg1[1], tmp[1])
        elif arg1[0] is _GENERAL:
            tmp = _apply_node_operation(node, (arg1[1].mult, div))[1]
            if tmp != tmp:
                # This catches if the multiplier division results in nan
                return _CONSTANT, tmp
            arg1[1].mult = tmp
            return arg1
        elif arg1[0] is _CONSTANT:
            return _apply_node_operation(node, (arg1[1], div))
    elif arg1[0] is _CONSTANT and not arg1[1]:
        return _CONSTANT, 0
    nonlin = node_result_to_amplrepn(arg1).compile_repn(
        visitor, visitor.template.division
    )
    nonlin = node_result_to_amplrepn(arg2).compile_repn(visitor, *nonlin)
    return (_GENERAL, AMPLRepn(0, None, nonlin))


def handle_pow_node(visitor, node, arg1, arg2):
    if arg2[0] is _CONSTANT:
        if arg1[0] is _CONSTANT:
            return _apply_node_operation(node, (arg1[1], arg2[1]))
        elif not arg2[1]:
            return _CONSTANT, 1
        elif arg2[1] == 1:
            return arg1
    nonlin = node_result_to_amplrepn(arg1).compile_repn(visitor, visitor.template.pow)
    nonlin = node_result_to_amplrepn(arg2).compile_repn(visitor, *nonlin)
    return (_GENERAL, AMPLRepn(0, None, nonlin))


def handle_abs_node(visitor, node, arg1):
    if arg1[0] is _CONSTANT:
        return (_CONSTANT, abs(arg1[1]))
    nonlin = node_result_to_amplrepn(arg1).compile_repn(visitor, visitor.template.abs)
    return (_GENERAL, AMPLRepn(0, None, nonlin))


def handle_unary_node(visitor, node, arg1):
    if arg1[0] is _CONSTANT:
        return _apply_node_operation(node, (arg1[1],))
    nonlin = node_result_to_amplrepn(arg1).compile_repn(
        visitor, visitor.template.unary[node.name]
    )
    return (_GENERAL, AMPLRepn(0, None, nonlin))


def handle_exprif_node(visitor, node, arg1, arg2, arg3):
    if arg1[0] is _CONSTANT:
        if arg1[1]:
            return arg2
        else:
            return arg3
    nonlin = node_result_to_amplrepn(arg1).compile_repn(
        visitor, visitor.template.exprif
    )
    nonlin = node_result_to_amplrepn(arg2).compile_repn(visitor, *nonlin)
    nonlin = node_result_to_amplrepn(arg3).compile_repn(visitor, *nonlin)
    return (_GENERAL, AMPLRepn(0, None, nonlin))


def handle_equality_node(visitor, node, arg1, arg2):
    if arg1[0] is _CONSTANT and arg2[0] is _CONSTANT:
        return (_CONSTANT, arg1[1] == arg2[1])
    nonlin = node_result_to_amplrepn(arg1).compile_repn(
        visitor, visitor.template.equality
    )
    nonlin = node_result_to_amplrepn(arg2).compile_repn(visitor, *nonlin)
    return (_GENERAL, AMPLRepn(0, None, nonlin))


def handle_inequality_node(visitor, node, arg1, arg2):
    if arg1[0] is _CONSTANT and arg2[0] is _CONSTANT:
        return (_CONSTANT, node._apply_operation((arg1[1], arg2[1])))
    nonlin = node_result_to_amplrepn(arg1).compile_repn(
        visitor, visitor.template.strict_inequality_map[node.strict]
    )
    nonlin = node_result_to_amplrepn(arg2).compile_repn(visitor, *nonlin)
    return (_GENERAL, AMPLRepn(0, None, nonlin))


def handle_ranged_inequality_node(visitor, node, arg1, arg2, arg3):
    if arg1[0] is _CONSTANT and arg2[0] is _CONSTANT and arg3[0] is _CONSTANT:
        return (_CONSTANT, node._apply_operation((arg1[1], arg2[1], arg3[1])))
    op = visitor.template.strict_inequality_map[node.strict]
    nl, args, named = node_result_to_amplrepn(arg1).compile_repn(
        visitor, visitor.template.and_expr + op[0]
    )
    nl2, args2, named = node_result_to_amplrepn(arg2).compile_repn(
        visitor, '', None, named
    )
    nl += nl2 + op[1] + nl2
    args.extend(args2)
    args.extend(args2)
    nonlin = node_result_to_amplrepn(arg3).compile_repn(visitor, nl, args, named)
    return (_GENERAL, AMPLRepn(0, None, nonlin))


def handle_named_expression_node(visitor, node, arg1):
    _id = id(node)
    # Note that while named subexpressions ('defined variables' in the
    # ASL NL file vernacular) look like variables, they are not allowed
    # to appear in the 'linear' portion of a constraint / objective
    # definition.  We will return this as a "var" template, but
    # wrapped in the nonlinear portion of the expression tree.
    repn = node_result_to_amplrepn(arg1)

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

    if not visitor.use_named_exprs:
        return _GENERAL, repn.duplicate()

    mult, repn.mult = repn.mult, 1
    if repn.named_exprs is None:
        repn.named_exprs = set()

    # When converting this shared subexpression to a (nonlinear)
    # node, we want to just reference this subexpression:
    repn.nl = (visitor.template.var, (_id,))

    if repn.nonlinear:
        # As we will eventually need the compiled form of any nonlinear
        # expression, we will go ahead and compile it here.  We do not
        # do the same for the linear component as we will only need the
        # linear component compiled to a dict if we are emitting the
        # original (linear + nonlinear) V line (which will not happen if
        # the V line is part of a larger linear operator).
        if repn.nonlinear.__class__ is list:
            repn.compile_nonlinear_fragment(visitor)

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
            sub_repn = AMPLRepn(0, None, None)
            sub_repn.nonlinear = repn.nonlinear
            sub_repn.nl = (visitor.template.var, (sub_id,))
            sub_repn.named_exprs = set(repn.named_exprs)

            repn.named_exprs.add(sub_id)
            repn.nonlinear = sub_repn.nl

            # See above for the meaning of this source information
            nl_info = list(expression_source)
            visitor.subexpression_cache[sub_id] = (sub_node, sub_repn, nl_info)
            # It is important that the NL subexpression comes before the
            # main named expression:
            visitor.subexpression_order.append(sub_id)
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
            return (_MONOMIAL, next(iter(repn.linear)), 1)
        else:
            return (_CONSTANT, repn.const)

    # Defer recording this _id until after we know that this repn will
    # not be directly substituted (and to ensure that the NL fragment is
    # added to the order first).
    visitor.subexpression_order.append(_id)

    return (_GENERAL, repn.duplicate())


def handle_external_function_node(visitor, node, *args):
    func = node._fcn._function
    # There is a special case for external functions: these are the only
    # expressions that can accept string arguments. As we currently pass
    # these as 'precompiled' general NL fragments, the normal trap for
    # constant subexpressions will miss constant external function calls
    # that contain strings.  We will catch that case here.
    if all(
        arg[0] is _CONSTANT or (arg[0] is _GENERAL and arg[1].nl and not arg[1].nl[1])
        for arg in args
    ):
        arg_list = [arg[1] if arg[0] is _CONSTANT else arg[1].const for arg in args]
        return _apply_node_operation(node, arg_list)
    if func in visitor.external_functions:
        if node._fcn._library != visitor.external_functions[func][1]._library:
            raise RuntimeError(
                "The same external function name (%s) is associated "
                "with two different libraries (%s through %s, and %s "
                "through %s).  The ASL solver will fail to link "
                "correctly."
                % (
                    func,
                    visitor.external_byFcn[func]._library,
                    visitor.external_byFcn[func]._library.name,
                    node._fcn._library,
                    node._fcn.name,
                )
            )
    else:
        visitor.external_functions[func] = (len(visitor.external_functions), node._fcn)
    comment = f'\t#{node.local_name}' if visitor.symbolic_solver_labels else ''
    nonlin = node_result_to_amplrepn(args[0]).compile_repn(
        visitor,
        visitor.template.external_fcn
        % (visitor.external_functions[func][0], len(args), comment),
    )
    for arg in args[1:]:
        nonlin = node_result_to_amplrepn(arg).compile_repn(visitor, *nonlin)
    return (_GENERAL, AMPLRepn(0, None, nonlin))


_operator_handles = {
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
    _GeneralExpressionData: handle_named_expression_node,
    ScalarExpression: handle_named_expression_node,
    kernel.expression.expression: handle_named_expression_node,
    kernel.expression.noclone: handle_named_expression_node,
    # Note: objectives are special named expressions
    _GeneralObjectiveData: handle_named_expression_node,
    ScalarObjective: handle_named_expression_node,
    kernel.objective.objective: handle_named_expression_node,
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


def _before_native(visitor, child):
    return False, (_CONSTANT, child)


def _before_string(visitor, child):
    visitor.encountered_string_arguments = True
    ans = AMPLRepn(child, None, None)
    ans.nl = (visitor.template.string % (len(child), child), ())
    return False, (_GENERAL, ans)


def _before_var(visitor, child):
    _id = id(child)
    if _id not in visitor.var_map:
        if child.fixed:
            return False, (_CONSTANT, child())
        visitor.var_map[_id] = child
    return False, (_MONOMIAL, _id, 1)


def _before_npv(visitor, child):
    # TBD: It might be more efficient to cache the value of NPV
    # expressions to avoid duplicate evaluations.  However, current
    # examples do not benefit from this cache.
    #
    # _id = id(child)
    # if _id in visitor.value_cache:
    #     child = visitor.value_cache[_id]
    # else:
    #     child = visitor.value_cache[_id] = child()
    # return False, (_CONSTANT, child)
    try:
        tmp = False, (_CONSTANT, child())
        if tmp[1][1].__class__ is complex:
            return True, None
        return tmp
    except:
        # If there was an exception evaluating the subexpression, then
        # we need to descend into it (in case there is something like 0 *
        # nan that we need to map to 0)
        return True, None


def _before_monomial(visitor, child):
    #
    # The following are performance optimizations for common
    # situations (Monomial terms and Linear expressions)
    #
    arg1, arg2 = child._args_
    if arg1.__class__ not in native_types:
        # TBD: It might be more efficient to cache the value of NPV
        # expressions to avoid duplicate evaluations.  However, current
        # examples do not benefit from this cache.
        #
        # _id = id(arg1)
        # if _id in visitor.value_cache:
        #     arg1 = visitor.value_cache[_id]
        # else:
        #     arg1 = visitor.value_cache[_id] = arg1()
        try:
            arg1 = arg1()
        except:
            # If there was an exception evaluating the subexpression,
            # then we need to descend into it (in case there is something
            # like 0 * nan that we need to map to 0)
            return True, None

    if arg2.fixed:
        arg2 = arg2.value
        if arg2 is None:
            _prod = None
        else:
            _prod = arg1 * arg2
        if not (arg1 and arg2) and _prod:
            # What is this checking for?
            deprecation_warning(
                f"Encountered {arg1}*{arg2} in expression tree.  "
                "Mapping the NaN result to 0 for compatibility "
                "with the nl_v1 writer.  In the future, this NaN "
                "will be preserved/emitted to comply with IEEE-754.",
                version='6.4.3',
            )
            _prod = 0
        return (_CONSTANT, _prod)

    # Trap multiplication by 0.
    if not arg1:
        return False, (_CONSTANT, 0)
    _id = id(arg2)
    if _id not in visitor.var_map:
        visitor.var_map[_id] = arg2
    return False, (_MONOMIAL, _id, arg1)


def _before_linear(visitor, child):
    # Because we are going to modify the LinearExpression in this
    # walker, we need to make a copy of the arg list from the original
    # expression tree.
    var_map = visitor.var_map
    const = 0
    linear = {}
    for arg in child.args:
        if arg.__class__ is MonomialTermExpression:
            c, v = arg.args
            if c.__class__ not in native_types:
                c = c()
            if v.fixed:
                if v.value is None:
                    const = None
                else:
                    const += c * v.value
            elif c:
                # What is this check? c is nonzero?
                _id = id(v)
                if _id not in var_map:
                    var_map[_id] = v
                if _id in linear:
                    linear[_id] += c
                else:
                    linear[_id] = c
        elif arg.__class__ in native_types:
            # const is an "accumulated" constant and therefore could be None.
            # arg is an unprocessed node of the expression system and therefore
            # should not be None
            if const is not None:
                const += arg
        else:
            if const is not None:
                const += arg()
    if linear:
        return False, (_GENERAL, AMPLRepn(const, linear, None))
    else:
        return False, (_CONSTANT, const)


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


def _before_general_expression(visitor, child):
    return True, None


# Register an initial set of known expression types with the "before
# child" expression handler lookup table.
_before_child_handlers = {_type: _before_native for _type in native_numeric_types}
for _type in native_types:
    if issubclass(_type, str):
        _before_child_handlers[_type] = _before_string
# general operators
for _type in _operator_handles:
    _before_child_handlers[_type] = _before_general_expression
# named subexpressions
for _type in (
    _GeneralExpressionData,
    ScalarExpression,
    kernel.expression.expression,
    kernel.expression.noclone,
    _GeneralObjectiveData,
    ScalarObjective,
    kernel.objective.objective,
):
    _before_child_handlers[_type] = _before_named_expression
# Special linear / summation expressions
_before_child_handlers[MonomialTermExpression] = _before_monomial
_before_child_handlers[LinearExpression] = _before_linear
_before_child_handlers[SumExpression] = _before_general_expression


class AMPLRepnVisitor(StreamBasedExpressionVisitor):
    def __init__(
        self,
        template,
        subexpression_cache,
        subexpression_order,
        external_functions,
        var_map,
        used_named_expressions,
        symbolic_solver_labels,
        use_named_exprs,
    ):
        super().__init__()
        self.template = template
        self.subexpression_cache = subexpression_cache
        self.subexpression_order = subexpression_order
        self.external_functions = external_functions
        self.active_expression_source = None
        self.var_map = var_map
        self.used_named_expressions = used_named_expressions
        self.symbolic_solver_labels = symbolic_solver_labels
        self.use_named_exprs = use_named_exprs
        self.encountered_string_arguments = False
        # self.value_cache = {}

    def initializeWalker(self, expr):
        expr, src, src_idx = expr
        self.active_expression_source = (src_idx, id(src))
        walk, result = self.beforeChild(None, expr, 0)
        if not walk:
            return False, self.finalizeResult(result)
        return True, expr

    def beforeChild(self, node, child, child_idx):
        try:
            return _before_child_handlers[child.__class__](self, child)
        except KeyError:
            self._register_new_before_child_processor(child)
        return _before_child_handlers[child.__class__](self, child)

    def enterNode(self, node):
        # SumExpression are potentially large nary operators.  Directly
        # populate the result
        if node.__class__ is SumExpression:
            data = AMPLRepn(0, {}, None)
            data.nonlinear = []
            return node.args, data
        else:
            return node.args, []

    def exitNode(self, node, data):
        if data.__class__ is AMPLRepn:
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
        ans = node_result_to_amplrepn(result)

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
                # ans.nonlinear = orig.nonlinear
            ans.nl = None

        if ans.nonlinear.__class__ is list:
            ans.compile_nonlinear_fragment(self)

        if not ans.linear:
            ans.linear = {}
        linear = ans.linear
        if ans.mult != 1:
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

    def _register_new_before_child_processor(self, child):
        handlers = _before_child_handlers
        child_type = child.__class__
        if child_type in native_numeric_types:
            handlers[child_type] = _before_native
        elif issubclass(child_type, str):
            handlers[child_type] = _before_string
        elif child_type in native_types:
            handlers[child_type] = _before_native
        elif not child.is_expression_type():
            if child.is_potentially_variable():
                handlers[child_type] = _before_var
            else:
                handlers[child_type] = _before_npv
        elif not child.is_potentially_variable():
            handlers[child_type] = _before_npv
            # If we descend into the named expression (because of an
            # evaluation error), then on the way back out, we will use
            # the potentially variable handler to process the result.
            pv_base_type = child.potentially_variable_base_class()
            if pv_base_type not in handlers:
                try:
                    child.__class__ = pv_base_type
                    _register_new_before_child_processor(self, child)
                finally:
                    child.__class__ = child_type
            if pv_base_type in _operator_handles:
                _operator_handles[child_type] = _operator_handles[pv_base_type]
        elif id(child) in self.subexpression_cache or issubclass(
            child_type, _GeneralExpressionData
        ):
            handlers[child_type] = _before_named_expression
            _operator_handles[child_type] = handle_named_expression_node
        else:
            handlers[child_type] = _before_general_expression
