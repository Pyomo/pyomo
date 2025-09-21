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
from copy import deepcopy
from itertools import chain

from pyomo.common.collections import ComponentSet
from pyomo.common.errors import MouseTrap
from pyomo.common.numeric_types import native_types, native_numeric_types

import pyomo.core.expr as expr
import pyomo.repn.linear as linear

from pyomo.core.expr import ExpressionType
from pyomo.repn.linear import LinearRepn
from pyomo.repn.util import ExprType, initialize_exit_node_dispatcher, val2str

_CONSTANT = ExprType.CONSTANT
_VARIABLE = ExprType.VARIABLE
_LINEAR = ExprType.LINEAR

code_type = deepcopy.__class__


class LinearTemplateRepn(LinearRepn):
    __slots__ = ("linear_sum",)

    def __init__(self):
        super().__init__()
        self.linear_sum = []

    def __str__(self):
        linear = (
            "{"
            + ", ".join(f"{val2str(k)}: {val2str(v)}" for k, v in self.linear.items())
            + "}"
        )
        linear_sum = []
        for subrepn, subind, subsets in self.linear_sum:
            linear_sum.append(
                val2str(subrepn)
                + ", ["
                + ', '.join(
                    ("(" + ', '.join(val2str(j) for j in i) + ")") for i in subind
                )
                + "], ["
                + ', '.join(
                    ("(" + ', '.join(val2str(j) for j in i) + ")") for i in subsets
                )
                + "]"
            )
        linear_sum = '[' + ', '.join(linear_sum) + ']'
        return (
            f"{self.__class__.__name__}(mult={val2str(self.multiplier)}, "
            f"const={val2str(self.constant)}, "
            f"linear={linear}, "
            f"linear_sum={linear_sum}, "
            f"nonlinear={self.nonlinear})"
        )

    @staticmethod
    def constant_flag(val):
        if val.__class__ in native_numeric_types:
            return val
        return 2  # something not 0 or 1

    @staticmethod
    def multiplier_flag(val):
        if val.__class__ in native_numeric_types:
            return val
        return 2  # something not 0 or 1

    def walker_exitNode(self):
        if self.nonlinear is not None:
            return _GENERAL, self
        elif self.linear or self.linear_sum:
            return _LINEAR, self
        else:
            return _CONSTANT, self.multiplier * self.constant

    def duplicate(self):
        ans = super().duplicate()
        ans.linear_sum = [(r[0].duplicate(),) + r[1:] for r in self.linear_sum]
        return ans

    def append(self, other):
        """Append a child result from StreamBasedExpressionVisitor.acceptChildResult()

        Notes
        -----
        This method assumes that the operator was "+". It is implemented
        so that we can directly use a LinearTemplateRepn() as a data object in
        the expression walker (thereby avoiding the function call for a
        custom callback)

        """
        super().append(other)
        _type, other = other
        if other.__class__ is self.__class__ and other.linear_sum:
            if self.multiplier_flag(other.multiplier) != 1:
                mult = other.multiplier
                for term in other.linear_sum:
                    term[0].multiplier *= mult
            self.linear_sum.extend(other.linear_sum)

    def _build_evaluator(
        self, smap, expr_cache, multiplier, repetitions, remove_fixed_vars
    ):
        if self.nonlinear is not None:
            raise InvalidConstraintError(
                "LinearTemplateRepn cannot build an evaluator for template "
                "constraints containing general nonlinear terms"
            )
        ans = []
        multiplier *= self.multiplier
        constant = self.constant
        if constant.__class__ not in native_types or constant:
            constant *= multiplier
            if not repetitions or constant.__class__ not in native_types:
                ans.append('const += ' + constant.to_string(smap=smap))
                constant = 0
            else:
                constant *= repetitions
        if multiplier.__class__ not in native_types or multiplier:
            for k, coef in list(self.linear.items()):
                coef *= multiplier
                if coef.__class__ not in native_types:
                    coef = coef.to_string(smap=smap)
                else:
                    # Note that coef should never be trivially 0 (the
                    # visitor should remove most of those), so there is
                    # no reason to check and skip this term...
                    coef = repr(coef)

                indent = ''
                if k in expr_cache:
                    k = expr_cache[k]
                    if k.__class__ not in native_types:
                        # Directly substitute the expression into the
                        # 'linear[vid] = coef' below
                        k = k.to_string(smap=smap)
                        # If we are removing fixed vars, add that logic here:
                        if remove_fixed_vars:
                            ans.append(f"v = {k}")
                            ans.append('if v.__class__ is tuple:')
                            ans.append('    const += v[0] * {coef}')
                            ans.append('    v = None')
                            ans.append('else:')
                            k = 'v'
                            indent = '    '
                ans.append(indent + f'linear_indices.append({k})')
                ans.append(indent + f'linear_data.append({coef})')

        for subrepn, subindices, subsets in self.linear_sum:
            ans.extend(
                '    ' * i
                + f"for {','.join(smap.getSymbol(i) for i in _idx)} in "
                + (
                    _set.to_string(smap=smap)
                    if _set.is_expression_type()
                    else smap.getSymbol(_set)
                )
                + ":"
                for i, (_idx, _set) in enumerate(zip(subindices, subsets))
            )
            try:
                subrep = 1
                for _set in subsets:
                    subrep *= len(_set)
            except:
                subrep = 0
            subans, subconst = subrepn._build_evaluator(
                smap, expr_cache, multiplier, repetitions * subrep, remove_fixed_vars
            )
            indent = '    ' * (len(subsets))
            ans.extend(indent + line for line in subans)
            constant += subconst
        return ans, constant

    def compile(self, env, smap, expr_cache, args, remove_fixed_vars=False):
        ans, constant = self._build_evaluator(smap, expr_cache, 1, 1, remove_fixed_vars)
        if not ans:
            return constant
        indent = '\n    '
        if not constant and ans and ans[0].startswith('const +='):
            # Convert initial "const +=" to "const ="
            ans[0] = ''.join(ans[0].split('+', 1))
        else:
            ans.insert(0, 'const = ' + repr(constant))
        fcn_body = indent.join(ans[1:])
        if 'const' not in fcn_body:
            # No constants in the expression.  Move the initial const
            # term to the return value and avoid declaring the local
            # variable
            ans = ['return ' + ans[0].split('=', 1)[1]]
            if fcn_body:
                ans.insert(0, fcn_body)
        else:
            ans = [ans[0], fcn_body, 'return const']
        ans.insert(
            0, f"def build_expr(linear_indices, linear_data, {', '.join(args)}):"
        )
        ans = indent.join(ans)
        # build the function in the env namespace, then remove and
        # return the compiled function.  The function's globals will
        # still be bound to env
        exec(ans, env)
        return env.pop('build_expr')


class LinearTemplateBeforeChildDispatcher(linear.LinearBeforeChildDispatcher):
    @staticmethod
    def _before_var(visitor, child):
        # Note: the LinearBeforeChildDispatcher returns Var ids as
        # id(var), whereas we are returning the actual variable order
        # here.
        #
        # TODO(?): add a "_id" field to all VarData so that walkers can
        # assign "useful local IDs" to variables that they encounter?
        desc, ans = linear.LinearBeforeChildDispatcher._before_var(visitor, child)
        if ans[0] is _LINEAR:
            vo = visitor.var_recorder.var_order
            ans[1].linear = {vo[_id]: coef for _id, coef in ans[1].linear.items()}
        return desc, ans

    @staticmethod
    def _before_monomial(visitor, child):
        # Note: the LinearBeforeChildDispatcher returns Var ids as
        # id(var), whereas we are returning the actual variable order
        # here.
        #
        # TODO(?): add a "_id" field to all VarData so that walkers can
        # assign "useful local IDs" to variables that they encounter?
        desc, ans = linear.LinearBeforeChildDispatcher._before_monomial(visitor, child)
        if not desc and ans[0] is _LINEAR:
            vo = visitor.var_recorder.var_order
            ans[1].linear = {vo[_id]: coef for _id, coef in ans[1].linear.items()}
        return desc, ans

    @staticmethod
    def _before_linear(visitor, child):
        # Note: the LinearBeforeChildDispatcher returns Var ids as
        # id(var), whereas we are returning the actual variable order
        # here.
        #
        # TODO(?): add a "_id" field to all VarData so that walkers can
        # assign "useful local IDs" to variables that they encounter?
        desc, ans = linear.LinearBeforeChildDispatcher._before_linear(visitor, child)
        if not desc and ans[0] is _LINEAR:
            vo = visitor.var_recorder.var_order
            ans[1].linear = {vo[_id]: coef for _id, coef in ans[1].linear.items()}
        return desc, ans

    @staticmethod
    def _before_indexed_var(visitor, child):
        if child not in visitor.indexed_vars:
            visitor.var_recorder.add(child)
            visitor.indexed_vars.add(child)
        return False, (_VARIABLE, child)

    @staticmethod
    def _before_indexed_param(visitor, child):
        if child not in visitor.indexed_params:
            visitor.indexed_params.add(child)
            name = visitor.symbolmap.getSymbol(child)
            visitor.env[name] = child.extract_values()
        return False, (_CONSTANT, child)

    @staticmethod
    def _before_indexed_component(visitor, child):
        visitor.env[visitor.symbolmap.getSymbol(child)] = child
        return False, (_CONSTANT, child)

    @staticmethod
    def _before_index_template(visitor, child):
        symb = visitor.symbolmap.getSymbol(child)
        visitor.env[symb] = 0
        visitor.expr_cache[id(child)] = child
        return False, (_CONSTANT, child)

    @staticmethod
    def _before_component(visitor, child):
        visitor.env[visitor.symbolmap.getSymbol(child)] = child
        return False, (_CONSTANT, child)

    @staticmethod
    def _before_named_expression(visitor, child):
        raise MouseTrap("We do not yet support Expression components")


def _handle_getitem(visitor, node, comp, *args):
    expr = comp[1][tuple(arg[1] for arg in args)]
    if comp[0] is _CONSTANT:
        return (_CONSTANT, expr)
    elif comp[0] is _VARIABLE:
        # Because we are passing up an id() and not the expression
        # itself, we need to cache the expression that we just created
        # to preserve a reference to it and prevent deallocation / GC
        visitor.expr_cache[id(expr)] = expr
        ans = visitor.Result()
        ans.linear[id(expr)] = 1
        return (_LINEAR, ans)


def _handle_templatesum(visitor, node, comp, *args):
    ans = visitor.Result()
    if comp[0] is _LINEAR:
        ans.linear_sum.append((comp[1], node.template_iters(), [a[1] for a in args]))
        return _LINEAR, ans
    else:
        raise DeveloperError()


def define_exit_node_handlers(_exit_node_handlers=None):
    if _exit_node_handlers is None:
        _exit_node_handlers = {}
    linear.define_exit_node_handlers(_exit_node_handlers)

    _exit_node_handlers[expr.GetItemExpression] = {None: _handle_getitem}
    _exit_node_handlers[expr.TemplateSumExpression] = {None: _handle_templatesum}

    return _exit_node_handlers


class LinearTemplateRepnVisitor(linear.LinearRepnVisitor):
    Result = LinearTemplateRepn
    before_child_dispatcher = LinearTemplateBeforeChildDispatcher()
    exit_node_dispatcher = linear.ExitNodeDispatcher(
        initialize_exit_node_dispatcher(define_exit_node_handlers())
    )

    def __init__(self, subexpression_cache, var_recorder, remove_fixed_vars=False):
        super().__init__(subexpression_cache, var_recorder=var_recorder)
        self.indexed_vars = set()
        self.indexed_params = set()
        self.expr_cache = {}
        self.env = var_recorder.env
        self.symbolmap = var_recorder.symbolmap
        self.expanded_templates = {}
        self.remove_fixed_vars = remove_fixed_vars

    def enterNode(self, node):
        # SumExpression are potentially large nary operators.  Directly
        # populate the result
        if node.__class__ is expr.TemplateSumExpression:
            return node.template_args(), []
        if node.__class__ in linear.sum_like_expression_types:
            return node.args, self.Result()
        else:
            return node.args, []

    def expand_expression(self, obj, template_info):
        env = self.env
        try:
            body, lb, ub = self.expanded_templates[id(template_info)]
        except KeyError:
            smap = self.symbolmap
            expr, indices = template_info
            args = [smap.getSymbol(i) for i in indices]
            if expr.is_expression_type(ExpressionType.RELATIONAL):
                lb, body, ub = obj.to_bounded_expression()
                if body is not None:
                    body = self.walk_expression(body).compile(
                        env, smap, self.expr_cache, args, False
                    )
                if lb is not None:
                    lb = self.walk_expression(lb).compile(
                        env, smap, self.expr_cache, args, True
                    )
                if ub is not None:
                    ub = self.walk_expression(ub).compile(
                        env, smap, self.expr_cache, args, True
                    )
            elif expr is not None:
                lb = ub = None
                body = self.walk_expression(expr).compile(
                    env, smap, self.expr_cache, args, False
                )
            else:
                body = lb = ub = None
            self.expanded_templates[id(template_info)] = body, lb, ub

        linear_indices = []
        linear_data = []
        index = obj.index()
        if index.__class__ is not tuple:
            if index is None and not obj.parent_component().is_indexed():
                index = ()
            else:
                index = (index,)
        if lb.__class__ is code_type:
            lb = lb(linear_indices, linear_data, *index)
            if linear_indices:
                raise RuntimeError(f"Constraint {obj} has non-fixed lower bound")
        if ub.__class__ is code_type:
            ub = ub(linear_indices, linear_data, *index)
            if linear_indices:
                raise RuntimeError(f"Constraint {obj} has non-fixed upper bound")
        return (
            body(linear_indices, linear_data, *index),
            linear_indices,
            linear_data,
            lb,
            ub,
        )
