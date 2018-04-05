#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from six import string_types

from pyomo.core.kernel.numvalue import native_types

import pyomo.core.base.expr as EXPR
import pyomo.core.base.expr_coopr3 as coopr3

from pyomo.core.base.component import _ComponentBase, ComponentUID
from pyomo.opt import TerminationCondition, SolverStatus

_acceptable_termination_conditions = set([
    TerminationCondition.optimal,
    TerminationCondition.globallyOptimal,
    TerminationCondition.locallyOptimal,
])
_infeasible_termination_conditions = set([
    TerminationCondition.infeasible,
    TerminationCondition.invalidProblem,
])


class NORMAL(object): pass
class INFEASIBLE(object): pass
class NONOPTIMAL(object): pass

def verify_successful_solve(results):
    status = results.solver.status
    term = results.solver.termination_condition

    if status == SolverStatus.ok and term in _acceptable_termination_conditions:
        return NORMAL
    elif term in _infeasible_termination_conditions:
        return INFEASIBLE
    else:
        return NONOPTIMAL


def clone_without_expression_components(expr, substitute):
    ans = [EXPR.clone_expression(expr, substitute=substitute)]
    _stack = [ (ans, 0, 1) ]
    while _stack:
        _argList, _idx, _len = _stack.pop()
        while _idx < _len:
            _sub = _argList[_idx]
            _idx += 1
            if type(_sub) in native_types:
                pass
            elif _sub.is_expression():
                _stack.append(( _argList, _idx, _len ))
                if not isinstance(_sub, EXPR._ExpressionBase):
                    _argList[_idx-1] = EXPR.clone_expression(
                        _sub._args[0], substitute=substitute )
                elif type(_sub) is coopr3._ProductExpression:
                    if _sub._denominator:
                        _stack.append(
                            (_sub._denominator, 0, len(_sub._denominator)) )
                    _argList = _sub._numerator
                else:
                    _argList = _sub._args
                    # HACK: As we may have to replace arguments, if the
                    # _args is a tuple, then we will convert it to a
                    # list.
                    if type(_argList) is tuple:
                        _argList = _sub._args = list(_argList)
                _idx = 0
                _len = len(_argList)
    return ans[0]


def target_list(x):
    if isinstance(x, ComponentUID):
        return [ x ]
    elif isinstance(x, (_ComponentBase, string_types)):
        return [ ComponentUID(x) ]
    elif hasattr(x, '__iter__'):
        ans = []
        for i in x:
            if isinstance(i, ComponentUID):
                ans.append(i)
            elif isinstance(i, (_ComponentBase, string_types)):
                ans.append(ComponentUID(i))
            else:
                raise ValueError(
                    "Expected ComponentUID, Component, Component name, "
                    "or list of these.\n\tReceived %s" % (type(i),))

        return ans
    else:
        raise ValueError(
            "Expected ComponentUID, Component, Component name, "
            "or list of these.\n\tReceived %s" % (type(x),))
