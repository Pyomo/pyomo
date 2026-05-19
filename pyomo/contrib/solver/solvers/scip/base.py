# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

from __future__ import annotations

import logging
import math
from typing import List, Optional, Sequence, Mapping

from pyomo.common.collections import ComponentMap
from pyomo.common.numeric_types import native_numeric_types
from pyomo.common.config import ConfigValue
from pyomo.common.dependencies import attempt_import
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
from pyomo.contrib.solver.common.config import BranchAndBoundConfig
from pyomo.contrib.solver.common.solution_loader import SolutionLoader
from pyomo.contrib.solver.common.util import NoSolutionError
from pyomo.core.base.expression import ExpressionData, ScalarExpression
from pyomo.core.base.param import ParamData, ScalarParam
from pyomo.core.base.units_container import _PyomoUnit
from pyomo.core.base.var import VarData, ScalarVar
from pyomo.core.expr.numvalue import is_constant, NumericConstant
from pyomo.core.expr.numeric_expr import (
    NegationExpression,
    PowExpression,
    ProductExpression,
    MonomialTermExpression,
    DivisionExpression,
    SumExpression,
    LinearExpression,
    UnaryFunctionExpression,
    NPV_NegationExpression,
    NPV_PowExpression,
    NPV_ProductExpression,
    NPV_DivisionExpression,
    NPV_SumExpression,
    NPV_UnaryFunctionExpression,
)
from pyomo.core.expr.relational_expr import (
    EqualityExpression,
    InequalityExpression,
    RangedExpression,
)
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor
from pyomo.gdp.disjunct import AutoLinkedBinaryVar

logger = logging.getLogger(__name__)

scip, scip_available = attempt_import('pyscipopt')


class ScipConfig(BranchAndBoundConfig):
    def __init__(
        self,
        description=None,
        doc=None,
        implicit=False,
        implicit_domain=None,
        visibility=0,
    ):
        BranchAndBoundConfig.__init__(
            self,
            description=description,
            doc=doc,
            implicit=implicit,
            implicit_domain=implicit_domain,
            visibility=visibility,
        )
        self.warmstart_discrete_vars: bool = self.declare(
            'warmstart_discrete_vars',
            ConfigValue(
                default=False,
                domain=bool,
                description="If True, the current values of the integer variables "
                "will be passed to Scip.",
            ),
        )


def _handle_var(node, data, opt, visitor):
    if node not in opt._pyomo_var_to_solver_var_map:
        scip_var = opt._add_var(node)
    else:
        scip_var = opt._pyomo_var_to_solver_var_map[node]
    return scip_var


def _handle_param(node, data, opt, visitor):
    # for the persistent interface, we create scip variables in place
    # of parameters. However, this makes things complicated for range
    # constraints because scip does not allow variables in the
    # lower and upper parts of range constraints
    if visitor.in_range:
        return node.value
    if not opt.is_persistent():
        return node.value
    if node.is_constant():
        return node.value
    if node not in opt._pyomo_param_to_solver_param_map:
        scip_param = opt._add_param(node)
    else:
        scip_param = opt._pyomo_param_to_solver_param_map[node]
    return scip_param


def _handle_constant(node, data, opt, visitor):
    return node.value


def _handle_float(node, data, opt, visitor):
    return float(node)


def _handle_negation(node, data, opt, visitor):
    return -data[0]


def _handle_pow(node, data, opt, visitor):
    x, y = data  # x ** y = exp(log(x**y)) = exp(y*log(x))
    if is_constant(node.args[1]):
        return x**y
    else:
        xlb, xub = compute_bounds_on_expr(node.args[0])
        if xlb > 0:
            return scip.exp(y * scip.log(x))
        else:
            return x**y  # scip will probably raise an error here


def _handle_product(node, data, opt, visitor):
    assert len(data) == 2
    return data[0] * data[1]


def _handle_division(node, data, opt, visitor):
    return data[0] / data[1]


def _handle_sum(node, data, opt, visitor):
    return sum(data)


def _handle_exp(node, data, opt, visitor):
    return scip.exp(data[0])


def _handle_log(node, data, opt, visitor):
    return scip.log(data[0])


def _handle_log10(node, data, opt, visitor):
    return scip.log(data[0]) / math.log(10)


def _handle_sin(node, data, opt, visitor):
    return scip.sin(data[0])


def _handle_cos(node, data, opt, visitor):
    return scip.cos(data[0])


def _handle_sqrt(node, data, opt, visitor):
    return scip.sqrt(data[0])


def _handle_abs(node, data, opt, visitor):
    return abs(data[0])


def _handle_tan(node, data, opt, visitor):
    return scip.sin(data[0]) / scip.cos(data[0])


def _handle_tanh(node, data, opt, visitor):
    x = data[0]
    _exp = scip.exp
    return (_exp(x) - _exp(-x)) / (_exp(x) + _exp(-x))


_unary_map = {
    'exp': _handle_exp,
    'log': _handle_log,
    'sin': _handle_sin,
    'cos': _handle_cos,
    'sqrt': _handle_sqrt,
    'abs': _handle_abs,
    'tan': _handle_tan,
    'log10': _handle_log10,
    'tanh': _handle_tanh,
}


def _handle_unary(node, data, opt, visitor):
    if node.getname() in _unary_map:
        return _unary_map[node.getname()](node, data, opt, visitor)
    else:
        raise NotImplementedError(f'unable to handle unary expression: {str(node)}')


def _handle_equality(node, data, opt, visitor):
    return data[0] == data[1]


def _handle_ranged(node, data, opt, visitor):
    # note that the lower and upper parts of the
    # range constraint cannot have variables
    return data[0] <= (data[1] <= data[2])


def _handle_inequality(node, data, opt, visitor):
    return data[0] <= data[1]


def _handle_named_expression(node, data, opt, visitor):
    return data[0]


def _handle_unit(node, data, opt, visitor):
    return node.value


_operator_map = {
    NegationExpression: _handle_negation,
    PowExpression: _handle_pow,
    ProductExpression: _handle_product,
    MonomialTermExpression: _handle_product,
    DivisionExpression: _handle_division,
    SumExpression: _handle_sum,
    LinearExpression: _handle_sum,
    UnaryFunctionExpression: _handle_unary,
    NPV_NegationExpression: _handle_negation,
    NPV_PowExpression: _handle_pow,
    NPV_ProductExpression: _handle_product,
    NPV_DivisionExpression: _handle_division,
    NPV_SumExpression: _handle_sum,
    NPV_UnaryFunctionExpression: _handle_unary,
    EqualityExpression: _handle_equality,
    RangedExpression: _handle_ranged,
    InequalityExpression: _handle_inequality,
    ScalarExpression: _handle_named_expression,
    ExpressionData: _handle_named_expression,
    VarData: _handle_var,
    ScalarVar: _handle_var,
    ParamData: _handle_param,
    ScalarParam: _handle_param,
    float: _handle_float,
    int: _handle_float,
    AutoLinkedBinaryVar: _handle_var,
    _PyomoUnit: _handle_unit,
    NumericConstant: _handle_constant,
}


class _PyomoToScipVisitor(StreamBasedExpressionVisitor):
    def __init__(self, solver, **kwds):
        super().__init__(**kwds)
        self.solver = solver
        self.in_range = False

    def initializeWalker(self, expr):
        self.in_range = False
        return True, None

    def exitNode(self, node, data):
        nt = type(node)
        if nt in _operator_map:
            return _operator_map[nt](node, data, self.solver, self)
        elif nt in native_numeric_types:
            _operator_map[nt] = _handle_float
            return _handle_float(node, data, self.solver, self)
        else:
            raise NotImplementedError(f'unrecognized expression type: {nt}')

    def enterNode(self, node):
        if type(node) is RangedExpression:
            self.in_range = True
        return None, []


class ScipSolutionLoader(SolutionLoader):
    def __init__(self, solver_model, var_map, con_map, pyomo_model, opt) -> None:
        super().__init__()
        self._solver_model = solver_model
        self._var_map = var_map
        self._con_map = con_map
        self._pyomo_model = pyomo_model
        # make sure the scip model does not get freed until the solution loader is garbage collected
        self._opt = opt
        self._active_solution_id = 0

    def _set_solution_id(self, solution_id: int) -> int:
        if solution_id is None:
            solution_id = 0
        previous_id = self._active_solution_id
        self._active_solution_id = solution_id
        return previous_id

    def get_number_of_solutions(self) -> int:
        return self._solver_model.getNSols()

    def get_solution_ids(self) -> List:
        return list(range(self.get_number_of_solutions()))

    def get_vars(
        self, vars_to_load: Optional[Sequence[VarData]] = None
    ) -> Mapping[VarData, float]:
        if self.get_number_of_solutions() == 0:
            raise NoSolutionError()
        if vars_to_load is None:
            vars_to_load = list(self._var_map.keys())
        sol = self._solver_model.getSols()[self._active_solution_id]
        res = ComponentMap()
        for v in vars_to_load:
            sv = self._var_map[v]
            res[v] = sol[sv]
        return res
