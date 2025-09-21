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


from collections.abc import Iterable, Mapping, MutableSet
from typing import Any, List, Optional, Tuple

from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.numeric_types import value
from pyomo.contrib.solver.common.util import collect_vars_and_named_exprs
from pyomo.core.base.block import BlockData
from pyomo.core.base.constraint import Constraint, ConstraintData
from pyomo.core.base.objective import Objective, ObjectiveData
from pyomo.core.base.var import VarData
from pyomo.core.expr.calculus.diff_with_pyomo import reverse_sd


def get_active_objectives(block: BlockData) -> List[ObjectiveData]:
    """
    Retrieve all active ObjectiveData objects from a Pyomo Block.

    Args:
        block (BlockData): The Pyomo block to search for objectives.

    Returns:
        List[ObjectiveData]: A sorted list of all active objectives in the block.
    """
    generator = block.component_data_objects(
        Objective, descend_into=True, active=True, sort=True
    )
    return list(generator)


def get_active_constraints(block: BlockData) -> List[ConstraintData]:
    """
    Retrieve all active ConstraintData objects from a Pyomo Block.

    Args:
        block (BlockData): The Pyomo block to search for constraints.

    Returns:
        List[ConstraintData]: A sorted list of all active constraints in the block.
    """
    generator = block.component_data_objects(
        Constraint, descend_into=True, active=True, sort=True
    )
    return list(generator)


class Problem:
    """
    Intermediate representation of a Pyomo model for KNITRO.

    Collects all active objectives, constraints, and referenced variables from a Pyomo Block.
    This class is used to extract and organize model data before passing it to the solver.

    Attributes:
        objs (List[ObjectiveData]): List of active objectives.
        cons (List[ConstraintData]): List of active constraints.
        variables (List[VarData]): List of all referenced variables.
    """

    objs: List[ObjectiveData]
    cons: List[ConstraintData]
    variables: List[VarData]
    _vars: MutableSet[VarData]

    def __init__(self, block: Optional[BlockData] = None):
        self._vars = ComponentSet()
        self.objs = []
        self.cons = []
        self.variables = []
        if block is not None:
            self.add_block(block)

    def clear(self):
        self.objs.clear()
        self.cons.clear()
        self.variables.clear()
        self._vars.clear()

    def set_block(self, block: BlockData):
        self.clear()
        self.add_block(block)

    def add_block(self, block: BlockData):
        new_objs = get_active_objectives(block)
        new_cons = get_active_constraints(block)
        self.objs.extend(new_objs)
        self.cons.extend(new_cons)

        for obj in new_objs:
            _, variables, _, _ = collect_vars_and_named_exprs(obj.expr)
            for v in variables:
                self._vars.add(v)
        for con in new_cons:
            _, variables, _, _ = collect_vars_and_named_exprs(con.body)
            for v in variables:
                self._vars.add(v)
        self.variables.extend(self._vars)


class NonlinearExpressionData:
    """
    Holds the data required to evaluate a non-linear expression.

    Attributes:
        body (Optional[Any]): The Pyomo expression representing the non-linear body.
        variables (List[VarData]): List of variables referenced in the expression.
        grad (Optional[Mapping[VarData, Any]]): Gradient information for the non-linear expression.
    """

    body: Optional[Any]
    variables: List[VarData]
    grad: Optional[Mapping[VarData, Any]]
    hessian: Optional[Mapping[Tuple[VarData, VarData], Any]]

    def __init__(
        self,
        expr: Optional[Any],
        variables: Iterable[VarData],
        *,
        compute_grad: bool = True,
        compute_hess: bool = False,
    ):
        self.body = expr
        self.variables = list(variables)
        self.grad = None
        self.hessian = None
        if compute_grad:
            self.compute_gradient()
        if compute_hess:
            if not compute_grad:
                msg = "Hessian computation requires gradient computation."
                raise ValueError(msg)
            self.compute_hessian()

    @property
    def grad_vars(self) -> List[VarData]:
        if self.grad is None:
            msg = "Gradient information is not available for this expression."
            raise ValueError(msg)
        return list(self.grad.keys())

    @property
    def hess_vars(self) -> List[Tuple[VarData, VarData]]:
        if self.hessian is None:
            msg = "Hessian information is not available for this expression."
            raise ValueError(msg)
        return list(self.hessian.keys())

    def compute_gradient(self):
        diff_map = reverse_sd(self.body)
        variables = ComponentSet(self.variables)
        self.grad = ComponentMap()
        for v, expr in diff_map.items():
            if v in variables:
                self.grad[v] = expr

    def compute_hessian(self):
        variables = ComponentSet(self.variables)
        self.hessian = ComponentMap()
        for v1, expr in self.grad.items():
            diff_map = reverse_sd(expr)
            for v2, diff_expr in diff_map.items():
                if v2 not in variables:
                    continue
                var1 = v1
                var2 = v2
                if id(var1) > id(var2):
                    var1, var2 = var2, var1
                if (var1, var2) not in self.hessian:
                    self.hessian[(var1, var2)] = diff_expr
                else:
                    self.hessian[(var1, var2)] += diff_expr

    def create_evaluator(self, vmap: Mapping[int, int]):
        """
        Create a callable evaluator for the non-linear expression.

        Args:
            vmap (Mapping[int, int]): A mapping from variable id to index in the solver's variable vector.

        Returns:
            Callable[[List[float]], float]: A function that takes a list of variable values (x)
            and returns the evaluated value of the expression.
        """

        def _fn(x: List[float]) -> float:
            for var in self.variables:
                i = vmap[id(var)]
                var.set_value(x[i])
            return value(self.body)

        return _fn

    def create_gradient_evaluator(self, vmap: Mapping[int, int]):
        """
        Create a callable gradient evaluator for the non-linear expression.

        Args:
            vmap (Mapping[int, int]): A mapping from variable id to index in the solver's variable vector.

        Returns:
            Callable[[List[float]], List[float]]: A function that takes a list of variable values (x)
            and returns the gradient of the expression with respect to its variables.

        Raises:
            ValueError: If gradient information is not available for this expression.
        """

        if self.grad is None:
            msg = "Gradient information is not available for this expression."
            raise ValueError(msg)

        def _grad(x: List[float]) -> List[float]:
            for var in self.variables:
                i = vmap[id(var)]
                var.set_value(x[i])
            return [value(expr) for expr in self.grad.values()]

        return _grad

    def create_hessian_evaluator(self, vmap: Mapping[int, int]):
        """
        Create a callable Hessian evaluator for the non-linear expression.

        Args:
            vmap (Mapping[int, int]): A mapping from variable id to index in the solver's variable vector.
        Returns:
            Callable[[List[float]], List[Tuple[int, int, float]]]: A function that takes a list of variable values (x)
            and returns the Hessian of the expression as a list of (row, column, value) tuples.
        Raises:
            ValueError: If Hessian information is not available for this expression.
        """

        if self.hessian is None:
            msg = "Hessian information is not available for this expression."
            raise ValueError(msg)

        def _hess(x: List[float], mu: float) -> List[float]:
            for var in self.variables:
                i = vmap[id(var)]
                var.set_value(x[i])
            return [mu * value(expr) for expr in self.hessian.values()]

        return _hess
