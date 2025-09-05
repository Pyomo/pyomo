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


from collections.abc import Iterable, Mapping, MutableMapping
from typing import Any, List, Optional

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
    _var_map: MutableMapping[int, VarData]

    def __init__(self, block: Optional[BlockData] = None):
        self._var_map = {}
        self.objs = []
        self.cons = []
        self.variables = []
        if block is not None:
            self.add_block(block)

    def clear(self):
        self.objs.clear()
        self.cons.clear()
        self.variables.clear()
        self._var_map.clear()

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
                self._var_map[id(v)] = v
        for con in new_cons:
            _, variables, _, _ = collect_vars_and_named_exprs(con.body)
            for v in variables:
                self._var_map[id(v)] = v
        self.variables.extend(self._var_map.values())


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

    def __init__(
        self,
        expr: Optional[Any],
        variables: Iterable[VarData],
        *,
        compute_grad: bool = True,
    ):
        self.body = expr
        self.variables = list(variables)
        if compute_grad:
            self.grad = reverse_sd(self.body)
        else:
            self.grad = None

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
            # Set the values of the Pyomo variables from the solver's vector `x`
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
            # Set the values of the Pyomo variables from the solver's vector `x`
            for var in self.variables:
                i = vmap[id(var)]
                var.set_value(x[i])
            return [value(self.grad.get(var, 0.0)) for var in self.variables]

        return _grad
