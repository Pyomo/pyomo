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

from collections.abc import Iterable, Mapping, MutableSet, Sequence
from typing import Optional

from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.numeric_types import value
from pyomo.contrib.solver.common.util import collect_vars_and_named_exprs
from pyomo.contrib.solver.solvers.knitro.typing import Function
from pyomo.core.base.block import BlockData
from pyomo.core.base.constraint import Constraint, ConstraintData
from pyomo.core.base.enums import SortComponents
from pyomo.core.base.expression import Expression
from pyomo.core.base.objective import Objective, ObjectiveData
from pyomo.core.base.var import VarData
from pyomo.core.expr.calculus.diff_with_pyomo import reverse_sd


def get_active_objectives(block: BlockData) -> list[ObjectiveData]:
    """Retrieve all active ObjectiveData objects from a Pyomo Block.

    Args:
        block (BlockData): The Pyomo block to search for objectives.

    Returns:
        list[ObjectiveData]: A sorted list of all active objectives in the block.

    """
    generator = block.component_data_objects(
        Objective, descend_into=True, active=True, sort=True
    )
    return list(generator)


def get_active_constraints(block: BlockData) -> list[ConstraintData]:
    """Retrieve all active ConstraintData objects from a Pyomo Block.

    Args:
        block (BlockData): The Pyomo block to search for constraints.

    Returns:
        list[ConstraintData]: A sorted list of all active constraints in the block.

    """
    generator = block.component_data_objects(
        Constraint, descend_into=True, active=True, sort=SortComponents.deterministic
    )
    return list(generator)


class KnitroModelData:
    """Intermediate representation of a Pyomo model for KNITRO.

    Collects all active objectives, constraints, and referenced variables from a Pyomo Block.
    This class is used to extract and organize model data before passing it to the solver.

    Attributes:
        objs (list[ObjectiveData]): list of active objectives.
        cons (list[ConstraintData]): list of active constraints.
        variables (list[VarData]): list of all referenced variables.

    """

    objs: list[ObjectiveData]
    cons: list[ConstraintData]
    variables: list[VarData]
    _vars: MutableSet[VarData]

    def __init__(self, block: Optional[BlockData] = None) -> None:
        """Initialize a Problem instance.

        Args:
            block (Optional[BlockData]): Pyomo block to initialize from. If None,
                creates an empty problem that can be populated later.

        """
        self._vars = ComponentSet()
        self.objs = []
        self.cons = []
        self.variables = []
        if block is not None:
            self.add_block(block)

    def clear(self) -> None:
        """Clear all objectives, constraints, and variables from the problem."""
        self.objs.clear()
        self.cons.clear()
        self.variables.clear()
        self._vars.clear()

    def set_block(self, block: BlockData) -> None:
        """Replace the current problem data with data from a new block.

        Args:
            block (BlockData): The Pyomo block to extract data from.

        """
        self.clear()
        self.add_block(block)

    def add_block(self, block: BlockData) -> None:
        """Add objectives, constraints, and variables from a block to the problem.

        Args:
            block (BlockData): The Pyomo block to extract data from.

        """
        new_objs = get_active_objectives(block)
        new_cons = get_active_constraints(block)
        self.objs.extend(new_objs)
        self.cons.extend(new_cons)

        # Collect variables from objectives
        for obj in new_objs:
            _, variables, _, _ = collect_vars_and_named_exprs(obj.expr)  # type: ignore
            for var in variables:
                self._vars.add(var)

        # Collect variables from constraints
        for con in new_cons:
            _, variables, _, _ = collect_vars_and_named_exprs(con.body)  # type: ignore
            for var in variables:
                self._vars.add(var)

        # Update the variables list with unique variables only
        self.variables = list(self._vars)


def set_var_values(
    variables: Iterable[VarData], values: Sequence[float], var_map: Mapping[int, int]
) -> None:
    """Set the values of a list of Pyomo variables from a list of values.

    Args:
        variables (Iterable[VarData]): The variables to set.
        values (list[float]): The list of values to assign to the variables.
        var_map (Mapping[int, int]): A mapping from variable id to index in the
            values list.

    """
    for var in variables:
        i = var_map[id(var)]
        var.set_value(values[i])


class NonlinearExpressionData(Function):
    """Holds the data required to evaluate a non-linear expression.

    This class stores a Pyomo expression along with its variables and can compute
    gradient and Hessian information for use with optimization solvers.

    Attributes:
        variables (list[VarData]): list of variables referenced in the expression.
        func_expr (Expression): The Pyomo expression representing the non-linear function.
        grad_map (Mapping[VarData, Expression]): Gradient expressions mapped by variable.
        hess_map (Mapping[tuple[VarData, VarData], Expression]): Hessian expressions
            mapped by variable pairs.
        diff_order (int): Level of differentiation to compute:
            - 0: function evaluation only
            - 1: function + gradient
            - 2: function + gradient + hessian

    """

    variables: list[VarData]
    func_expr: Expression
    grad_map: Mapping[VarData, Expression]
    hess_map: Mapping[tuple[VarData, VarData], Expression]
    diff_order: int

    def __init__(
        self,
        expr: Expression,
        variables: Iterable[VarData],
        var_map: Mapping[int, int],
        diff_order: int = 0,
    ) -> None:
        """Initialize NonlinearExpressionData.

        Args:
            expr (Expression): The Pyomo expression to evaluate.
            variables (Iterable[VarData]): Variables referenced in the expression.
            diff_order (int): Level of differentiation to compute:
                - 0: function evaluation only
                - 1: function + gradient
                - 2: function + gradient + hessian

        """
        self.func_expr = expr
        self.variables = list(variables)
        self.diff_order = diff_order
        self._var_map = var_map
        if diff_order >= 1:
            self.compute_gradient()
        if diff_order >= 2:
            self.compute_hessian()

    @property
    def grad_vars(self) -> list[VarData]:
        """Get the list of variables for which gradients are available.

        Returns:
            list[VarData]: Variables with gradient information.

        """
        return list(self.grad_map.keys())

    @property
    def hess_vars(self) -> list[tuple[VarData, VarData]]:
        """Get the list of variable pairs for which Hessian entries are available.

        Returns:
            list[tuple[VarData, VarData]]: Variable pairs with Hessian information.

        """
        return list(self.hess_map.keys())

    def compute_gradient(self) -> None:
        """Compute gradient expressions for the nonlinear expression.

        This method computes the gradient of the expression with respect to all
        variables and stores the results in the grad attribute.
        """
        derivative = reverse_sd(self.func_expr)
        variables = ComponentSet(self.variables)
        self.grad_map = ComponentMap()
        for v, expr in derivative.items():
            if v in variables:
                self.grad_map[v] = expr

    def compute_hessian(self) -> None:
        """Compute Hessian expressions for the nonlinear expression.

        This method computes the Hessian matrix of the expression with respect to all
        variables and stores the results in the hess attribute. Only the upper triangle
        of the Hessian is stored to avoid redundancy.

        Note:
            This method requires that compute_gradient() has been called first.

        """
        variables = ComponentSet(self.variables)
        self.hess_map = ComponentMap()
        for v1, grad_expr in self.grad_map.items():
            derivative = reverse_sd(grad_expr)
            for v2, hess_expr in derivative.items():
                if v2 not in variables:
                    continue
                # Store only upper triangle: ensure var1 <= var2 by ID
                var1, var2 = (v1, v2) if id(v1) <= id(v2) else (v2, v1)
                if (var1, var2) not in self.hess_map:
                    self.hess_map[(var1, var2)] = hess_expr

    def evaluate(self, x: list[float]) -> float:
        set_var_values(self.variables, x, self._var_map)
        return value(self.func_expr)

    def gradient(self, x: list[float]) -> list[float]:
        set_var_values(self.variables, x, self._var_map)
        return [value(g) for g in self.grad_map.values()]

    def hessian(self, x: list[float], mu: float) -> list[float]:
        set_var_values(self.variables, x, self._var_map)
        return [mu * value(h) for h in self.hess_map.values()]
