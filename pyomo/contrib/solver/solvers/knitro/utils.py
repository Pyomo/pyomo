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
from typing import Any, Callable, List, Optional, Tuple

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
        """
        Initialize a Problem instance.

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

    def clear(self):
        """Clear all objectives, constraints, and variables from the problem."""
        self.objs.clear()
        self.cons.clear()
        self.variables.clear()
        self._vars.clear()

    def set_block(self, block: BlockData):
        """
        Replace the current problem data with data from a new block.

        Args:
            block (BlockData): The Pyomo block to extract data from.
        """
        self.clear()
        self.add_block(block)

    def add_block(self, block: BlockData):
        """
        Add objectives, constraints, and variables from a block to the problem.

        Args:
            block (BlockData): The Pyomo block to extract data from.
        """
        new_objs = get_active_objectives(block)
        new_cons = get_active_constraints(block)
        self.objs.extend(new_objs)
        self.cons.extend(new_cons)

        # Collect variables from objectives
        for obj in new_objs:
            _, variables, _, _ = collect_vars_and_named_exprs(obj.expr)
            self._vars.update(variables)

        # Collect variables from constraints
        for con in new_cons:
            _, variables, _, _ = collect_vars_and_named_exprs(con.body)
            self._vars.update(variables)

        # Update the variables list with unique variables only
        self.variables = list(self._vars)


class NonlinearExpressionData:
    """
    Holds the data required to evaluate a non-linear expression.

    This class stores a Pyomo expression along with its variables and can compute
    gradient and Hessian information for use with optimization solvers.

    Attributes:
        body (Optional[Any]): The Pyomo expression representing the non-linear body.
        variables (List[VarData]): List of variables referenced in the expression.
        grad (Optional[Mapping[VarData, Any]]): Gradient expressions mapped by variable.
        hess (Optional[Mapping[Tuple[VarData, VarData], Any]]): Hessian expressions
            mapped by variable pairs.
    """

    body: Optional[Any]
    variables: List[VarData]
    grad: Optional[Mapping[VarData, Any]]
    hess: Optional[Mapping[Tuple[VarData, VarData], Any]]

    def __init__(
        self,
        expr: Optional[Any],
        variables: Iterable[VarData],
        *,
        differentiation_order: int = 0,
    ):
        """
        Initialize NonlinearExpressionData.

        Args:
            expr (Optional[Any]): The Pyomo expression to evaluate.
            variables (Iterable[VarData]): Variables referenced in the expression.
            differentiation_order (int): Level of differentiation to compute:
                - 0: function evaluation only
                - 1: function + gradient
                - 2: function + gradient + Hessian
        """
        self.body = expr
        self.variables = list(variables)
        self.grad = None
        self.hess = None
        if differentiation_order > 0:
            self.compute_gradient()
        if differentiation_order > 1:
            self.compute_hessian()

    @property
    def grad_vars(self) -> List[VarData]:
        """
        Get the list of variables for which gradients are available.

        Returns:
            List[VarData]: Variables with gradient information.

        Raises:
            ValueError: If gradient information is not available.
        """
        if self.grad is None:
            msg = "Gradient information is not available for this expression."
            raise ValueError(msg)
        return list(self.grad.keys())

    @property
    def hess_vars(self) -> List[Tuple[VarData, VarData]]:
        """
        Get the list of variable pairs for which Hessian entries are available.

        Returns:
            List[Tuple[VarData, VarData]]: Variable pairs with Hessian information.

        Raises:
            ValueError: If Hessian information is not available.
        """
        if self.hess is None:
            msg = "Hessian information is not available for this expression."
            raise ValueError(msg)
        return list(self.hess.keys())

    def compute_gradient(self):
        """
        Compute gradient expressions for the nonlinear expression.

        This method computes the gradient of the expression with respect to all
        variables and stores the results in the grad attribute.
        """
        derivative = reverse_sd(self.body)
        variables = ComponentSet(self.variables)
        self.grad = ComponentMap()
        for v, expr in derivative.items():
            if v in variables:
                self.grad[v] = expr

    def compute_hessian(self):
        """
        Compute Hessian expressions for the nonlinear expression.

        This method computes the Hessian matrix of the expression with respect to all
        variables and stores the results in the hess attribute. Only the upper triangle
        of the Hessian is stored to avoid redundancy.

        Note:
            This method requires that compute_gradient() has been called first.
        """
        if self.grad is None:
            msg = "Gradient must be computed before Hessian. Call compute_gradient() first."
            raise ValueError(msg)

        variables = ComponentSet(self.variables)
        self.hess = ComponentMap()
        for v1, grad_expr in self.grad.items():
            derivative = reverse_sd(grad_expr)
            for v2, hess_expr in derivative.items():
                if v2 not in variables:
                    continue
                # Store only upper triangle: ensure var1 <= var2 by ID
                var1, var2 = (v1, v2) if id(v1) <= id(v2) else (v2, v1)
                if (var1, var2) not in self.hess:
                    self.hess[(var1, var2)] = hess_expr

    def create_evaluator(
        self, vmap: Mapping[int, int]
    ) -> Callable[[List[float]], float]:
        """
        Create a callable evaluator for the non-linear expression.

        Args:
            vmap (Mapping[int, int]): A mapping from variable id to index in the
                solver's variable vector.

        Returns:
            Callable[[List[float]], float]: A function that takes a list of variable
                values (x) and returns the evaluated value of the expression.
        """

        def _fn(x: List[float]) -> float:
            for var in self.variables:
                i = vmap[id(var)]
                var.set_value(x[i])
            return value(self.body)

        return _fn

    def create_gradient_evaluator(
        self, vmap: Mapping[int, int]
    ) -> Callable[[List[float]], List[float]]:
        """
        Create a callable gradient evaluator for the non-linear expression.

        Args:
            vmap (Mapping[int, int]): A mapping from variable id to index in the
                solver's variable vector.

        Returns:
            Callable[[List[float]], List[float]]: A function that takes a list of
                variable values (x) and returns the gradient of the expression with
                respect to its variables.

        """
        if self.grad is None:
            msg = "Gradient information is not available for this expression."
            raise ValueError(msg)

        def _grad(x: List[float]) -> List[float]:
            # Set all variables, not just gradient variables, to ensure consistency
            for var in self.variables:
                i = vmap[id(var)]
                var.set_value(x[i])
            return [value(expr) for expr in self.grad.values()]

        return _grad

    def create_hessian_evaluator(
        self, vmap: Mapping[int, int]
    ) -> Callable[[List[float], float], List[float]]:
        """
        Create a callable Hessian evaluator for the non-linear expression.

        Args:
            vmap (Mapping[int, int]): A mapping from variable id to index in the
                solver's variable vector.

        Returns:
            Callable[[List[float], float], List[float]]: A function that takes a list
                of variable values (x) and a multiplier (mu) and returns the scaled
                Hessian of the expression as a list of values corresponding to the
                variable pairs in self.hess.

        """
        if self.hess is None:
            msg = "Hessian information is not available for this expression."
            raise ValueError(msg)

        def _hess(x: List[float], mu: float) -> List[float]:
            # Set all variables to ensure consistency
            for var in self.variables:
                i = vmap[id(var)]
                var.set_value(x[i])
            return [mu * value(expr) for expr in self.hess.values()]

        return _hess
