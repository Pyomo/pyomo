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

from collections.abc import Callable
from typing import Any, Optional, Protocol, TypeVar, Union

from pyomo.common.enums import Enum
from pyomo.core.base.constraint import ConstraintData
from pyomo.core.base.var import VarData


class BoundType(Enum):
    EQ = 0
    LO = 1
    UP = 2


class StructureType(Enum):
    CONSTANT = 0
    LINEAR = 1
    QUADRATIC = 2


class ValueType(Enum):
    PRIMAL = 0
    DUAL = 1

    @property
    def sign(self) -> float:
        return -1.0 if self == ValueType.DUAL else 1.0


ItemType = Union[VarData, ConstraintData]
T = TypeVar("T", bound=ItemType)


class Atom(Protocol):
    def func(self, x: list[float]) -> float: ...
    def grad(self, x: list[float]) -> list[float]: ...
    def hess(self, x: list[float], mu: float) -> list[float]: ...


class Request(Protocol):
    x: list[float]
    sigma: float
    lambda_: list[float]


class Result(Protocol):
    obj: float
    c: list[float]
    objGrad: list[float]
    jac: list[float]
    hess: list[float]


class Callback:
    def __init__(
        self,
        func: Callable[[Any, Any, Request, Result, Optional[Any]], int],
        grad: Callable[[Any, Any, Request, Result, Optional[Any]], int],
        hess: Callable[[Any, Any, Request, Result, Optional[Any]], int],
    ) -> None:
        self.func = func
        self.grad = grad
        self.hess = hess


class UnreachableError(Exception):
    """Raised when code reaches a theoretically unreachable state."""

    pass
