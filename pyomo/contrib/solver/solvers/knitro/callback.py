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
from typing import Any, Optional, Protocol

from pyomo.contrib.solver.solvers.knitro.typing import (
    Callback,
    CallbackFunction,
    CallbackRequest,
    CallbackResult,
    Function,
)

CallbackHandlerFunction = Callable[[CallbackRequest, CallbackResult], int]


class CallbackHandler(Protocol):
    _function: Function

    def func(self, req: CallbackRequest, res: CallbackResult) -> int: ...
    def grad(self, req: CallbackRequest, res: CallbackResult) -> int: ...
    def hess(self, req: CallbackRequest, res: CallbackResult) -> int: ...

    def expand(self) -> Callback:
        procs = (self.func, self.grad, self.hess)
        return Callback(*map(self._expand, procs))

    @staticmethod
    def _expand(proc: CallbackHandlerFunction) -> CallbackFunction:
        def _expanded(
            kc: Any,
            cb: Any,
            req: CallbackRequest,
            res: CallbackResult,
            user_data: Any = None,
        ) -> int:
            return proc(req, res)

        return _expanded


class ObjectiveCallbackHandler(CallbackHandler):
    def __init__(self, function: Function) -> None:
        self._function = function

    def func(self, req: CallbackRequest, res: CallbackResult) -> int:
        res.obj = self._function.evaluate(req.x)
        return 0

    def grad(self, req: CallbackRequest, res: CallbackResult) -> int:
        res.objGrad[:] = self._function.gradient(req.x)
        return 0

    def hess(self, req: CallbackRequest, res: CallbackResult) -> int:
        res.hess[:] = self._function.hessian(req.x, req.sigma)
        return 0


class ConstraintCallbackHandler(CallbackHandler):
    i: int

    def __init__(self, i: int, function: Function) -> None:
        self.i = i
        self._function = function

    def func(self, req: CallbackRequest, res: CallbackResult) -> int:
        res.c[:] = [self._function.evaluate(req.x)]
        return 0

    def grad(self, req: CallbackRequest, res: CallbackResult) -> int:
        res.jac[:] = self._function.gradient(req.x)
        return 0

    def hess(self, req: CallbackRequest, res: CallbackResult) -> int:
        res.hess[:] = self._function.hessian(req.x, req.lambda_[self.i])
        return 0


def build_callback_handler(function: Function, idx: Optional[int]) -> CallbackHandler:
    if idx is None:
        return ObjectiveCallbackHandler(function)
    return ConstraintCallbackHandler(idx, function)
