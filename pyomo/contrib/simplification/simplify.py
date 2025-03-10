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

import logging
import warnings

from pyomo.common.enums import NamedIntEnum
from pyomo.core.expr.sympy_tools import sympy2pyomo_expression, sympyify_expression
from pyomo.core.expr.numeric_expr import NumericExpression
from pyomo.core.expr.numvalue import value, is_constant

from pyomo.contrib.simplification.ginac import (
    interface as ginac_interface,
    interface_available as ginac_available,
)


def simplify_with_sympy(expr: NumericExpression):
    if is_constant(expr):
        return value(expr)
    object_map, sympy_expr = sympyify_expression(expr, keep_mutable_parameters=True)
    new_expr = sympy2pyomo_expression(sympy_expr.simplify(), object_map)
    if is_constant(new_expr):
        new_expr = value(new_expr)
    return new_expr


def simplify_with_ginac(expr: NumericExpression, ginac_interface):
    if is_constant(expr):
        return value(expr)
    ginac_expr = ginac_interface.to_ginac(expr)
    return ginac_interface.from_ginac(ginac_expr.normal())


class Simplifier(object):
    class Mode(NamedIntEnum):
        auto = 0
        sympy = 1
        ginac = 2

    def __init__(
        self, suppress_no_ginac_warnings: bool = False, mode: Mode = Mode.auto
    ) -> None:
        if mode == Simplifier.Mode.auto:
            if ginac_available:
                mode = Simplifier.Mode.ginac
            else:
                if not suppress_no_ginac_warnings:
                    msg = (
                        "GiNaC does not seem to be available. Using SymPy. "
                        + "Note that the GiNaC interface is significantly faster."
                    )
                    logging.getLogger(__name__).warning(msg)
                    warnings.warn(msg)
                mode = Simplifier.Mode.sympy

        if mode == Simplifier.Mode.ginac:
            self.gi = ginac_interface.GinacInterface(False)
            self.simplify = self._simplify_with_ginac
        else:
            self.simplify = self._simplify_with_sympy

    def _simplify_with_ginac(self, expr: NumericExpression):
        return simplify_with_ginac(expr, self.gi)

    def _simplify_with_sympy(self, expr: NumericExpression):
        return simplify_with_sympy(expr)
