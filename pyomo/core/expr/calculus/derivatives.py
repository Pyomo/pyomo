#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import enum
from .diff_with_sympy import differentiate as sympy_diff
from .diff_with_pyomo import reverse_sd, reverse_ad


class Modes(str, enum.Enum):
    sympy='sympy'
    reverse_symbolic='reverse_symbolic'
    reverse_numeric='reverse_numeric'

    # Overloading __str__ is needed to match the behavior of the old
    # pyutilib.enum class (removed June 2020). There are spots in the
    # code base that expect the string representation for items in the
    # enum to not include the class name. New uses of enum shouldn't
    # need to do this.
    def __str__(self):
        return self.value


def differentiate(expr, wrt=None, wrt_list=None, mode=Modes.reverse_numeric):
    """Return derivative of expression.

    This function returns the derivative of expr with respect to one or
    more variables.  The type of the return value depends on the
    arguments wrt, wrt_list, and mode. See below for details.

    Parameters
    ----------
    expr: pyomo.core.expr.numeric_expr.ExpressionBase
        The expression to differentiate
    wrt: pyomo.core.base.var._GeneralVarData
        If specified, this function will return the derivative with
        respect to wrt. wrt is normally a _GeneralVarData, but could
        also be a _ParamData. wrt and wrt_list cannot both be specified.
    wrt_list: list of pyomo.core.base.var._GeneralVarData
        If specified, this function will return the derivative with
        respect to each element in wrt_list.  A list will be returned
        where the values are the derivatives with respect to the
        corresponding entry in wrt_list.
    mode: pyomo.core.expr.calculus.derivatives.Modes
        Specifies the method to use for differentiation. Should be one
        of the members of the Modes enum:

            Modes.sympy:
                The pyomo expression will be converted to a sympy
                expression. Differentiation will then be done with
                sympy, and the result will be converted back to a pyomo
                expression.  The sympy mode only does symbolic
                differentiation. The sympy mode requires exactly one of
                wrt and wrt_list to be specified.
            Modes.reverse_symbolic:
                Symbolic differentiation will be performed directly with
                the pyomo expression in reverse mode. If neither wrt nor
                wrt_list are specified, then a ComponentMap is returned
                where there will be a key for each node in the
                expression tree, and the values will be the symbolic
                derivatives.
            Modes.reverse_numeric:
                Numeric differentiation will be performed directly with
                the pyomo expression in reverse mode. If neither wrt nor
                wrt_list are specified, then a ComponentMap is returned
                where there will be a key for each node in the
                expression tree, and the values will be the floating
                point values of the derivatives at the current values of
                the variables.

    Returns
    -------
    res: float, :py:class:`ExpressionBase`, :py:class:`ComponentMap`, or list
        The value or expression of the derivative(s)

    """

    if mode == Modes.reverse_numeric or mode == Modes.reverse_symbolic:
        if mode == Modes.reverse_numeric:
            res = reverse_ad(expr=expr)
        else:
            res = reverse_sd(expr=expr)

        if wrt is not None:
            if wrt_list is not None:
                raise ValueError(
                    'differentiate(): Cannot specify both wrt and wrt_list.')
            if wrt in res:
                res = res[wrt]
            else:
                res = 0
        elif wrt_list is not None:
            _res = list()
            for _wrt in wrt_list:
                if _wrt in res:
                    _res.append(res[_wrt])
                else:
                    _res.append(0)
            res = _res
    elif mode is Modes.sympy:
        res = sympy_diff(expr=expr, wrt=wrt, wrt_list=wrt_list)
    else:
        raise ValueError(
            'differentiate(): Unrecognized differentiation mode: {0}'.format(
                mode))

    return res


differentiate.Modes = Modes
