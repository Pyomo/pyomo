# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.core.base import TransformationFactory
from pyomo.contrib.piecewise import FunctionType
from pyomo.contrib.piecewise.transform.piecewise_linear_transformation_base import (
    PiecewiseLinearTransformationBase,
)
from pyomo.contrib.piecewise.transform.piecewise_to_mip_visitor import (
    PiecewiseLinearToMIP,
)
from pyomo.environ import Constraint, NonNegativeIntegers, Var


@TransformationFactory.register(
    'contrib.piecewise.epigraph_hypograph',
    doc="Transforms convex/concave piecewise linear functions to their epigraphical/"
    "hypographical LP formulations",
)
class PWLToEpigraphOrHypograph(PiecewiseLinearTransformationBase):
    CONFIG = PiecewiseLinearTransformationBase.CONFIG()
    _transformation_name = 'pw_linear_epigraph'

    def _transform_pw_linear_expr(self, pw_expr, pw_linear_func, transformation_block):
        transBlock = transformation_block.transformed_functions[
            len(transformation_block.transformed_functions)
        ]
        # get the PiecewiseLinearFunctionExpression
        dimension = pw_expr.nargs()

        # Create variable to substitute
        substitute_var = transBlock.substitute_var = Var()
        pw_linear_func.map_transformation_var(pw_expr, substitute_var)

        transBlock.epigraphical_constraints = Constraint(NonNegativeIntegers)

        if pw_linear_func.function_type == FunctionType.CONVEX:
            # This will be epigraph
            epigraph = True
        # this is the hypograph (the function is concave)
        elif pw_linear_func.function_type == FunctionType.CONCAVE:
            epigraph = False
        elif pw_linear_func.function_type == FunctionType.UNSPECIFIED:
            # we should autodetect if the dimension isn't insane, yell otherwise
            raise NotImplementedError(
                "It is (quadratically, in the number of pieces) possible to "
                "auto-detect if a piecewise-linear "
                "function is convex or concave, but we don't currently have an "
                "implementation. PRs are welcome, or manually specify in the "
                "PiecewiseLinearFunction constructor the convexity/concavity "
                "using the 'convex' argument."
            )
        else:
            raise ValueError(
                f"Unrecognized value for function_type of piecewise-linear function "
                f"'{pw_linear_func.name}': {pw_linear_func.function_type}"
            )

        for idx, linear_func in enumerate(pw_linear_func._linear_functions):
            linear_func_expr = linear_func(*pw_expr.args)
            if epigraph:
                transBlock.epigraphical_constraints[idx] = (
                    linear_func_expr <= substitute_var
                )
            else:
                transBlock.epigraphical_constraints[idx] = (
                    linear_func_expr >= substitute_var
                )

        return substitute_var
