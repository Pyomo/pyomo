# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.contrib.piecewise.transform.piecewise_linear_transformation_base import (
    PiecewiseLinearTransformationBase
)
from pyomo.contrib.piecewise.transform.piecewise_to_mip_visitor import (
    PiecewiseLinearToMIP,
)
from pyomo.environ import (
    Constraint,
    NonNegativeIntegers,
    Var
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

        # Create variable to substitue
        substitute_var = transBlock.substitute_var = Var()
        pw_linear_func.map_transformation_var(pw_expr, substitute_var)

        transBlock.epigraphical_constraints = Constraint(NonNegativeIntegers)
        
        epigraph = False
        if pw_linear_func.convex is None:
            # we should autodetect if the dimension isn't insane, yell otherwise
            raise NotImplementedError("Convexity not specified")
        if pw_linear_func.convex:
            # This will be epigraph
            epigraph = True
        # Else this is the hypograph (the function is concave)

        linear_func_expr = linear_func(*pw_expr.args)
        for idx, linear_func in enumerate(pw_linear_func._linear_functions):
            if epigraph:
                transBlock.epigraphical_constraints[idx] = (
                    linear_func_expr <= substitute_var
                )
            else:
                transBlock.epigraphical_constraints[idx] = (
                    linear_func_expr >= substitute_var
                )

        return substitute_var
