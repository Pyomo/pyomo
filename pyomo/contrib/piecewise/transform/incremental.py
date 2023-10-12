from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
from pyomo.contrib.piecewise.transform.piecewise_to_gdp_transformation import (
    PiecewiseLinearToGDP,
)
from pyomo.core import Constraint, Binary, NonNegativeIntegers, Suffix, Var, RangeSet
from pyomo.core.base import TransformationFactory
from pyomo.gdp import Disjunct, Disjunction
from pyomo.common.errors import DeveloperError
from pyomo.core.expr.visitor import SimpleExpressionVisitor
from pyomo.core.expr.current import identify_components
from math import ceil, log2

@TransformationFactory.register(
    'contrib.piecewise.incremental',
    doc=
    """
    TODO document
    """,
)
class IncrementalInnerGDPTransformation(PiecewiseLinearToGDP):
    """
    TODO document
    """
    CONFIG = PiecewiseLinearToGDP.CONFIG()
    _transformation_name = 'pw_linear_incremental'
     
    # Implement to use PiecewiseLinearToGDP. This function returns the Var
    # that replaces the transformed piecewise linear expr
    def _transform_pw_linear_expr(self, pw_expr, pw_linear_func, transformation_block):
        self.DEBUG = False
        # Get a new Block() in transformation_block.transformed_functions, which
        # is a Block(Any)
        transBlock = transformation_block.transformed_functions[
            len(transformation_block.transformed_functions)
        ]

        dimension = pw_expr.nargs()
        substitute_var = transBlock.substitute_var = Var()
        pw_linear_func.map_transformation_var(pw_expr, substitute_var)
        self.substitute_var_lb = float('inf')
        self.substitute_var_ub = -float('inf')
