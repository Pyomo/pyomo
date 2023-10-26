from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
from pyomo.contrib.piecewise.transform.piecewise_to_gdp_transformation import (
    PiecewiseLinearToGDP,
)
from pyomo.core import Constraint, Binary, NonNegativeIntegers, Suffix, Var, RangeSet, Param
from pyomo.core.base import TransformationFactory
from pyomo.gdp import Disjunct, Disjunction
from pyomo.common.errors import DeveloperError
from pyomo.core.expr.visitor import SimpleExpressionVisitor
from pyomo.core.expr.current import identify_components
from math import ceil, log2


@TransformationFactory.register(
    "contrib.piecewise.incremental",
    doc="""
    TODO document
    """,
)
class IncrementalInnerGDPTransformation(PiecewiseLinearToGDP):
    """
    TODO document
    """

    CONFIG = PiecewiseLinearToGDP.CONFIG()
    _transformation_name = "pw_linear_incremental"

    # Implement to use PiecewiseLinearToGDP. This function returns the Var
    # that replaces the transformed piecewise linear expr
    def _transform_pw_linear_expr(self, pw_expr, pw_linear_func, transformation_block):
        self.DEBUG = False
        # Get a new Block() in transformation_block.transformed_functions, which
        # is a Block(Any)
        transBlock = transformation_block.transformed_functions[
            len(transformation_block.transformed_functions)
        ]

        # Dimensionality of the PWLF
        dimension = pw_expr.nargs()
        transBlock.dimension_indices = RangeSet(0, dimension - 1)

        # Substitute Var that will hold the value of the PWLE
        substitute_var = transBlock.substitute_var = Var()
        pw_linear_func.map_transformation_var(pw_expr, substitute_var)

        # Bounds for the substitute_var that we will widen
        self.substitute_var_lb = float("inf")
        self.substitute_var_ub = -float("inf")

        # Simplices are tuples of indices of points. Give them their own indices, too
        simplices = pw_linear_func._simplices
        num_simplices = len(simplices)
        transBlock.simplex_indices = RangeSet(0, num_simplices - 1)
        transBlock.simplex_indices_except_last = RangeSet(0, num_simplices - 2)
        # Assumption: the simplices are really simplices and all have the same number of points,
        # which is dimension + 1
        transBlock.simplex_point_indices = RangeSet(0, dimension)
        transBlock.nonzero_simplex_point_indices = RangeSet(1, dimension)
        transBlock.last_simplex_point_index = Param(initialize=dimension)


        # Ordering of simplices to follow Vielma
        # TODO: this enumeration must satisfy O1 (Vielma): each T_i \cap T_{i-1} is nonempty
        self.simplex_ordering = {
            n: n for n in transBlock.simplex_indices
        }

        # Enumeration of simplices: map from simplex number to correct simplex object
        self.idx_to_simplex = {
            n: simplices[m] for n, m in self.simplex_ordering.items()
        }
        # Associate simplex indices with correct linear functions
        self.idx_to_lin_func = {
            n: pw_linear_func._linear_functions[m] for n, m in self.simplex_ordering.items()
        }

        # For each individual simplex, the points need to be permuted in a way that
        # satisfies O1 and O2 (Vielma). TODO TODO TODO
        self.vertex_ordering = {
            (T, n): n
            for T in transBlock.simplex_indices
            for n in transBlock.simplex_point_indices
        }

        # Inital vertex (v_0^0 in Vielma)
        self.initial_vertex = pw_linear_func._points[self.idx_to_simplex[0][self.vertex_ordering[0, 0]]]

        # delta_i^j = delta[simplex][point]
        transBlock.delta = Var(
            transBlock.simplex_indices,
            transBlock.nonzero_simplex_point_indices,
            bounds=(0, 1),
        )
        transBlock.delta_one_constraint = Constraint(
            # figure out if this needs to be 0 or 1
            expr=sum(
                transBlock.delta[0, j] for j in transBlock.nonzero_simplex_point_indices
            )
            <= 1
        )
        # Set up the binary y_i variables, which interleave with the delta_i^j in
        # an odd way
        transBlock.y_binaries = Var(
            transBlock.simplex_indices_except_last,
            domain=Binary
        )

        # If the delta for the final point in simplex i is not one, y_i must be zero. That is,
        # y_i is one for and only for simplices that are completely "used"
        @transBlock.Constraint(transBlock.simplex_indices_except_last)
        def y_below_delta(m, i):
            return (transBlock.y_binaries[i] <= transBlock.delta[i, transBlock.last_simplex_point_index])
        
        # The sum of the deltas for simplex i+1 should be less than y_i. The overall
        # effect of these two constraints is that for simplices with y_i=1, the final
        # delta being one and others zero is enforced. For the first simplex with y_i=0,
        # the choice of deltas is free except that they must add to one. For following
        # simplices with y_i=0, all deltas are fixed at zero.
        @transBlock.Constraint(transBlock.simplex_indices_except_last)
        def deltas_below_y(m, i):
            return (sum(transBlock.delta[i + 1, j] for j in transBlock.nonzero_simplex_point_indices) <= transBlock.y_binaries[i])

        # Now we can relate the deltas and x. x is a sum along differences of points, 
        # weighted by deltas (12a.1)
        @transBlock.Constraint(transBlock.dimension_indices)
        def x_constraint(b, n):
            return (pw_expr.args[n] ==
                self.initial_vertex[n] + sum(
                    sum(
                        # delta_i^j * (v_i^j - v_i^0)
                        transBlock.delta[i, j] * (pw_linear_func._points[self.idx_to_simplex[i][self.vertex_ordering[i, j]]][n]
                                                - pw_linear_func._points[self.idx_to_simplex[i][self.vertex_ordering[i, 0]]][n])
                        for j in transBlock.nonzero_simplex_point_indices
                    )
                    for i in transBlock.simplex_indices
                )
            )

        # Now we can set the substitute Var for the PWLE (12a.2)
        transBlock.set_substitute = Constraint(
            expr=substitute_var
            == self.idx_to_lin_func[0](*self.initial_vertex) + sum(
                    sum(
                        # delta_i^j * (f(v_i^j) - f(v_i^0))
                        transBlock.delta[i, j] * (self.idx_to_lin_func[i](*pw_linear_func._points[self.idx_to_simplex[i][self.vertex_ordering[i, j]]])
                                                - self.idx_to_lin_func[i](*pw_linear_func._points[self.idx_to_simplex[i][self.vertex_ordering[i, 0]]]))
                        for j in transBlock.nonzero_simplex_point_indices
                    )
                    for i in transBlock.simplex_indices
                )
        )