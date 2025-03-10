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

from pyomo.contrib.piecewise.transform.piecewise_linear_transformation_base import (
    PiecewiseLinearTransformationBase,
)
from pyomo.contrib.piecewise.triangulations import Triangulation
from pyomo.core import Constraint, Binary, Var, RangeSet, Param
from pyomo.core.base import TransformationFactory


@TransformationFactory.register(
    "contrib.piecewise.incremental",
    doc="""
    The incremental MIP formulation of a piecewise-linear function, as described
    by [1]. To work in the multivariate case, the underlying triangulation must
    satisfy these properties:
     (1) The simplices are ordered T_1, ..., T_N such that T_i has nonempty intersection
         with T_{i+1}. It doesn't have to be a whole face; just a vertex is enough.
     (2) On each simplex T_i, the vertices are ordered T_i^1, ..., T_i^n such
         that T_i^n = T_{i+1}^1
    In Pyomo, the Triangulation.OrderedJ1 triangulation is compatible with this
    transformation.

    References
    ----------
    [1] J.P. Vielma, S. Ahmed, and G. Nemhauser, "Mixed-integer models
        for nonseparable piecewise-linear optimization: unifying framework
        and extensions," Operations Research, vol. 58, no. 2, pp. 305-315,
        2010.
    """,
)
class IncrementalMIPTransformation(PiecewiseLinearTransformationBase):

    CONFIG = PiecewiseLinearTransformationBase.CONFIG()
    _transformation_name = "pw_linear_incremental"

    # Implement to use PiecewiseLinearTransformationBase. This function returns the Var
    # that replaces the transformed piecewise linear expr
    def _transform_pw_linear_expr(self, pw_expr, pw_linear_func, transformation_block):
        if pw_linear_func.triangulation not in (
            Triangulation.OrderedJ1,
            Triangulation.AssumeValid,
        ):
            # almost certain not to work
            raise ValueError(
                "Incremental transformation specified, but the triangulation "
                f"{pw_linear_func.triangulation} may not be appropriately ordered. This "
                "would likely lead to incorrect results! The built-in "
                "Triangulation.OrderedJ1 triangulation has an appropriate ordering for "
                "this transformation. If you know what you are doing, you can also "
                "suppress this error by setting the triangulation tag to "
                "Triangulation.AssumeValid during PiecewiseLinearFunction construction."
            )
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
        substitute_var_lb = float("inf")
        substitute_var_ub = -float("inf")

        # Simplices are tuples of indices of points. Give them their own indices, too
        simplices = pw_linear_func._simplices
        num_simplices = len(simplices)
        transBlock.simplex_indices = RangeSet(0, num_simplices - 1)
        transBlock.simplex_indices_except_last = RangeSet(0, num_simplices - 2)
        # Assumption: the simplices are really simplices and all have the same number of
        # points, which is dimension + 1
        transBlock.simplex_point_indices = RangeSet(0, dimension)
        transBlock.nonzero_simplex_point_indices = RangeSet(1, dimension)
        transBlock.last_simplex_point_index = Param(initialize=dimension)

        # We don't seem to get a convenient opportunity later, so let's just widen
        # the bounds here. All we need to do is go through the corners of each simplex.
        for P, linear_func in zip(
            transBlock.simplex_indices, pw_linear_func._linear_functions
        ):
            for v in transBlock.simplex_point_indices:
                val = linear_func(*pw_linear_func._points[simplices[P][v]])
                if val < substitute_var_lb:
                    substitute_var_lb = val
                if val > substitute_var_ub:
                    substitute_var_ub = val
        # Now set those bounds
        transBlock.substitute_var.setlb(substitute_var_lb)
        transBlock.substitute_var.setub(substitute_var_ub)

        # Initial vertex (v_0^0 in Vielma)
        initial_vertex = pw_linear_func._points[simplices[0][0]]

        # delta_i^j = delta[simplex][point]
        transBlock.delta = Var(
            transBlock.simplex_indices,
            transBlock.nonzero_simplex_point_indices,
            bounds=(0, 1),
        )
        transBlock.delta_one_constraint = Constraint(
            # 0 for for us because we are indexing from zero here (12b.1)
            expr=sum(
                transBlock.delta[0, j] for j in transBlock.nonzero_simplex_point_indices
            )
            <= 1
        )
        # Set up the binary y_i variables, which interleave with the delta_i^j in
        # an odd way
        transBlock.y_binaries = Var(
            transBlock.simplex_indices_except_last, domain=Binary
        )

        # If the delta for the final point in simplex i is not one, y_i must be zero.
        # That is, y_i is one for and only for simplices that are completely "used"
        @transBlock.Constraint(transBlock.simplex_indices_except_last)
        def y_below_delta(m, i):
            return (
                transBlock.y_binaries[i]
                <= transBlock.delta[i, transBlock.last_simplex_point_index]
            )

        # The sum of the deltas for simplex i+1 should be less than y_i. The overall
        # effect of these two constraints is that for simplices with y_i=1, the final
        # delta being one and others zero is enforced. For the first simplex with y_i=0,
        # the choice of deltas is free except that they must add to one. For following
        # simplices with y_i=0, all deltas are fixed at zero.
        @transBlock.Constraint(transBlock.simplex_indices_except_last)
        def deltas_below_y(m, i):
            return (
                sum(
                    transBlock.delta[i + 1, j]
                    for j in transBlock.nonzero_simplex_point_indices
                )
                <= transBlock.y_binaries[i]
            )

        # Now we can relate the deltas and x. x is a sum along differences of points,
        # weighted by deltas (12a.1)
        @transBlock.Constraint(transBlock.dimension_indices)
        def x_constraint(b, n):
            return pw_expr.args[n] == initial_vertex[n] + sum(
                # delta_i^j * (v_i^j - v_i^0)
                transBlock.delta[i, j]
                * (
                    pw_linear_func._points[simplices[i][j]][n]
                    - pw_linear_func._points[simplices[i][0]][n]
                )
                for j in transBlock.nonzero_simplex_point_indices
                for i in transBlock.simplex_indices
            )

        # Now we can set the substitute Var for the PWLE (12a.2)
        transBlock.set_substitute = Constraint(
            expr=substitute_var
            == pw_linear_func._linear_functions[0](*initial_vertex)
            + sum(
                # delta_i^j * (f(v_i^j) - f(v_i^0))
                transBlock.delta[i, j]
                * (
                    pw_linear_func._linear_functions[i](
                        *pw_linear_func._points[simplices[i][j]]
                    )
                    - pw_linear_func._linear_functions[i](
                        *pw_linear_func._points[simplices[i][0]]
                    )
                )
                for j in transBlock.nonzero_simplex_point_indices
                for i in transBlock.simplex_indices
            )
        )

        return substitute_var
