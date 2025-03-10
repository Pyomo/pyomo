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
from pyomo.core import Constraint, Binary, Var, RangeSet, Set
from pyomo.core.base import TransformationFactory
from pyomo.common.errors import DeveloperError
from math import ceil, log2


@TransformationFactory.register(
    "contrib.piecewise.disaggregated_logarithmic",
    doc="""
    Represent a piecewise linear function "logarithmically" by using a MIP with
    log_2(|P|) binary decision variables. This is a direct-to-MIP transformation;
    GDP is not used.
    """,
)
class DisaggregatedLogarithmicMIPTransformation(PiecewiseLinearTransformationBase):
    """Represent a piecewise linear function "logarithmically" as a MIP.

    This transformation represents a piecewise linear function
    "logarithmically" by using a MIP with :math:`log_2(|P|)` binary
    decision variables, following the "disaggregated logarithmic" method
    from [VAN10]_.

    This is a direct-to-MIP transformation; GDP is not used.  This
    method of logarithmically formulating the piecewise linear function
    imposes no restrictions on the family of polytopes, but we assume we
    have simplices in this code.

    """

    CONFIG = PiecewiseLinearTransformationBase.CONFIG()
    _transformation_name = "pw_linear_disaggregated_log"

    # Implement to use PiecewiseLinearTransformationBase. This function returns the Var
    # that replaces the transformed piecewise linear expr
    def _transform_pw_linear_expr(self, pw_expr, pw_linear_func, transformation_block):
        # Get a new Block for our transformation in transformation_block.transformed_functions,
        # which is a Block(Any). This is where we will put our new components.
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
        # Assumption: the simplices are really full-dimensional simplices and all have the
        # same number of points, which is dimension + 1
        transBlock.simplex_point_indices = RangeSet(0, dimension)

        # Enumeration of simplices: map from simplex number to simplex object
        idx_to_simplex = {k: v for k, v in zip(transBlock.simplex_indices, simplices)}

        # List of tuples of simplex indices with their linear function
        simplex_indices_and_lin_funcs = list(
            zip(transBlock.simplex_indices, pw_linear_func._linear_functions)
        )

        # We don't seem to get a convenient opportunity later, so let's just widen
        # the bounds here. All we need to do is go through the corners of each simplex.
        for P, linear_func in simplex_indices_and_lin_funcs:
            for v in transBlock.simplex_point_indices:
                val = linear_func(*pw_linear_func._points[idx_to_simplex[P][v]])
                if val < substitute_var_lb:
                    substitute_var_lb = val
                if val > substitute_var_ub:
                    substitute_var_ub = val
        transBlock.substitute_var.setlb(substitute_var_lb)
        transBlock.substitute_var.setub(substitute_var_ub)

        log_dimension = ceil(log2(num_simplices))
        transBlock.log_simplex_indices = RangeSet(0, log_dimension - 1)
        transBlock.binaries = Var(transBlock.log_simplex_indices, domain=Binary)

        # Injective function B: \mathcal{P} -> {0,1}^ceil(log_2(|P|)) used to identify simplices
        # (really just polytopes are required) with binary vectors. Any injective function
        # is enough here.
        B = {}
        for i in transBlock.simplex_indices:
            # map index(P) -> corresponding vector in {0, 1}^n
            B[i] = self._get_binary_vector(i, log_dimension)

        # Build up P_0 and P_plus ahead of time.

        # {P \in \mathcal{P} | B(P)_l = 0}
        def P_0_init(m, l):
            return [p for p in transBlock.simplex_indices if B[p][l] == 0]

        transBlock.P_0 = Set(transBlock.log_simplex_indices, initialize=P_0_init)

        # {P \in \mathcal{P} | B(P)_l = 1}
        def P_plus_init(m, l):
            return [p for p in transBlock.simplex_indices if B[p][l] == 1]

        transBlock.P_plus = Set(transBlock.log_simplex_indices, initialize=P_plus_init)

        # The lambda variables \lambda_{P,v} are indexed by the simplex and the point in it
        transBlock.lambdas = Var(
            transBlock.simplex_indices, transBlock.simplex_point_indices, bounds=(0, 1)
        )

        # Numbered citations are from Vielma et al 2010, Mixed-Integer Models
        # for Nonseparable Piecewise-Linear Optimization

        # Sum of all lambdas is one (6b)
        transBlock.convex_combo = Constraint(
            expr=sum(
                transBlock.lambdas[P, v]
                for P in transBlock.simplex_indices
                for v in transBlock.simplex_point_indices
            )
            == 1
        )

        # The branching rules, establishing using the binaries that only one simplex's lambda
        # coefficients may be nonzero
        # Enabling lambdas when binaries are on
        @transBlock.Constraint(transBlock.log_simplex_indices)  # (6c.1)
        def simplex_choice_1(b, l):
            return (
                sum(
                    transBlock.lambdas[P, v]
                    for P in transBlock.P_plus[l]
                    for v in transBlock.simplex_point_indices
                )
                <= transBlock.binaries[l]
            )

        # Disabling lambdas when binaries are on
        @transBlock.Constraint(transBlock.log_simplex_indices)  # (6c.2)
        def simplex_choice_2(b, l):
            return (
                sum(
                    transBlock.lambdas[P, v]
                    for P in transBlock.P_0[l]
                    for v in transBlock.simplex_point_indices
                )
                <= 1 - transBlock.binaries[l]
            )

        # for i, (simplex, pwlf) in enumerate(choices):
        # x_i = sum(lambda_P,v v_i, P in polytopes, v in V(P))
        @transBlock.Constraint(transBlock.dimension_indices)  # (6a.1)
        def x_constraint(b, i):
            return pw_expr.args[i] == sum(
                transBlock.lambdas[P, v]
                * pw_linear_func._points[idx_to_simplex[P][v]][i]
                for P in transBlock.simplex_indices
                for v in transBlock.simplex_point_indices
            )

        # Make the substitute Var equal the PWLE (6a.2)
        transBlock.set_substitute = Constraint(
            expr=substitute_var
            == sum(
                transBlock.lambdas[P, v]
                * linear_func(*pw_linear_func._points[idx_to_simplex[P][v]])
                for v in transBlock.simplex_point_indices
                for (P, linear_func) in simplex_indices_and_lin_funcs
            )
        )

        return substitute_var

    # Not a Gray code, just a regular binary representation
    # TODO test the Gray codes too
    # note: Must have num != 0 and ceil(log2(num)) > length to be valid
    def _get_binary_vector(self, num, length):
        ans = []
        for i in range(length):
            ans.append(num & 1)
            num >>= 1
        assert not num
        ans.reverse()
        return tuple(ans)
