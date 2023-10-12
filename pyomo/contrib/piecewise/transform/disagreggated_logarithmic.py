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
    "contrib.piecewise.disaggregated_logarithmic",
    doc="""
    Represent a piecewise linear function "logarithmically" by using a MIP with
    log_2(|P|) binary decision variables. This method of logarithmically 
    formulating the piecewise linear function imposes no restrictions on the 
    family of polytopes. This method is due to Vielma et al., 2010.
    """,
)
class DisaggregatedLogarithmicInnerGDPTransformation(PiecewiseLinearToGDP):
    """
    Represent a piecewise linear function "logarithmically" by using a MIP with
    log_2(|P|) binary decision variables. This method of logarithmically
    formulating the piecewise linear function imposes no restrictions on the
    family of polytopes. This method is due to Vielma et al., 2010.
    """

    CONFIG = PiecewiseLinearToGDP.CONFIG()
    _transformation_name = "pw_linear_disaggregated_log"

    # Implement to use PiecewiseLinearToGDP. This function returns the Var
    # that replaces the transformed piecewise linear expr
    def _transform_pw_linear_expr(self, pw_expr, pw_linear_func, transformation_block):

        # Get a new Block() in transformation_block.transformed_functions, which
        # is a Block(Any). This is where we will put our new components.
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
        # Assumption: the simplices are really simplices and all have the same number of points,
        # which is dimension + 1
        transBlock.simplex_point_indices = RangeSet(0, dimension)

        # Enumeration of simplices, map from simplex number to simplex object
        self.idx_to_simplex = {k: v for k, v in zip(transBlock.simplex_indices, simplices)}
        # Inverse of previous enumeration
        self.simplex_to_idx = {v: k for k, v in self.idx_to_simplex.items()}

        # List of tuples of simplices with their linear function
        simplices_and_lin_funcs = list(zip(simplices, pw_linear_func._linear_functions))

        # We don't seem to get a convenient opportunity later, so let's just widen 
        # the bounds here. All we need to do is go through the corners of each simplex.
        for P, linear_func in simplices_and_lin_funcs:
            for v in transBlock.simplex_point_indices:
                val = linear_func(*pw_linear_func._points[P[v]])
                if val < self.substitute_var_lb:
                    self.substitute_var_lb = val
                if val > self.substitute_var_ub:
                    self.substitute_var_ub = val
        # Now set those bounds
        if self.substitute_var_lb < float('inf'):
            transBlock.substitute_var.setlb(self.substitute_var_lb)
        if self.substitute_var_ub > -float('inf'):
            transBlock.substitute_var.setub(self.substitute_var_ub)

        log_dimension = ceil(log2(num_simplices))
        transBlock.log_simplex_indices = RangeSet(0, log_dimension - 1)
        binaries = transBlock.binaries = Var(transBlock.log_simplex_indices, domain=Binary)

        # Injective function B: \mathcal{P} -> {0,1}^ceil(log_2(|P|)) used to identify simplices
        # (really just polytopes are required) with binary vectors. Any injective function
        # is enough here.
        B = {}
        for i in transBlock.simplex_indices:
            # map index(P) -> corresponding vector in {0, 1}^n
            B[i] = self._get_binary_vector(i, log_dimension)

        # The lambda variables \lambda_{P,v} are indexed by the simplex and the point in it
        transBlock.lambdas = Var(transBlock.simplex_indices, transBlock.simplex_point_indices, bounds=(0, 1))

        # Sum of all lambdas is one (6b)
        transBlock.convex_combo = Constraint(
            expr=sum(
                transBlock.lambdas[P, v]
                for P in transBlock.simplex_indices
                for v in transBlock.simplex_point_indices
            )
            == 1
        )

        # The branching rules, establishing using the binaries that only one simplex's lambdas
        # may be nonzero
        @transBlock.Constraint(transBlock.log_simplex_indices)  # (6c.1)
        def simplex_choice_1(b, l):
            return (
                sum(
                    transBlock.lambdas[self.simplex_to_idx[P], v]
                    for P in self._P_plus(B, l, simplices)
                    for v in transBlock.simplex_point_indices
                )
                <= binaries[l]
            )

        @transBlock.Constraint(transBlock.log_simplex_indices)  # (6c.2)
        def simplex_choice_2(b, l):
            return (
                sum(
                    transBlock.lambdas[self.simplex_to_idx[P], v]
                    for P in self._P_0(B, l, simplices)
                    for v in transBlock.simplex_point_indices
                )
                <= 1 - binaries[l]
            )

        # for i, (simplex, pwlf) in enumerate(choices):
        # x_i = sum(lambda_P,v v_i, P in polytopes, v in V(P))
        @transBlock.Constraint(transBlock.dimension_indices)  # (6a.1)
        def x_constraint(b, i):
            return pw_expr.args[i] == sum(
                transBlock.lambdas[self.simplex_to_idx[P], v]
                * pw_linear_func._points[P[v]][i]
                for P in simplices
                for v in transBlock.simplex_point_indices
            )

        # Make the substitute Var equal the PWLE (6a.2)
        #for P, linear_func in simplices_and_lin_funcs:
        #    print(f"P, linear_func = {P}, {linear_func}")
        #    for v in transBlock.simplex_point_indices:
        #        print(f"    v={v}")
        #        print(f"    pt={pw_linear_func._points[P[v]]}")
        #        print(
        #            f"    lin_func_val = {linear_func(*pw_linear_func._points[P[v]])}"
        #        )
        transBlock.set_substitute = Constraint(
            expr=substitute_var
            == sum(
                sum(
                    transBlock.lambdas[self.simplex_to_idx[P], v]
                    * linear_func(*pw_linear_func._points[P[v]])
                    for v in transBlock.simplex_point_indices
                )
                for (P, linear_func) in simplices_and_lin_funcs
            )
        )

        return substitute_var

    # Not a gray code, just a regular binary representation
    # TODO this may not be optimal, test the gray codes too
    def _get_binary_vector(self, num, length):
        if num != 0 and ceil(log2(num)) > length:
            raise DeveloperError("Invalid input in _get_binary_vector")
        # Hack: use python's string formatting instead of bothering with modular
        # arithmetic. May be slow.
        return tuple(int(x) for x in format(num, f"0{length}b"))

    # Return {P \in \mathcal{P} | B(P)_l = 0}
    def _P_0(self, B, l, simplices):
        return [p for p in simplices if B[self.simplex_to_idx[p]][l] == 0]

    # Return {P \in \mathcal{P} | B(P)_l = 1}
    def _P_plus(self, B, l, simplices):
        return [p for p in simplices if B[self.simplex_to_idx[p]][l] == 1]
