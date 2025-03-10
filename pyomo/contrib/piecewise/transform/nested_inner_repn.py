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

from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
from pyomo.contrib.piecewise.transform.piecewise_linear_transformation_base import (
    PiecewiseLinearTransformationBase,
)
from pyomo.core import Constraint, NonNegativeIntegers, Suffix, Var
from pyomo.core.base import TransformationFactory
from pyomo.gdp import Disjunction
from pyomo.common.errors import DeveloperError


@TransformationFactory.register(
    "contrib.piecewise.nested_inner_repn_gdp",
    doc="""
    Represent a piecewise linear function by using a nested GDP to determine
    which polytope a point is in, then representing it as a convex combination
    of extreme points, with multipliers "local" to that particular polytope,
    i.e., not shared with neighbors. This formulation has linearly many Boolean
    variables, though up to variable substitution, it has logarithmically many.
    """,
)
class NestedInnerRepresentationGDPTransformation(PiecewiseLinearTransformationBase):
    """
    Represent a piecewise linear function by using a nested GDP to determine
    which polytope a point is in, then representing it as a convex combination
    of extreme points, with multipliers "local" to that particular polytope,
    i.e., not shared with neighbors. This method of formulating the piecewise
    linear function imposes no restrictions on the family of polytopes. Note
    that this is NOT a logarithmic formulation - it has linearly many Boolean
    variables.  However, it is inspired by the disaggregated logarithmic
    formulation of [1]. Up to variable substitution, the amount of Boolean
    variables is logarithmic, as in [1].

    References
    ----------
    [1] J.P. Vielma, S. Ahmed, and G. Nemhauser, "Mixed-integer models
        for nonseparable piecewise-linear optimization: unifying framework
        and extensions," Operations Research, vol. 58, no. 2, pp. 305-315,
        2010.
    """

    CONFIG = PiecewiseLinearTransformationBase.CONFIG()
    _transformation_name = "pw_linear_nested_inner_repn"

    # Implement to use PiecewiseLinearTransformationBase. This function returns the Var
    # that replaces the transformed piecewise linear expr
    def _transform_pw_linear_expr(self, pw_expr, pw_linear_func, transformation_block):
        # Get a new Block() in transformation_block.transformed_functions, which
        # is a Block(Any)
        transBlock = transformation_block.transformed_functions[
            len(transformation_block.transformed_functions)
        ]

        substitute_var = transBlock.substitute_var = Var()
        pw_linear_func.map_transformation_var(pw_expr, substitute_var)
        transBlock.substitute_var_lb = float("inf")
        transBlock.substitute_var_ub = -float("inf")

        choices = list(zip(pw_linear_func._simplices, pw_linear_func._linear_functions))

        # If there was only one choice, don't bother making a disjunction, just
        # use the linear function directly (but still use the substitute_var for
        # consistency).
        if len(choices) == 1:
            (_, linear_func) = choices[0]  # simplex isn't important in this case
            linear_func_expr = linear_func(*pw_expr.args)
            transBlock.set_substitute = Constraint(
                expr=substitute_var == linear_func_expr
            )
            (transBlock.substitute_var_lb, transBlock.substitute_var_ub) = (
                compute_bounds_on_expr(linear_func_expr)
            )
        else:
            # Add the disjunction
            transBlock.disj = self._get_disjunction(
                choices, transBlock, pw_expr, pw_linear_func, transBlock
            )

        # Set bounds as determined when setting up the disjunction
        if transBlock.substitute_var_lb < float("inf"):
            transBlock.substitute_var.setlb(transBlock.substitute_var_lb)
        if transBlock.substitute_var_ub > -float("inf"):
            transBlock.substitute_var.setub(transBlock.substitute_var_ub)

        return substitute_var

    # Recursively form the Disjunctions and Disjuncts. This shouldn't blow up
    # the stack, since the whole point is that we'll only go logarithmically
    # many calls deep.
    def _get_disjunction(
        self, choices, parent_block, pw_expr, pw_linear_func, root_block
    ):
        size = len(choices)

        # Our base cases will be 3 and 2, since it would be silly to construct
        # a Disjunction containing only one Disjunct. We can ensure that size
        # is never 1 unless it was only passed a single choice from the start,
        # which we can handle before calling.
        if size > 3:
            half = size // 2  # (integer divide)
            # This tree will be slightly heavier on the right side
            choices_l = choices[:half]
            choices_r = choices[half:]

            @parent_block.Disjunct()
            def d_l(b):
                b.inner_disjunction_l = self._get_disjunction(
                    choices_l, b, pw_expr, pw_linear_func, root_block
                )

            @parent_block.Disjunct()
            def d_r(b):
                b.inner_disjunction_r = self._get_disjunction(
                    choices_r, b, pw_expr, pw_linear_func, root_block
                )

            return Disjunction(expr=[parent_block.d_l, parent_block.d_r])
        elif size == 3:
            # Let's stay heavier on the right side for consistency. So the left
            # Disjunct will be the one to contain constraints, rather than a
            # Disjunction
            @parent_block.Disjunct()
            def d_l(b):
                simplex, linear_func = choices[0]
                self._set_disjunct_block_constraints(
                    b, simplex, linear_func, pw_expr, pw_linear_func, root_block
                )

            @parent_block.Disjunct()
            def d_r(b):
                b.inner_disjunction_r = self._get_disjunction(
                    choices[1:], b, pw_expr, pw_linear_func, root_block
                )

            return Disjunction(expr=[parent_block.d_l, parent_block.d_r])
        elif size == 2:
            # In this case both sides are regular Disjuncts
            @parent_block.Disjunct()
            def d_l(b):
                simplex, linear_func = choices[0]
                self._set_disjunct_block_constraints(
                    b, simplex, linear_func, pw_expr, pw_linear_func, root_block
                )

            @parent_block.Disjunct()
            def d_r(b):
                simplex, linear_func = choices[1]
                self._set_disjunct_block_constraints(
                    b, simplex, linear_func, pw_expr, pw_linear_func, root_block
                )

            return Disjunction(expr=[parent_block.d_l, parent_block.d_r])
        else:
            raise DeveloperError(
                "Unreachable: 1 or 0 choices were passed to "
                "_get_disjunction in nested_inner_repn.py."
            )

    def _set_disjunct_block_constraints(
        self, b, simplex, linear_func, pw_expr, pw_linear_func, root_block
    ):
        # Define the lambdas sparsely like in the normal inner repn,
        # only the first few will participate in constraints
        b.lambdas = Var(NonNegativeIntegers, dense=False, bounds=(0, 1))

        # Get the extreme points to add up
        extreme_pts = []
        for idx in simplex:
            extreme_pts.append(pw_linear_func._points[idx])

        # Constrain sum(lambda_i) = 1
        b.convex_combo = Constraint(
            expr=sum(b.lambdas[i] for i in range(len(extreme_pts))) == 1
        )
        linear_func_expr = linear_func(*pw_expr.args)

        # Make the substitute Var equal the PWLE
        b.set_substitute = Constraint(
            expr=root_block.substitute_var == linear_func_expr
        )

        # Widen the variable bounds to those of this linear func expression
        (lb, ub) = compute_bounds_on_expr(linear_func_expr)
        if lb is not None and lb < root_block.substitute_var_lb:
            root_block.substitute_var_lb = lb
        if ub is not None and ub > root_block.substitute_var_ub:
            root_block.substitute_var_ub = ub

        # Constrain x = \sum \lambda_i v_i
        @b.Constraint(range(pw_expr.nargs()))  # dimension
        def linear_combo(d, i):
            return pw_expr.args[i] == sum(
                d.lambdas[j] * pt[i] for j, pt in enumerate(extreme_pts)
            )

        # Mark the lambdas as local in order to prevent disagreggating multiple
        # times in the hull transformation
        b.LocalVars = Suffix(direction=Suffix.LOCAL)
        b.LocalVars[b] = [v for v in b.lambdas.values()]
