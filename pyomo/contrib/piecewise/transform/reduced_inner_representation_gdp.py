#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
from pyomo.contrib.piecewise.transform.piecewise_to_gdp_transformation import (
    PiecewiseLinearToGDP,
)
from pyomo.core import Constraint, NonNegativeIntegers, Var
from pyomo.core.base import TransformationFactory
from pyomo.gdp import Disjunct, Disjunction


@TransformationFactory.register(
    'contrib.piecewise.reduced_inner_repn_gdp',
    doc="Convert piecewise-linear model to a GDP "
    "using an inner representation of the "
    "simplices that are the domains of the linear "
    "functions.",
)
class ReducedInnerRepresentationGDPTransformation(PiecewiseLinearToGDP):
    """
    Convert a model involving piecewise linear expressions into a GDP by
    representing the piecewise linear functions as Disjunctions where the
    simplices over which the linear functions are defined are represented
    in a reduced "inner" representation--as convex combinations of their extreme
    points. We refer to this as 'reduced' since we create only one multiplier
    for each extreme point in the union of the extreme points over all the
    simplices. Within the Disjuncts, we then enforce that all of the multipliers
    for extreme points not in the simplex are 0.

    This transformation can be called in one of two ways:
        1) The default, where 'descend_into_expressions' is False. This is
           more computationally efficient, but relies on the
           PiecewiseLinearFunctions being declared on the same Block in which
           they are used in Expressions (if you are hoping to maintain the
           original hierarchical structure of the model). In this mode,
           targets must be Blocks and/or PiecewiseLinearFunctions.
        2) With 'descend_into_expressions' True. This is less computationally
           efficient, but will respect hierarchical structure by finding
           uses of PiecewiseLinearFunctions in Constraint and Obective
           expressions and putting their transformed counterparts on the same
           parent Block as the component owning their parent expression. In
           this mode, targets must be Blocks, Constraints, and/or Objectives.
    """

    CONFIG = PiecewiseLinearToGDP.CONFIG()
    _transformation_name = 'pw_linear_reduced_inner_repn'

    def _transform_pw_linear_expr(self, pw_expr, pw_linear_func, transformation_block):
        transBlock = transformation_block.transformed_functions[
            len(transformation_block.transformed_functions)
        ]

        # get the PiecewiseLinearFunctionExpression
        dimension = pw_expr.nargs()
        transBlock.disjuncts = Disjunct(NonNegativeIntegers)
        substitute_var = transBlock.substitute_var = Var()
        pw_linear_func.map_transformation_var(pw_expr, substitute_var)
        substitute_var_lb = float('inf')
        substitute_var_ub = -float('inf')
        extreme_pts_by_simplex = {}
        linear_func_by_extreme_pt = {}
        # Save all the extreme points as sets since we will need to check set
        # containment to build the constraints fixing the multipliers to 0. We
        # can also build the data structure that will allow us to later build
        # the linear func expression
        for simplex, linear_func in zip(
            pw_linear_func._simplices, pw_linear_func._linear_functions
        ):
            extreme_pts = extreme_pts_by_simplex[simplex] = set()
            for idx in simplex:
                extreme_pts.add(idx)
                if idx not in linear_func_by_extreme_pt:
                    linear_func_by_extreme_pt[idx] = linear_func

            # We're going to want bounds on the substitute var, so we use
            # interval arithmetic to figure those out as we go.
            (lb, ub) = compute_bounds_on_expr(linear_func(*pw_expr.args))
            if lb is not None and lb < substitute_var_lb:
                substitute_var_lb = lb
            if ub is not None and ub > substitute_var_ub:
                substitute_var_ub = ub

        # set the bounds on the substitute var
        if substitute_var_lb < float('inf'):
            transBlock.substitute_var.setlb(substitute_var_lb)
        if substitute_var_ub > -float('inf'):
            transBlock.substitute_var.setub(substitute_var_ub)

        num_extreme_pts = len(pw_linear_func._points)
        # lambda[i] will be the multiplier for the extreme point with index i in
        # pw_linear_fun._points
        transBlock.lambdas = Var(range(num_extreme_pts), bounds=(0, 1))

        # Now that we have all of the extreme points, we can make the
        # disjunctive constraints
        for simplex in pw_linear_func._simplices:
            disj = transBlock.disjuncts[len(transBlock.disjuncts)]
            cons = disj.lambdas_zero_for_other_simplices = Constraint(
                NonNegativeIntegers
            )
            extreme_pts = extreme_pts_by_simplex[simplex]
            for i in range(num_extreme_pts):
                if i not in extreme_pts:
                    cons[len(cons)] = transBlock.lambdas[i] <= 0
        # Make the disjunction
        transBlock.pick_a_piece = Disjunction(
            expr=[d for d in transBlock.disjuncts.values()]
        )

        # Now we make the global constraints
        transBlock.convex_combo = Constraint(
            expr=sum(transBlock.lambdas[i] for i in range(num_extreme_pts)) == 1
        )
        transBlock.linear_func = Constraint(
            expr=sum(
                linear_func_by_extreme_pt[j](*pt) * transBlock.lambdas[j]
                for (j, pt) in enumerate(pw_linear_func._points)
            )
            == substitute_var
        )

        @transBlock.Constraint(range(dimension))
        def linear_combo(b, i):
            return pw_expr.args[i] == sum(
                pt[i] * transBlock.lambdas[j]
                for (j, pt) in enumerate(pw_linear_func._points)
            )

        return transBlock.substitute_var
