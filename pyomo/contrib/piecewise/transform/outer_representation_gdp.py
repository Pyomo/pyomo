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

import pyomo.common.dependencies.numpy as np
from pyomo.common.dependencies.scipy import spatial
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
from pyomo.contrib.piecewise.transform.piecewise_linear_transformation_base import (
    PiecewiseLinearTransformationBase,
)
from pyomo.core import Constraint, NonNegativeIntegers, Suffix, Var
from pyomo.core.base import TransformationFactory
from pyomo.gdp import Disjunct, Disjunction


@TransformationFactory.register(
    'contrib.piecewise.outer_repn_gdp',
    doc="Convert piecewise-linear model to a GDP "
    "using an outer (Ax <= b) representation of "
    "the simplices that are the domains of the "
    "linear functions.",
)
class OuterRepresentationGDPTransformation(PiecewiseLinearTransformationBase):
    """
    Convert a model involving piecewise linear expressions into a GDP by
    representing the piecewise linear functions as Disjunctions where the
    simplices over which the linear functions are defined are represented
    in an "outer" representation--in sets of constraints of the form Ax <= b.

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

    CONFIG = PiecewiseLinearTransformationBase.CONFIG()
    _transformation_name = 'pw_linear_outer_repn'

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
        if dimension > 1:
            A = np.ones((dimension + 1, dimension + 1))
            b = np.zeros(dimension + 1)
            b[-1] = 1

        for simplex, linear_func in zip(
            pw_linear_func._simplices, pw_linear_func._linear_functions
        ):
            disj = transBlock.disjuncts[len(transBlock.disjuncts)]

            if dimension == 1:
                # We don't need scipy, and the polytopes are 1-dimensional
                # simplices, so they are defined by two bounds constraints:
                disj.simplex_halfspaces = Constraint(
                    expr=(
                        pw_linear_func._points[simplex[0]][0],
                        pw_expr.args[0],
                        pw_linear_func._points[simplex[1]][0],
                    )
                )
            else:
                disj.simplex_halfspaces = Constraint(range(dimension + 1))
                # we will use scipy to get the convex hull of the extreme
                # points of the simplex
                extreme_pts = []
                for idx in simplex:
                    extreme_pts.append(pw_linear_func._points[idx])
                chull = spatial.ConvexHull(extreme_pts)
                vars = pw_expr.args
                for i, eqn in enumerate(chull.equations):
                    # The equations are given as normal vectors (A) followed by
                    # offsets (b) such that Ax + b <= 0 gives the halfspaces
                    # defining the simplex. (See Qhull documentation)
                    disj.simplex_halfspaces[i] = (
                        sum(eqn[j] * v for j, v in enumerate(vars))
                        + float(eqn[dimension])
                        <= 0
                    )

            linear_func_expr = linear_func(*pw_expr.args)
            disj.set_substitute = Constraint(expr=substitute_var == linear_func_expr)
            (lb, ub) = compute_bounds_on_expr(linear_func_expr)
            if lb is not None and lb < substitute_var_lb:
                substitute_var_lb = lb
            if ub is not None and ub > substitute_var_ub:
                substitute_var_ub = ub

        if substitute_var_lb < float('inf'):
            transBlock.substitute_var.setlb(substitute_var_lb)
        if substitute_var_ub > -float('inf'):
            transBlock.substitute_var.setub(substitute_var_ub)
        transBlock.pick_a_piece = Disjunction(
            expr=[d for d in transBlock.disjuncts.values()]
        )

        return transBlock.substitute_var
