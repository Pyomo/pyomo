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
from pyomo.gdp import Disjunct, Disjunction


@TransformationFactory.register(
    'contrib.piecewise.inner_repn_gdp',
    doc="Convert piecewise-linear model to a GDP "
    "using an inner representation of the "
    "simplices that are the domains of the linear "
    "functions.",
)
class InnerRepresentationGDPTransformation(PiecewiseLinearTransformationBase):
    """
    Convert a model involving piecewise linear expressions into a GDP by
    representing the piecewise linear functions as Disjunctions where the
    simplices over which the linear functions are defined are represented
    in an "inner" representation--as convex combinations of their extreme
    points. The multipliers defining the convex combination are local to
    each Disjunct, so there is one per extreme point in each simplex.

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
    _transformation_name = 'pw_linear_inner_repn'

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
        for simplex, linear_func in zip(
            pw_linear_func._simplices, pw_linear_func._linear_functions
        ):
            disj = transBlock.disjuncts[len(transBlock.disjuncts)]
            disj.lambdas = Var(NonNegativeIntegers, dense=False, bounds=(0, 1))
            extreme_pts = []
            for idx in simplex:
                extreme_pts.append(pw_linear_func._points[idx])

            disj.convex_combo = Constraint(
                expr=sum(disj.lambdas[i] for i in range(len(extreme_pts))) == 1
            )
            linear_func_expr = linear_func(*pw_expr.args)
            disj.set_substitute = Constraint(expr=substitute_var == linear_func_expr)
            (lb, ub) = compute_bounds_on_expr(linear_func_expr)
            if lb is not None and lb < substitute_var_lb:
                substitute_var_lb = lb
            if ub is not None and ub > substitute_var_ub:
                substitute_var_ub = ub

            @disj.Constraint(range(dimension))
            def linear_combo(disj, i):
                return pw_expr.args[i] == sum(
                    disj.lambdas[j] * pt[i] for j, pt in enumerate(extreme_pts)
                )

            # Mark the lambdas as local so that we don't do anything silly in
            # the hull transformation.
            disj.LocalVars = Suffix(direction=Suffix.LOCAL)
            disj.LocalVars[disj] = [v for v in disj.lambdas.values()]

        if substitute_var_lb < float('inf'):
            transBlock.substitute_var.setlb(substitute_var_lb)
        if substitute_var_ub > -float('inf'):
            transBlock.substitute_var.setub(substitute_var_ub)
        transBlock.pick_a_piece = Disjunction(
            expr=[d for d in transBlock.disjuncts.values()]
        )

        return transBlock.substitute_var
