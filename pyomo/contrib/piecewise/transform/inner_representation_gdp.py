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

from pyomo.common.collections import ComponentMap
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.common.modeling import unique_component_name
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
from pyomo.contrib.piecewise import PiecewiseLinearFunction
from pyomo.contrib.piecewise.transform.piecewise_to_mip_visitor import (
    PiecewiseLinearToMIP)
from pyomo.core import (
    Constraint, Objective, Var, BooleanVar, Expression, Suffix, Param, Set,
    SetOf, RangeSet, ExternalFunction, Connector, SortComponents, Any,
    NonNegativeIntegers, NonNegativeReals)
from pyomo.core.base import Transformation, TransformationFactory
from pyomo.core.base.block import _BlockData, Block
from pyomo.core.util import target_list
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import is_child_of
from pyomo.network import Port

@TransformationFactory.register('contrib.inner_repn_gdp',
                                doc="Convert piecewise-linear model to a GDP "
                                "using an inner representation of the "
                                "simplices that are the domains of the linear "
                                "functions.")
class InnerRepresentationGDPTransformation(Transformation):
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
    CONFIG = ConfigDict('piecewise.inner_repn_gdp')
    CONFIG.declare('targets', ConfigValue(
        default=None,
        domain=target_list,
        description="target or list of targets that will be transformed",
        doc="""
        This specifies the list of components to transform. If None (default),
        the entire model is transformed. Note that if the transformation is
        done out of place, the list of targets should be attached to the model
        before it is cloned, and the list will specify the targets on the cloned
        instance."""
    ))
    CONFIG.declare('descend_into_expressions', ConfigValue(
        default=False,
        domain=bool,
        description="Whether to look for uses of PiecewiseLinearFunctions in "
        "the Constraint and Objective expressions, rather than assuming "
        "all PiecewiseLinearFunctions are on the active tree(s) of 'instance' "
        "and 'targets.'",
        doc="""
        It is *strongly* recommended that, in hierarchical models, the
        PiecewiseLinearFunction components are on the same Block as where
        they are used in expressions. If you follow this recommendation,
        this option can remain False, which will make this transformation
        more efficient. However, if you do not follow the recommendation,
        unless you know what you are doing, turn this option to 'True' to
        ensure that all of the uses of PiecewiseLinearFunctions are
        transformed.
        """
    ))
    def __init__(self):
        super().__init__()
        self.handlers = {
            Constraint: self._transform_constraint,
            Objective: self._transform_objective,
            Var:         False,
            BooleanVar:  False,
            Connector:   False,
            Expression:  False,
            Suffix:      False,
            Param:       False,
            Set:         False,
            SetOf:       False,
            RangeSet:    False,
            Disjunction: False,
            Disjunct:    False,
            Block:       self._transform_block,
            ExternalFunction: False,
            Port:        False,
            PiecewiseLinearFunction: self._transform_piecewise_linear_function,
        }
        self._transformation_blocks = {}

    def _apply_to(self, instance, **kwds):
        try:
            self._apply_to_impl(instance, **kwds)
        finally:
            self._transformation_blocks.clear()

    def _apply_to_impl(self, instance, **kwds):
        config = self.CONFIG(kwds.pop('options', {}))
        config.set_value(kwds)

        targets = config.targets
        if targets is None:
            targets = (instance, )

        knownBlocks = {}
        not_walking_exprs_msg = (
            "When not descending into expressions, Constraints "
            "and Objectives are not valid targets. Please specify "
            "PiecewiseLinearFunction component and the Blocks "
            "containing them, or (at the cost of some performance "
            "in this transformation), set the 'descend_into_expressions' "
            "option to 'True'.")
        for t in targets:
            if not is_child_of(parent=instance, child=t,
                               knownBlocks=knownBlocks):
                raise ValueError("Target '%s' is not a component on instance "
                                 "'%s'!" % (t.name, instance.name))
            if t.ctype is PiecewiseLinearFunction:
                if config.descend_into_expressions:
                    raise ValueError(
                        "When descending into expressions, the transformation "
                        "cannot take PiecewiseLinearFunction components as "
                        "targets. Please instead specify the Blocks, "
                        "Constraints, and Objectives where your "
                        "PiecewiseLinearFunctions have been used in "
                        "expressions.")
                self._transform_piecewise_linear_function(
                    t, config.descend_into_expressions)
            elif t.ctype is Block or isinstance(t, _BlockData):
                self._transform_block(t, config.descend_into_expressions)
            elif t.ctype is Constraint:
                if not config.descend_into_expressions:
                    raise ValueError(
                        "Encountered Constraint target '%s':\n%s"
                        % (t.name, not_walking_exprs_msg))
                self._transform_constraint(t, config.descend_into_expressions)
            elif t.ctype is Objective:
                if not config.descend_into_expressions:
                    raise ValueError(
                        "Encountered Objective target '%s':\n%s"
                        % (t.name, not_walking_exprs_msg))
                self._transform_objective(t, config.descend_into_expressions)
            else:
                raise ValueError(
                    "Target '%s' is not a PiecewiseLinearFunction, Block or "
                    "Constraint. It was of type '%s' and can't be transformed."
                    % (t.name, type(t)))

    def _get_transformation_block(self, parent):
        if parent in self._transformation_blocks:
            return self._transformation_blocks[parent]

        nm = unique_component_name(
            parent,
            '_pyomo_contrib_pw_linear_inner_repn')
        self._transformation_blocks[parent] = transBlock = Block()
        parent.add_component(nm, transBlock)

        transBlock.transformed_functions = Block(Any)
        return transBlock

    def _transform_block(self, block, descend_into_expressions):
        blocks = block.values() if block.is_indexed() else (block,)
        for b in blocks:
            for obj in b.component_objects(
                    active=True,
                    descend_into=(Block, Disjunct),
                    sort=SortComponents.deterministic):
                handler = self.handlers.get(obj.ctype, None)
                if not handler:
                    if handler is None:
                        raise RuntimeError(
                            "No transformation handler registered for modeling "
                            "components of type '%s'." % obj.ctype)
                    continue
                handler(obj, descend_into_expressions)

    def _transform_pw_linear_expr(self, pw_expr, pw_linear_func,
                                  transformation_block):
        transBlock = transformation_block.transformed_functions[
            len(transformation_block.transformed_functions)]

        # get the PiecewiseLinearFunctionExpression
        dimension = pw_expr.nargs()
        transBlock.disjuncts = Disjunct(NonNegativeIntegers)
        substitute_var = transBlock.substitute_var = Var()
        pw_linear_func.map_transformation_var(pw_expr,
                                              substitute_var)
        substitute_var_lb = float('inf')
        substitute_var_ub = -float('inf')
        for simplex, linear_func in zip(pw_linear_func._simplices,
                                        pw_linear_func._linear_functions):
            disj = transBlock.disjuncts[len(transBlock.disjuncts)]
            disj.lambdas = Var(NonNegativeIntegers, dense=False,
                               bounds=(0,1))
            extreme_pts = []
            for idx in simplex:
                extreme_pts.append(pw_linear_func._points[idx])

            disj.convex_combo = Constraint(
                expr=sum(disj.lambdas[i] for i in range(len(extreme_pts))) == 1)
            linear_func_expr = linear_func(*pw_expr.args)
            disj.set_substitute = Constraint(expr=substitute_var ==
                                             linear_func_expr)
            (lb, ub) = compute_bounds_on_expr(linear_func_expr)
            if lb is not None and lb < substitute_var_lb:
                substitute_var_lb = lb
            if ub is not None and ub > substitute_var_ub:
                substitute_var_ub = ub
            @disj.Constraint(range(dimension))
            def linear_combo(disj, i):
                return pw_expr.args[i] == sum(disj.lambdas[j]*pt[i] for j, pt in
                                              enumerate(extreme_pts))

            # Mark the lambdas as local so that we don't do anything silly in
            # the hull transformation.
            disj.LocalVars = Suffix(direction=Suffix.LOCAL)
            disj.LocalVars[disj] = [v for v in disj.lambdas.values()]

        if substitute_var_lb < float('inf'):
            transBlock.substitute_var.setlb(substitute_var_lb)
        if substitute_var_ub > -float('inf'):
            transBlock.substitute_var.setub(substitute_var_ub)
        transBlock.pick_a_piece = Disjunction(
            expr=[d for d in transBlock.disjuncts.values()])

        return transBlock.substitute_var

    def _transform_piecewise_linear_function(self, pw_linear_func,
                                             descend_into_expressions):
        if descend_into_expressions:
            return

        transBlock = self._get_transformation_block(
            pw_linear_func.parent_block())
        _functions = pw_linear_func.values() if pw_linear_func.is_indexed() \
                     else (pw_linear_func,)
        for pw_func in _functions:
            for pw_expr in pw_func._expressions.values():
                substitute_var = self._transform_pw_linear_expr(pw_expr.expr,
                                                                pw_func,
                                                                transBlock)
                # We change the named expression to point to the variable that
                # will take the appropriate value of the piecewise linear
                # function.
                pw_expr.expr = substitute_var

    def _transform_constraint(self, constraint, descend_into_expressions):
        if not descend_into_expressions:
            return

        transBlock = self._get_transformation_block(constraint.parent_block())
        visitor = PiecewiseLinearToMIP(self._transform_pw_linear_expr,
                                       transBlock)

        _constraints = constraint.values() if constraint.is_indexed() else \
                       (constraint,)
        for c in _constraints:
            visitor.walk_expression((c.expr, c, 0))

    def _transform_objective(self, objective, descend_into_expressions):
        if not descend_into_expressions:
            return

        transBlock = self._get_transformation_block(objective.parent_block())
        visitor = PiecewiseLinearToMIP(self._transform_pw_linear_expr,
                                       transBlock)

        _objectives = objective.values() if objective.is_indexed() else \
                      (objective,)
        for o in _objectives:
            visitor.walk_expression((o.expr, o, 0))
