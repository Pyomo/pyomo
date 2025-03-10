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

from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.common.errors import DeveloperError
from pyomo.common.modeling import unique_component_name
from pyomo.contrib.piecewise import PiecewiseLinearFunction
from pyomo.contrib.piecewise.transform.piecewise_to_mip_visitor import (
    PiecewiseLinearToMIP,
)
from pyomo.core import (
    Constraint,
    Objective,
    Var,
    BooleanVar,
    Expression,
    Suffix,
    Param,
    Set,
    SetOf,
    RangeSet,
    ExternalFunction,
    Connector,
    SortComponents,
    Any,
    LogicalConstraint,
)
from pyomo.core.base import Transformation
from pyomo.core.base.block import Block
from pyomo.core.util import target_list
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import is_child_of
from pyomo.network import Port


class PiecewiseLinearTransformationBase(Transformation):
    """
    Base class for transformations of piecewise-linear models to GDPs, MIPs, etc.
    """

    CONFIG = ConfigDict('contrib.piecewise_linear_transformation_base')
    CONFIG.declare(
        'targets',
        ConfigValue(
            default=None,
            domain=target_list,
            description="target or list of targets that will be transformed",
            doc="""
            This specifies the list of components to transform. If None (default),
            the entire model is transformed. Note that if the transformation is
            done out of place, the list of targets should be attached to the model
            before it is cloned, and the list will specify the targets on the cloned
            instance.""",
        ),
    )
    CONFIG.declare(
        'descend_into_expressions',
        ConfigValue(
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
            """,
        ),
    )

    def __init__(self):
        super().__init__()
        self.handlers = {
            Constraint: self._transform_constraint,
            Objective: self._transform_objective,
            Var: False,
            BooleanVar: False,
            Connector: False,
            Expression: False,
            Suffix: False,
            Param: False,
            Set: False,
            SetOf: False,
            RangeSet: False,
            Disjunction: False,
            Disjunct: False,
            Block: self._transform_block,
            ExternalFunction: False,
            Port: False,
            PiecewiseLinearFunction: self._transform_piecewise_linear_function,
            LogicalConstraint: False,
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
            targets = (instance,)

        knownBlocks = {}
        not_walking_exprs_msg = (
            "When not descending into expressions, Constraints "
            "and Objectives are not valid targets. Please specify "
            "PiecewiseLinearFunction component and the Blocks "
            "containing them, or (at the cost of some performance "
            "in this transformation), set the 'descend_into_expressions' "
            "option to 'True'."
        )
        for t in targets:
            if not is_child_of(parent=instance, child=t, knownBlocks=knownBlocks):
                raise ValueError(
                    "Target '%s' is not a component on instance "
                    "'%s'!" % (t.name, instance.name)
                )
            if t.ctype is PiecewiseLinearFunction:
                if config.descend_into_expressions:
                    raise ValueError(
                        "When descending into expressions, the transformation "
                        "cannot take PiecewiseLinearFunction components as "
                        "targets. Please instead specify the Blocks, "
                        "Constraints, and Objectives where your "
                        "PiecewiseLinearFunctions have been used in "
                        "expressions."
                    )
                self._transform_piecewise_linear_function(
                    t, config.descend_into_expressions
                )
            elif issubclass(t.ctype, Block):
                self._transform_block(t, config.descend_into_expressions)
            elif t.ctype is Constraint:
                if not config.descend_into_expressions:
                    raise ValueError(
                        "Encountered Constraint target '%s':\n%s"
                        % (t.name, not_walking_exprs_msg)
                    )
                self._transform_constraint(t, config.descend_into_expressions)
            elif t.ctype is Objective:
                if not config.descend_into_expressions:
                    raise ValueError(
                        "Encountered Objective target '%s':\n%s"
                        % (t.name, not_walking_exprs_msg)
                    )
                self._transform_objective(t, config.descend_into_expressions)
            else:
                raise ValueError(
                    "Target '%s' is not a PiecewiseLinearFunction, Block or "
                    "Constraint. It was of type '%s' and can't be transformed."
                    % (t.name, type(t))
                )

    def _get_transformation_block(self, parent):
        if parent in self._transformation_blocks:
            return self._transformation_blocks[parent]

        nm = unique_component_name(
            parent, '_pyomo_contrib_%s' % self._transformation_name
        )
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
                sort=SortComponents.deterministic,
            ):
                handler = self.handlers.get(obj.ctype, None)
                if not handler:
                    if handler is None:
                        raise RuntimeError(
                            "No transformation handler registered for modeling "
                            "components of type '%s'." % obj.ctype
                        )
                    continue
                handler(obj, descend_into_expressions)

    def _transform_piecewise_linear_function(
        self, pw_linear_func, descend_into_expressions
    ):
        if descend_into_expressions:
            return

        transBlock = self._get_transformation_block(pw_linear_func.parent_block())
        _functions = (
            pw_linear_func.values()
            if pw_linear_func.is_indexed()
            else (pw_linear_func,)
        )
        for pw_func in _functions:
            for pw_expr in pw_func._expressions.values():
                substitute_var = self._transform_pw_linear_expr(
                    pw_expr.expr, pw_func, transBlock
                )
                # We change the named expression to point to the variable that
                # will take the appropriate value of the piecewise linear
                # function.
                pw_expr.expr = substitute_var

        # Deactivate so that modern writers don't complain
        pw_linear_func.deactivate()

    def _transform_constraint(self, constraint, descend_into_expressions):
        if not descend_into_expressions:
            return

        transBlock = self._get_transformation_block(constraint.parent_block())
        visitor = PiecewiseLinearToMIP(self._transform_pw_linear_expr, transBlock)

        _constraints = constraint.values() if constraint.is_indexed() else (constraint,)
        for c in _constraints:
            visitor.walk_expression((c.expr, c, 0))

    def _transform_objective(self, objective, descend_into_expressions):
        if not descend_into_expressions:
            return

        transBlock = self._get_transformation_block(objective.parent_block())
        visitor = PiecewiseLinearToMIP(self._transform_pw_linear_expr, transBlock)

        _objectives = objective.values() if objective.is_indexed() else (objective,)
        for o in _objectives:
            visitor.walk_expression((o.expr, o, 0))

    def _transform_pw_linear_expr(self, pw_expr, pw_linear_func, transformation_block):
        raise DeveloperError(
            "Derived class failed to implement '_transform_pw_linear_expr'"
        )
