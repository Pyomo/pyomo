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

from pyomo.contrib.cp.transform.logical_to_disjunctive_walker import (
    LogicalToDisjunctiveVisitor,
)
from pyomo.common.collections import ComponentMap
from pyomo.common.modeling import unique_component_name
from pyomo.common.config import ConfigDict, ConfigValue

from pyomo.core import (
    TransformationFactory,
    VarList,
    Binary,
    LogicalConstraint,
    Block,
    ConstraintList,
    Transformation,
    NonNegativeIntegers,
)
from pyomo.core.base.block import _BlockData
from pyomo.core.base import SortComponents
from pyomo.core.util import target_list
from pyomo.gdp import Disjunct, Disjunction


@TransformationFactory.register(
    "contrib.logical_to_disjunctive",
    doc="Convert logical propositions with only Boolean arguments to MIP "
    "representation and convert logical expressions with mixed "
    "integer-Boolean arguments (such as atleast, atmost, and exactly) to "
    "disjunctive representation",
)
class LogicalToDisjunctive(Transformation):
    """
    Re-encode logical constraints as linear constraints,
    converting Boolean variables to binary.
    """

    CONFIG = ConfigDict('core.logical_to_disjunctive')
    CONFIG.declare(
        'targets',
        ConfigValue(
            default=None,
            domain=target_list,
            description="target or list of targets that will be relaxed",
            doc="""
            This specifies the list of LogicalConstraints to transform, or the
            list of Blocks or Disjuncts on which to transform all of the
            LogicalConstraints. Note that if the transformation is done out
            of place, the list of targets should be attached to the model before it
            is cloned, and the list will specify the targets on the cloned
            instance.
            """,
        ),
    )

    def _apply_to(self, model, **kwds):
        config = self.CONFIG(kwds.pop('options', {}))
        config.set_value(kwds)
        targets = config.targets
        if targets is None:
            targets = (model,)

        transBlocks = {}
        visitor = LogicalToDisjunctiveVisitor()
        for t in targets:
            if t.ctype is Block or isinstance(t, _BlockData):
                self._transform_block(t, model, visitor, transBlocks)
            elif t.ctype is LogicalConstraint:
                if t.is_indexed():
                    self._transform_constraint(t, visitor, transBlocks)
                else:
                    self._transform_constraintData(t, visitor, transBlocks)
            else:
                raise RuntimeError(
                    "Target '%s' was not a Block, Disjunct, or"
                    " LogicalConstraint. It was of type %s "
                    "and can't be transformed." % (t.name, type(t))
                )

    def _transform_constraint(self, constraint, visitor, transBlocks):
        for i in constraint.keys(sort=SortComponents.ORDERED_INDICES):
            self._transform_constraintData(constraint[i], visitor, transBlocks)
        constraint.deactivate()

    def _transform_block(self, target_block, model, new_varlists, transBlocks):
        _blocks = (
            target_block.values() if target_block.is_indexed() else (target_block,)
        )
        for block in _blocks:
            # Note that this changes the current (though not the original)
            # behavior of logical-to-linear because we descend into Disjuncts in
            # order to find logical constraints. In the context of creating a
            # traditional disjunctive program, this makes sense--we cannot have
            # logical constraints *anywhere* in the active tree after this
            # transformation.
            for logical_constraint in block.component_objects(
                ctype=LogicalConstraint, active=True, descend_into=(Block, Disjunct)
            ):
                self._transform_constraint(
                    logical_constraint, new_varlists, transBlocks
                )

    def _transform_constraintData(self, logical_constraint, visitor, transBlocks):
        # now create a transformation block on the constraint's parent block (if
        # we don't have one already)
        parent_block = logical_constraint.parent_block()
        xfrm_block = transBlocks.get(parent_block)
        if xfrm_block is None:
            xfrm_block = self._create_transformation_block(parent_block)
            transBlocks[parent_block] = xfrm_block

        # This is may be too cute, but just deceive the walker so it puts stuff
        # in the right place.
        visitor.constraints = xfrm_block.transformed_constraints
        visitor.z_vars = xfrm_block.auxiliary_vars
        visitor.disjuncts = xfrm_block.auxiliary_disjuncts
        visitor.disjunctions = xfrm_block.auxiliary_disjunctions
        visitor.walk_expression(logical_constraint.expr)
        logical_constraint.deactivate()

    def _create_transformation_block(self, context):
        new_xfrm_block_name = unique_component_name(context, '_logical_to_disjunctive')
        new_xfrm_block = Block(doc="Transformation objects for logical_to_disjunctive")
        context.add_component(new_xfrm_block_name, new_xfrm_block)

        new_xfrm_block.transformed_constraints = ConstraintList()
        new_xfrm_block.auxiliary_vars = VarList(domain=Binary)
        new_xfrm_block.auxiliary_disjuncts = Disjunct(NonNegativeIntegers)
        new_xfrm_block.auxiliary_disjunctions = Disjunction(NonNegativeIntegers)

        return new_xfrm_block
