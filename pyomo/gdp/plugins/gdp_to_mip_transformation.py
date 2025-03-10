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

from functools import wraps

from pyomo.common.autoslots import AutoSlots
from pyomo.common.collections import ComponentMap, DefaultComponentMap
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import unique_component_name

from pyomo.core.base import Transformation, TransformationFactory
from pyomo.core.base.external import ExternalFunction
from pyomo.core import (
    Any,
    Block,
    BooleanVar,
    Connector,
    Constraint,
    Expression,
    NonNegativeIntegers,
    Param,
    RangeSet,
    Reference,
    Set,
    SetOf,
    Suffix,
    Var,
)
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.transformed_disjunct import _TransformedDisjunct
from pyomo.gdp.util import (
    get_gdp_tree,
    get_src_constraint,
    get_src_disjunct,
    get_src_disjunction,
    get_transformed_constraints,
    _warn_for_active_disjunct,
)
from pyomo.network import Port

from weakref import ref as weakref_ref


class _GDPTransformationData(AutoSlots.Mixin):
    __slots__ = ('src_constraint', 'transformed_constraints')

    def __init__(self):
        self.src_constraint = ComponentMap()
        self.transformed_constraints = DefaultComponentMap(list)


Block.register_private_data_initializer(_GDPTransformationData, scope='pyomo.gdp')


class GDP_to_MIP_Transformation(Transformation):
    """
    Base class for transformations from GDP to MIP
    """

    def __init__(self, logger):
        """Initialize transformation object."""
        super(GDP_to_MIP_Transformation, self).__init__()
        self.logger = logger
        self.handlers = {
            Constraint: self._transform_constraint,
            Var: False,  # Note that if a Var appears on a Disjunct, we
            # still treat its bounds as global. If the
            # intent is for its bounds to be on the
            # disjunct, it should be declared with no bounds
            # and the bounds should be set in constraints on
            # the Disjunct.
            BooleanVar: False,
            Connector: False,
            Expression: False,
            Suffix: False,
            Param: False,
            Set: False,
            SetOf: False,
            RangeSet: False,
            Disjunction: False,  # In BigM, it's impossible to encounter an active
            # Disjunction because preprocessing would have
            # put it before its parent Disjunct in the order
            # of transformation. In hull, we intentionally
            # pass over active Disjunctions that are on
            # Disjuncts because we know they are in the list
            # of objects to transform after preprocessing, so
            # they will be transformed later.
            Disjunct: self._warn_for_active_disjunct,
            Block: False,
            ExternalFunction: False,
            Port: False,  # not Arcs, because those are deactivated after
            # the network.expand_arcs transformation
        }
        self._generate_debug_messages = False
        self._transformation_blocks = {}
        self._algebraic_constraints = {}

    def _restore_state(self):
        self._transformation_blocks.clear()
        self._algebraic_constraints.clear()
        if hasattr(self, '_config'):
            del self._config

    def _process_arguments(self, instance, **kwds):
        if not instance.ctype in (Block, Disjunct):
            raise GDP_Error(
                "Transformation called on %s of type %s. 'instance' "
                "must be a ConcreteModel, Block, or Disjunct (in "
                "the case of nested disjunctions)." % (instance.name, instance.ctype)
            )

        self._config = self.CONFIG(kwds.pop('options', {}))
        self._config.set_value(kwds)
        self._generate_debug_messages = is_debug_set(self.logger)

    def _transform_logical_constraints(self, instance, targets):
        # transform any logical constraints that might be anywhere on the stuff
        # we're about to transform. We do this before we preprocess targets
        # because we will likely create more disjunctive components that will
        # need transformation.
        disj_targets = []
        for t in targets:
            disj_datas = t.values() if t.is_indexed() else [t]
            if t.ctype is Disjunct:
                disj_targets.extend(disj_datas)
            if t.ctype is Disjunction:
                disj_targets.extend(
                    [d for disjunction in disj_datas for d in disjunction.disjuncts]
                )
        TransformationFactory('contrib.logical_to_disjunctive').apply_to(
            instance,
            targets=[blk for blk in targets if blk.ctype is Block] + disj_targets,
        )

    def _filter_targets(self, instance):
        targets = self._config.targets
        if targets is None:
            targets = (instance,)

        # FIXME: For historical reasons, Hull would silently skip
        # any targets that were explicitly deactivated.  This
        # preserves that behavior (although adds a warning).  We
        # should revisit that design decision and probably remove
        # this filter, as it is slightly ambiguous as to what it
        # means for the target to be deactivated: is it just the
        # target itself [historical implementation] or any block in
        # the hierarchy?
        def _filter_inactive(targets):
            for t in targets:
                if not t.active:
                    self.logger.warning(
                        f'GDP.{self.transformation_name} transformation passed '
                        f'a deactivated target ({t.name}). Skipping.'
                    )
                else:
                    yield t

        return list(_filter_inactive(targets))

    def _get_gdp_tree_from_targets(self, instance, targets):
        knownBlocks = {}
        # we need to preprocess targets to make sure that if there are any
        # disjunctions in targets that they appear before disjunctions that are
        # contained in their disjuncts. That is, in hull, we will transform from
        # root to leaf in order to avoid have to modify transformed constraints
        # more than once: It is most efficient to build nested transformed
        # constraints when we already have the disaggregated variables of the
        # parent disjunct.
        return get_gdp_tree(targets, instance)

    def _add_transformation_block(self, to_block):
        if to_block in self._transformation_blocks:
            return self._transformation_blocks[to_block], False

        # make a transformation block on to_block to put transformed disjuncts
        # on
        transBlockName = unique_component_name(
            to_block, '_pyomo_gdp_%s_reformulation' % self.transformation_name
        )
        self._transformation_blocks[to_block] = transBlock = Block()
        to_block.add_component(transBlockName, transBlock)
        transBlock.relaxedDisjuncts = _TransformedDisjunct(NonNegativeIntegers)

        return transBlock, True

    def _add_xor_constraint(self, disjunction, transBlock):
        # Put the disjunction constraint on the transformation block and
        # determine whether it is an OR or XOR constraint.
        # We never do this for just a DisjunctionData because we need to know
        # about the index set of its parent component (so that we can make the
        # index of this constraint match). So if we called this on a
        # DisjunctionData, we did something wrong.

        # first check if the constraint already exists
        if disjunction in self._algebraic_constraints:
            return self._algebraic_constraints[disjunction]

        # add the XOR (or OR) constraints to parent block (with unique name)
        # It's indexed if this is an IndexedDisjunction, not otherwise
        if disjunction.is_indexed():
            orC = Constraint(Any)
        else:
            orC = Constraint()
        orCname = unique_component_name(
            transBlock, disjunction.getname(fully_qualified=False) + '_xor'
        )
        transBlock.add_component(orCname, orC)
        self._algebraic_constraints[disjunction] = orC

        return orC

    def _setup_transform_disjunctionData(self, obj, root_disjunct):
        # Just because it's unlikely this is what someone meant to do...
        if len(obj.disjuncts) == 0:
            raise GDP_Error(
                "Disjunction '%s' is empty. This is "
                "likely indicative of a modeling error." % obj.name
            )

        # We always need to create or fetch a transformation block on the parent block.
        trans_block, new_block = self._add_transformation_block(obj.parent_block())
        # This is where we put exactly_one/or constraint
        algebraic_constraint = self._add_xor_constraint(
            obj.parent_component(), trans_block
        )

        # If requested, create or fetch the transformation block above the
        # nested hierarchy
        if root_disjunct is not None:
            # We want to put some transformed things on the root Disjunct's
            # parent's block so that they do not get re-transformed. (Note this
            # is never true for hull, but it calls this method with
            # root_disjunct=None. BigM can't put the exactly-one constraint up
            # here, but it can put everything else.)
            trans_block, new_block = self._add_transformation_block(
                root_disjunct.parent_block()
            )

        return trans_block, algebraic_constraint

    def _get_disjunct_transformation_block(self, disjunct, transBlock):
        if disjunct.transformation_block is not None:
            return disjunct.transformation_block

        # create a relaxation block for this disjunct
        relaxedDisjuncts = transBlock.relaxedDisjuncts
        relaxationBlock = relaxedDisjuncts[len(relaxedDisjuncts)]

        relaxationBlock.transformedConstraints = Constraint(Any)
        relaxationBlock.localVarReferences = Block()

        # add mappings to source disjunct (so we'll know we've relaxed)
        disjunct._transformation_block = weakref_ref(relaxationBlock)
        relaxationBlock._src_disjunct = weakref_ref(disjunct)

        return relaxationBlock

    def _transform_block_components(self, block, disjunct, *args):
        # Find all the variables declared here (including the indicator_var) and
        # add a reference on the transformation block so these will be
        # accessible when the Disjunct is deactivated. Note that in hull, we do
        # this after we have moved up the transformation blocks for nested
        # disjunctions, so that we don't have duplicate references.
        varRefBlock = disjunct._transformation_block().localVarReferences
        for v in block.component_objects(Var, descend_into=Block, active=None):
            varRefBlock.add_component(
                unique_component_name(varRefBlock, v.getname(fully_qualified=False)),
                Reference(v),
            )

        # Now look through the component map of block and transform everything
        # we have a handler for. Yell if we don't know how to handle it. (Note
        # that because we only iterate through active components, this means
        # non-ActiveComponent types cannot have handlers.)
        for obj in block.component_objects(active=True, descend_into=Block):
            handler = self.handlers.get(obj.ctype, None)
            if not handler:
                if handler is None:
                    raise GDP_Error(
                        "No %s transformation handler registered "
                        "for modeling components of type %s. If your "
                        "disjuncts contain non-GDP Pyomo components that "
                        "require transformation, please transform them first."
                        % (self.transformation_name, obj.ctype)
                    )
                continue
            # obj is what we are transforming, we pass disjunct
            # through so that we will have access to the indicator
            # variables down the line.
            handler(obj, disjunct, *args)

    def _transform_constraint(self, obj, disjunct, *args):
        raise NotImplementedError(
            "Class %s failed to implement '_transform_constraint'" % self.__class__
        )

    def _warn_for_active_disjunct(self, innerdisjunct, outerdisjunct, *args):
        _warn_for_active_disjunct(innerdisjunct, outerdisjunct)

    # These are all functions to retrieve transformed components from
    # original ones and vice versa.

    @wraps(get_src_disjunct)
    def get_src_disjunct(self, transBlock):
        return get_src_disjunct(transBlock)

    @wraps(get_src_disjunction)
    def get_src_disjunction(self, xor_constraint):
        return get_src_disjunction(xor_constraint)

    @wraps(get_src_constraint)
    def get_src_constraint(self, transformedConstraint):
        return get_src_constraint(transformedConstraint)

    @wraps(get_transformed_constraints)
    def get_transformed_constraints(self, srcConstraint):
        return get_transformed_constraints(srcConstraint)
