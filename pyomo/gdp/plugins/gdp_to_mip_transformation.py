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

from pyomo.common.log import is_debug_set
from pyomo.common.modeling import unique_component_name

from pyomo.core.base import Transformation, TransformationFactory
from pyomo.core.base.external import ExternalFunction
from pyomo.core import (
    Block, BooleanVar, Connector, Constraint, Expression, NonNegativeIntegers,
    Param, RangeSet, Set, SetOf, Suffix, Var)
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.transformed_disjunct import _TransformedDisjunct
from pyomo.gdp.util import get_gdp_tree
from pyomo.network import Port

class GDP_to_MIP_Transformation(Transformation):
    """
    Base class for transformations from GDP to MIP
    """
    def __init__(self):
        """Initialize transformation object."""
        super(GDP_to_MIP_Transformation, self).__init__()
        self.handlers = {
            Constraint:  self._transform_constraint,
            Var:         False, # Note that if a Var appears on a Disjunct, we
                                # still treat its bounds as global. If the
                                # intent is for its bounds to be on the
                                # disjunct, it should be declared with no bounds
                                # and the bounds should be set in constraints on
                                # the Disjunct.
            BooleanVar:  False,
            Connector:   False,
            Expression:  False,
            Suffix:      False,
            Param:       False,
            Set:         False,
            SetOf:       False,
            RangeSet:    False,
            Disjunction: False,# In BigM, it's impossible to encounter an active
                               # Disjunction because preprocessing would have
                               # put it before its parent Disjunct in the order
                               # of transformation. In hull, we intentionally
                               # pass over active Disjunctions that are on
                               # Disjuncts because we know they are in the list
                               # of objects to transform after preprocessing, so
                               # they will be transformed later.
            Disjunct:    self._warn_for_active_disjunct,
            Block:       False,
            ExternalFunction: False,
            Port:        False, # not Arcs, because those are deactivated after
                                # the network.expand_arcs transformation
        }
        self._generate_debug_messages = False
        self._transformation_blocks = {}
        self._algebraic_constraints = {}

    def _apply_to(self, instance, **kwds):
        try:
            self._apply_to_impl(instance, **kwds)
        finally:
            self._transformation_blocks.clear()
            self._algebraic_constraints.clear()

    def _apply_to_impl(self, instance, **kwds):
        if not instance.ctype in (Block, Disjunct):
            raise GDP_Error("Transformation called on %s of type %s. 'instance'"
                            " must be a ConcreteModel, Block, or Disjunct (in "
                            "the case of nested disjunctions)." %
                            (instance.name, instance.ctype))

        self._config = self.CONFIG(kwds.pop('options', {}))
        self._config.set_value(kwds)
        self._generate_debug_messages = is_debug_set(self.logger)

        # transform any logical constraints that might be anywhere on the stuff
        # we're about to transform. We do this before we preprocess targets
        # because we will likely create more disjunctive components that will
        # need transformation.
        disj_targets = []
        targets = (instance,) if self._config.targets is None else \
                  self._config.targets
        for t in targets:
            disj_datas = t.values() if t.is_indexed() else [t,]
            if t.ctype is Disjunct:
                disj_targets.extend(disj_datas)
            if t.ctype is Disjunction:
                disj_targets.extend([d for disjunction in disj_datas for d in
                                     disjunction.disjuncts])
        TransformationFactory('contrib.logical_to_disjunctive').apply_to(
            instance,
            targets=[blk for blk in targets if blk.ctype is Block] +
            disj_targets)

    def _get_gdp_tree_from_targets(self, instance):
        targets = self._config.targets
        knownBlocks = {}
        if targets is None:
            targets = ( instance, )

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
                        'GDP.Hull transformation passed a deactivated '
                        f'target ({t.name}). Skipping.')
                else:
                    yield t
        targets = list(_filter_inactive(targets))

        # we need to preprocess targets to make sure that if there are any
        # disjunctions in targets that they appear before disjunctions that are
        # contained in their disjuncts. That is, in hull, we will transform from
        # root to leaf in order to avoid have to modify transformed constraints
        # more than once: It is most efficient to build nested transformed
        # constraints when we already have the disaggregated variables of the
        # parent disjunct.
        return get_gdp_tree(targets, instance, knownBlocks)

    def _add_transformation_block(self, to_block, transformation_name):
        if to_block in self._transformation_blocks:
            return self._transformation_blocks[to_block]

        # make a transformation block on to_block to put transformed disjuncts
        # on
        transBlockName = unique_component_name(
            to_block,
            '_pyomo_gdp_%s_reformulation' % transformation_name)
        self._transformation_blocks[to_block] = transBlock = Block()
        to_block.add_component(transBlockName, transBlock)
        transBlock.relaxedDisjuncts = _TransformedDisjunct(NonNegativeIntegers)

        return transBlock

    def _transform_constraint(self, *args):
        raise NotImplementedError(
            "Transformation failed to implement _transform_constraint")
