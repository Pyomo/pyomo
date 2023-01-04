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
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr

from pyomo.core.base import Transformation, TransformationFactory
from pyomo.core.base.external import ExternalFunction
from pyomo.core import (
    Block, BooleanVar, Connector, Constraint, Expression, NonNegativeIntegers,
    Param, RangeSet, Set, SetOf, Suffix, Var)
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.transformed_disjunct import _TransformedDisjunct
from pyomo.gdp.util import get_gdp_tree
from pyomo.network import Port

class GDP_to_MIP_Transformation(Transformation):
    """
    Base class for transformations from GDP to MIP
    """
    def __init__(self, logger):
        """Initialize transformation object."""
        super(GDP_to_MIP_Transformation, self).__init__()
        self.logger = logger
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

    def _transform_logical_constraints(self, instance):
        # transform any logical constraints that might be anywhere on the stuff
        # we're about to transform. We do this before we preprocess targets
        # because we will likely create more disjunctive components that will
        # need transformation.
        disj_targets = []
        for t in self.targets:
            disj_datas = t.values() if t.is_indexed() else [t,]
            if t.ctype is Disjunct:
                disj_targets.extend(disj_datas)
            if t.ctype is Disjunction:
                disj_targets.extend([d for disjunction in disj_datas for d in
                                     disjunction.disjuncts])
        TransformationFactory('contrib.logical_to_disjunctive').apply_to(
            instance,
            targets=[blk for blk in self.targets if blk.ctype is Block] +
            disj_targets)

    def _filter_targets(self, instance):
        targets = self._config.targets
        if targets is None:
            targets = (instance, )

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
        self.targets = list(_filter_inactive(targets))

    def _get_gdp_tree_from_targets(self, instance):
        knownBlocks = {}
        # we need to preprocess targets to make sure that if there are any
        # disjunctions in targets that they appear before disjunctions that are
        # contained in their disjuncts. That is, in hull, we will transform from
        # root to leaf in order to avoid have to modify transformed constraints
        # more than once: It is most efficient to build nested transformed
        # constraints when we already have the disaggregated variables of the
        # parent disjunct.
        return get_gdp_tree(self.targets, instance)

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

class _BigM_MixIn(object):
    def _get_bigm_arg_list(self, bigm_args, block):
        # Gather what we know about blocks from args exactly once. We'll still
        # check for constraints in the moment, but if that fails, we've
        # preprocessed the time-consuming part of traversing up the tree.
        arg_list = []
        if bigm_args is None:
            return arg_list
        while block is not None:
            if block in bigm_args:
                arg_list.append({block: bigm_args[block]})
            block = block.parent_block()
        return arg_list

    def _process_M_value(self, m, lower, upper, need_lower, need_upper, src,
                         key, constraint, from_args=False):
        m = _convert_M_to_tuple(m, constraint)
        if need_lower and m[0] is not None:
            if from_args:
                self.used_args[key] = m
            lower = (m[0], src, key)
            need_lower = False
        if need_upper and m[1] is not None:
            if from_args:
                self.used_args[key] = m
            upper = (m[1], src, key)
            need_upper = False
        return lower, upper, need_lower, need_upper

    def _get_M_from_args(self, constraint, bigMargs, arg_list, lower, upper):
        # check args: we first look in the keys for constraint and
        # constraintdata. In the absence of those, we traverse up the blocks,
        # and as a last resort check for a value for None
        if bigMargs is None:
            return (lower, upper)

        # since we check for args first, we know lower[0] and upper[0] are both
        # None
        need_lower = constraint.lower is not None
        need_upper = constraint.upper is not None

        # check for the constraint itself and its container
        parent = constraint.parent_component()
        if constraint in bigMargs:
            m = bigMargs[constraint]
            (lower, upper,
             need_lower, need_upper) = self._process_M_value(m, lower, upper,
                                                             need_lower,
                                                             need_upper,
                                                             bigMargs,
                                                             constraint,
                                                             constraint,
                                                             from_args=True)
            if not need_lower and not need_upper:
                return lower, upper
        elif parent in bigMargs:
            m = bigMargs[parent]
            (lower, upper,
             need_lower, need_upper) = self._process_M_value(m, lower, upper,
                                                             need_lower,
                                                             need_upper,
                                                             bigMargs, parent,
                                                             constraint,
                                                             from_args=True)
            if not need_lower and not need_upper:
                return lower, upper

        # use the precomputed traversal up the blocks
        for arg in arg_list:
            for block, val in arg.items():
                (lower, upper,
                 need_lower,
                 need_upper) = self._process_M_value(val, lower, upper,
                                                     need_lower, need_upper,
                                                     bigMargs, block,
                                                     constraint,
                                                     from_args=True)
                if not need_lower and not need_upper:
                    return lower, upper

        # last check for value for None!
        if None in bigMargs:
            m = bigMargs[None]
            (lower, upper,
             need_lower, need_upper) = self._process_M_value(m, lower, upper,
                                                             need_lower,
                                                             need_upper,
                                                             bigMargs, None,
                                                             constraint,
                                                             from_args=True)
            if not need_lower and not need_upper:
                return lower, upper

        return lower, upper

    def _estimate_M(self, expr, constraint):
        expr_lb, expr_ub = compute_bounds_on_expr(
            expr, ignore_fixed=not self._config.assume_fixed_vars_permanent)
        if expr_lb is None or expr_ub is None:
            raise GDP_Error("Cannot estimate M for unbounded "
                            "expressions.\n\t(found while processing "
                            "constraint '%s'). Please specify a value of M "
                            "or ensure all variables that appear in the "
                            "constraint are bounded." % constraint.name)
        else:
            M = (expr_lb, expr_ub)
        return tuple(M)

    def _add_constraint_expressions(self, c, i, M, indicator_var, newConstraint,
                                    constraintMap):
        # Since we are both combining components from multiple blocks and using
        # local names, we need to make sure that the first index for
        # transformedConstraints is guaranteed to be unique. We just grab the
        # current length of the list here since that will be monotonically
        # increasing and hence unique. We'll append it to the
        # slightly-more-human-readable constraint name for something familiar
        # but unique. (Note that we really could do this outside of the loop
        # over the constraint indices, but I don't think it matters a lot.)
        unique = len(newConstraint)
        name = c.local_name + "_%s" % unique

        if c.lower is not None:
            if M[0] is None:
                raise GDP_Error("Cannot relax disjunctive constraint '%s' "
                                "because M is not defined." % name)
            M_expr = M[0] * (1 - indicator_var)
            newConstraint.add((name, i, 'lb'), c.lower <= c. body - M_expr)
            constraintMap[
                'transformedConstraints'][c] = [
                    newConstraint[name, i, 'lb']]
            constraintMap['srcConstraints'][
                newConstraint[name, i, 'lb']] = c
        if c.upper is not None:
            if M[1] is None:
                raise GDP_Error("Cannot relax disjunctive constraint '%s' "
                                "because M is not defined." % name)
            M_expr = M[1] * (1 - indicator_var)
            newConstraint.add((name, i, 'ub'), c.body - M_expr <= c.upper)
            transformed = constraintMap['transformedConstraints'].get(c)
            if transformed is not None:
                constraintMap['transformedConstraints'][
                    c].append(newConstraint[name, i, 'ub'])
            else:
                constraintMap[
                    'transformedConstraints'][c] = [
                        newConstraint[name, i, 'ub']]
            constraintMap['srcConstraints'][
                newConstraint[name, i, 'ub']] = c
