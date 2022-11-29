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

"""Big-M Generalized Disjunctive Programming transformation module."""

import logging

from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.config import ConfigBlock, ConfigValue
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import unique_component_name
from pyomo.common.deprecation import deprecated, deprecation_warning
from pyomo.contrib.cp.transform.logical_to_disjunctive_program import (
    LogicalToDisjunctive)
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
from pyomo.core import (
    Block, BooleanVar, Connector, Constraint, Param, Set, SetOf, Suffix, Var,
    Expression, SortComponents, TraversalStrategy, value, RangeSet,
    NonNegativeIntegers, Binary, Any)
from pyomo.core.base.external import ExternalFunction
from pyomo.core.base import Transformation, TransformationFactory, Reference
import pyomo.core.expr.current as EXPR
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.transformed_disjunct import _TransformedDisjunct
from pyomo.gdp.util import (
    is_child_of, get_src_disjunction, get_src_constraint, get_gdp_tree,
    get_transformed_constraints, _get_constraint_transBlock, get_src_disjunct,
     _warn_for_active_disjunct, preprocess_targets, _to_dict,
    _get_bigm_suffix_list, _convert_M_to_tuple, _warn_for_unused_bigM_args)
from pyomo.core.util import target_list
from pyomo.network import Port
from pyomo.repn import generate_standard_repn
from functools import wraps
from weakref import ref as weakref_ref, ReferenceType

logger = logging.getLogger('pyomo.gdp.bigm')

@TransformationFactory.register('gdp.bigm', doc="Relax disjunctive model using "
                                "big-M terms.")
class BigM_Transformation(Transformation):
    """Relax disjunctive model using big-M terms.

    Relaxes a disjunctive model into an algebraic model by adding Big-M
    terms to all disjunctive constraints.

    This transformation accepts the following keyword arguments:
        bigM: A user-specified value (or dict) of M values to use (see below)
        targets: the targets to transform [default: the instance]

    M values are determined as follows:
       1) if the constraint appears in the bigM argument dict
       2) if the constraint parent_component appears in the bigM
          argument dict
       3) if any block which is an ancestor to the constraint appears in
          the bigM argument dict
       3) if 'None' is in the bigM argument dict
       4) if the constraint or the constraint parent_component appear in
          a BigM Suffix attached to any parent_block() beginning with the
          constraint's parent_block and moving up to the root model.
       5) if None appears in a BigM Suffix attached to any
          parent_block() between the constraint and the root model.
       6) if the constraint is linear, estimate M using the variable bounds

    M values may be a single value or a 2-tuple specifying the M for the
    lower bound and the upper bound of the constraint body.

    Specifying "bigM=N" is automatically mapped to "bigM={None: N}".

    The transformation will create a new Block with a unique
    name beginning "_pyomo_gdp_bigm_reformulation".  That Block will
    contain an indexed Block named "relaxedDisjuncts", which will hold
    the relaxed disjuncts.  This block is indexed by an integer
    indicating the order in which the disjuncts were relaxed.
    Each block has a dictionary "_constraintMap":

        'srcConstraints': ComponentMap(<transformed constraint>:
                                       <src constraint>)
        'transformedConstraints': ComponentMap(<src constraint>:
                                               <transformed constraint>)

    All transformed Disjuncts will have a pointer to the block their transformed
    constraints are on, and all transformed Disjunctions will have a
    pointer to the corresponding 'Or' or 'ExactlyOne' constraint.

    """

    CONFIG = ConfigBlock("gdp.bigm")
    CONFIG.declare('targets', ConfigValue(
        default=None,
        domain=target_list,
        description="target or list of targets that will be relaxed",
        doc="""

        This specifies the list of components to relax. If None (default), the
        entire model is transformed. Note that if the transformation is done out
        of place, the list of targets should be attached to the model before it
        is cloned, and the list will specify the targets on the cloned
        instance."""
    ))
    CONFIG.declare('bigM', ConfigValue(
        default=None,
        domain=_to_dict,
        description="Big-M value used for constraint relaxation",
        doc="""

        A user-specified value, dict, or ComponentMap of M values that override
        M-values found through model Suffixes or that would otherwise be
        calculated using variable domains."""
    ))
    CONFIG.declare('assume_fixed_vars_permanent', ConfigValue(
        default=False,
        domain=bool,
        description="Boolean indicating whether or not to transform so that "
        "the transformed model will still be valid when fixed Vars are "
        "unfixed.",
        doc="""
        This is only relevant when the transformation will be estimating values
        for M. If True, the transformation will calculate M values assuming that
        fixed variables will always be fixed to their current values. This means
        that if a fixed variable is unfixed after transformation, the
        transformed model is potentially no longer valid. By default, the
        transformation will assume fixed variables could be unfixed in the
        future and will use their bounds to calculate the M value rather than
        their value. Note that this could make for a weaker LP relaxation
        while the variables remain fixed.
        """
    ))

    def __init__(self):
        """Initialize transformation object."""
        super(BigM_Transformation, self).__init__()
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
            Disjunction: False,# It's impossible to encounter an active
                               # Disjunction because preprocessing would have
                               # put it before its parent Disjunct in the order
                               # of transformation.
            Disjunct:    self._warn_for_active_disjunct,
            Block:       False,
            ExternalFunction: False,
            Port:        False, # not Arcs, because those are deactivated after
                                # the network.expand_arcs transformation
        }
        self._generate_debug_messages = False
        self._transformation_blocks = {}
        self._algebraic_constraints = {}

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

    def _apply_to(self, instance, **kwds):
        self._generate_debug_messages = is_debug_set(logger)
        self.used_args = ComponentMap() # If everything was sure to go well,
                                        # this could be a dictionary. But if
                                        # someone messes up and gives us a Var
                                        # as a key in bigMargs, I need the error
                                        # not to be when I try to put it into
                                        # this map!
        try:
            self._apply_to_impl(instance, **kwds)
        finally:
            self.used_args.clear()
            self._transformation_blocks.clear()
            self._algebraic_constraints.clear()

    def _apply_to_impl(self, instance, **kwds):
        if not instance.ctype in (Block, Disjunct):
            raise GDP_Error("Transformation called on %s of type %s. "
                            "'instance' must be a ConcreteModel, Block, or "
                            "Disjunct (in the case of nested disjunctions)." %
                            (instance.name, instance.ctype))

        config = self.CONFIG(kwds.pop('options', {}))

        # We will let args override suffixes and estimate as a last
        # resort. More specific args/suffixes override ones anywhere in
        # the tree. Suffixes lower down in the tree override ones higher
        # up.
        config.set_value(kwds)
        bigM = config.bigM
        self.assume_fixed_vars_permanent = config.assume_fixed_vars_permanent

        targets = config.targets
        # We need to check that all the targets are in fact on instance. As we
        # do this, we will use the set below to cache components we know to be
        # in the tree rooted at instance.
        knownBlocks = {}
        if targets is None:
            targets = (instance, )

        # FIXME: For historical reasons, BigM would silently skip
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
                    logger.warning(
                        'GDP.BigM transformation passed a deactivated '
                        f'target ({t.name}). Skipping.')
                else:
                    yield t
        targets = list(_filter_inactive(targets))

        # transform any logical constraints that might be anywhere on the stuff
        # we're about to transform. We do this before we preprocess targets
        # because we will likely create more disjunctive components that will
        # need transformation.
        disj_targets = []
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

        # we need to preprocess targets to make sure that if there are any
        # disjunctions in targets that their disjuncts appear before them in
        # the list.
        gdp_tree = get_gdp_tree(targets, instance, knownBlocks)
        preprocessed_targets = preprocess_targets(targets, instance,
                                                  knownBlocks,
                                                  gdp_tree=gdp_tree)

        for t in preprocessed_targets:
            if t.ctype is Disjunction:
                self._transform_disjunctionData(
                    t, t.index(), parent_disjunct=gdp_tree.parent(t),
                    root_disjunct=gdp_tree.root_disjunct(t))
            else:# We know t is a Disjunct after preprocessing
                self._transform_disjunct(
                    t, bigM, root_disjunct=gdp_tree.root_disjunct(t))

        # issue warnings about anything that was in the bigM args dict that we
        # didn't use
        _warn_for_unused_bigM_args(bigM, self.used_args, logger)

    def _add_transformation_block(self, to_block):
        if to_block in self._transformation_blocks:
            return self._transformation_blocks[to_block]

        # make a transformation block on to_block to put transformed disjuncts
        # on
        transBlockName = unique_component_name(
            to_block,
            '_pyomo_gdp_bigm_reformulation')
        self._transformation_blocks[to_block] = transBlock = Block()
        to_block.add_component(transBlockName, transBlock)
        transBlock.relaxedDisjuncts = _TransformedDisjunct(NonNegativeIntegers)

        return transBlock

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
            transBlock,disjunction.getname(fully_qualified=False) + '_xor')
        transBlock.add_component(orCname, orC)
        self._algebraic_constraints[disjunction] = orC

        return orC

    def _transform_disjunctionData(self, obj, index, parent_disjunct=None,
                                   root_disjunct=None):
        # Create or fetch the transformation block
        if root_disjunct is not None:
            # We want to put all the transformed things on the root
            # Disjunct's parent's block so that they do not get
            # re-transformed
            transBlock = self._add_transformation_block(
                root_disjunct.parent_block())
        else:
            # This isn't nested--just put it on the parent block.
            transBlock = self._add_transformation_block(obj.parent_block())

        # create or fetch the xor constraint
        xorConstraint = self._add_xor_constraint(obj.parent_component(),
                                                 transBlock)
        # Just because it's unlikely this is what someone meant to do...
        if len(obj.disjuncts) == 0:
            raise GDP_Error("Disjunction '%s' is empty. This is "
                            "likely indicative of a modeling error."  %
                            obj.name)

        # add or (or xor) constraint
        or_expr = sum(disjunct.binary_indicator_var for disjunct in
                      obj.disjuncts)

        rhs = 1 if parent_disjunct is None else \
              parent_disjunct.binary_indicator_var
        if obj.xor:
            xorConstraint[index] = or_expr == rhs
        else:
            xorConstraint[index] = or_expr >= rhs
        # Mark the DisjunctionData as transformed by mapping it to its XOR
        # constraint.
        obj._algebraic_constraint = weakref_ref(xorConstraint[index])

        # and deactivate for the writers
        obj.deactivate()

    def _transform_disjunct(self, obj, bigM, root_disjunct):
        root = root_disjunct.parent_block() if root_disjunct is not None else \
               obj.parent_block()
        transBlock = self._add_transformation_block(root)
        suffix_list = _get_bigm_suffix_list(obj)
        arg_list = self._get_bigm_arg_list(bigM, obj)

        # add reference to original disjunct on transformation block
        relaxedDisjuncts = transBlock.relaxedDisjuncts
        relaxationBlock = relaxedDisjuncts[len(relaxedDisjuncts)]
        # we will keep a map of constraints (hashable, ha!) to a tuple to
        # indicate what their M value is and where it came from, of the form:
        # ((lower_value, lower_source, lower_key), (upper_value, upper_source,
        # upper_key)), where the first tuple is the information for the lower M,
        # the second tuple is the info for the upper M, source is the Suffix or
        # argument dictionary and None if the value was calculated, and key is
        # the key in the Suffix or argument dictionary, and None if it was
        # calculated. (Note that it is possible the lower or upper is
        # user-specified and the other is not, hence the need to store
        # information for both.)
        relaxationBlock.bigm_src = {}
        relaxationBlock.localVarReferences = Block()
        # add the map that will link back and forth between transformed
        # constraints and their originals.
        relaxationBlock._constraintMap = {
            'srcConstraints': ComponentMap(),
            'transformedConstraints': ComponentMap()
        }
        relaxationBlock.transformedConstraints = Constraint(Any)
        obj._transformation_block = weakref_ref(relaxationBlock)
        relaxationBlock._src_disjunct = weakref_ref(obj)

        # This is crazy, but if the disjunction has been previously
        # relaxed, the disjunct *could* be deactivated.  This is a big
        # deal for Hull, as it uses the component_objects /
        # component_data_objects generators.  For BigM, that is OK,
        # because we never use those generators with active=True.  I am
        # only noting it here for the future when someone (me?) is
        # comparing the two relaxations.
        #
        # Transform each component within this disjunct
        self._transform_block_components(obj, obj, bigM, arg_list, suffix_list)

        # deactivate disjunct to keep the writers happy
        obj._deactivate_without_fixing_indicator()

    def _transform_block_components(self, block, disjunct, bigM, arg_list,
                                    suffix_list):
        # Find all the variables declared here (including the indicator_var) and
        # add a reference on the transformation block so these will be
        # accessible when the Disjunct is deactivated.
        varRefBlock = disjunct._transformation_block().localVarReferences
        for v in block.component_objects(Var, descend_into=Block, active=None):
            varRefBlock.add_component(unique_component_name(
                varRefBlock, v.getname(fully_qualified=False)), Reference(v))

        # Now look through the component map of block and transform everything
        # we have a handler for. Yell if we don't know how to handle it. (Note
        # that because we only iterate through active components, this means
        # non-ActiveComponent types cannot have handlers.)
        for obj in block.component_objects(active=True, descend_into=Block):
            handler = self.handlers.get(obj.ctype, None)
            if not handler:
                if handler is None:
                    raise GDP_Error(
                        "No BigM transformation handler registered "
                        "for modeling components of type %s. If your "
                        "disjuncts contain non-GDP Pyomo components that "
                        "require transformation, please transform them first."
                        % obj.ctype)
                continue
            # obj is what we are transforming, we pass disjunct
            # through so that we will have access to the indicator
            # variables down the line.
            handler(obj, disjunct, bigM, arg_list, suffix_list)

    def _warn_for_active_disjunct(self, innerdisjunct, outerdisjunct, bigMargs,
                                  arg_list, suffix_list):
        _warn_for_active_disjunct(innerdisjunct, outerdisjunct)

    def _transform_constraint(self, obj, disjunct, bigMargs, arg_list,
                              disjunct_suffix_list):
        # add constraint to the transformation block, we'll transform it there.
        transBlock = disjunct._transformation_block()
        bigm_src = transBlock.bigm_src
        constraintMap = transBlock._constraintMap

        disjunctionRelaxationBlock = transBlock.parent_block()
        
        # We will make indexes from ({obj.local_name} x obj.index_set() x ['lb',
        # 'ub']), but don't bother construct that set here, as taking Cartesian
        # products is kind of expensive (and redundant since we have the
        # original model)
        newConstraint = transBlock.transformedConstraints

        for i in sorted(obj.keys()):
            c = obj[i]
            if not c.active:
                continue

            lower = (None, None, None)
            upper = (None, None, None)

            # first, we see if an M value was specified in the arguments.
            # (This returns None if not)
            lower, upper = self._get_M_from_args(c, bigMargs, arg_list, lower,
                                                 upper)
            M = (lower[0], upper[0])

            if self._generate_debug_messages:
                logger.debug("GDP(BigM): The value for M for constraint '%s' "
                             "from the BigM argument is %s." % (c.name,
                                                                str(M)))

            # if we didn't get something we need from args, try suffixes:
            if (M[0] is None and c.lower is not None) or \
               (M[1] is None and c.upper is not None):
                # first get anything parent to c but below disjunct
                suffix_list = _get_bigm_suffix_list(
                    c.parent_block(),
                    stopping_block=disjunct)
                # prepend that to what we already collected for the disjunct.
                suffix_list.extend(disjunct_suffix_list)
                lower, upper = self._update_M_from_suffixes(c, suffix_list,
                                                            lower, upper)
                M = (lower[0], upper[0])

            if self._generate_debug_messages:
                logger.debug("GDP(BigM): The value for M for constraint '%s' "
                             "after checking suffixes is %s." % (c.name,
                                                                 str(M)))

            if c.lower is not None and M[0] is None:
                M = (self._estimate_M(c.body, c)[0] - c.lower, M[1])
                lower = (M[0], None, None)
            if c.upper is not None and M[1] is None:
                M = (M[0], self._estimate_M(c.body, c)[1] - c.upper)
                upper = (M[1], None, None)

            if self._generate_debug_messages:
                logger.debug("GDP(BigM): The value for M for constraint '%s' "
                             "after estimating (if needed) is %s." %
                             (c.name, str(M)))

            # save the source information
            bigm_src[c] = (lower, upper)

            self._add_constraint_expressions(c, i, M,
                                             disjunct.binary_indicator_var,
                                             newConstraint, constraintMap)

            # deactivate because we relaxed
            c.deactivate()

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

    def _update_M_from_suffixes(self, constraint, suffix_list, lower, upper):
        # It's possible we found half the answer in args, but we are still
        # looking for half the answer.
        need_lower = constraint.lower is not None and lower[0] is None
        need_upper = constraint.upper is not None and upper[0] is None
        M = None
        # first we check if the constraint or its parent is a key in any of the
        # suffix lists
        for bigm in suffix_list:
            if constraint in bigm:
                M = bigm[constraint]
                (lower, upper,
                 need_lower,
                 need_upper) = self._process_M_value(M, lower, upper,
                                                     need_lower, need_upper,
                                                     bigm, constraint,
                                                     constraint)
                if not need_lower and not need_upper:
                    return lower, upper

            # if c is indexed, check for the parent component
            if constraint.parent_component() in bigm:
                parent = constraint.parent_component()
                M = bigm[parent]
                (lower, upper,
                 need_lower,
                 need_upper) = self._process_M_value(M, lower, upper,
                                                     need_lower, need_upper,
                                                     bigm, parent, constraint)
                if not need_lower and not need_upper:
                    return lower, upper

        # if we didn't get an M that way, traverse upwards through the blocks
        # and see if None has a value on any of them.
        if M is None:
            for bigm in suffix_list:
                if None in bigm:
                    M = bigm[None]
                    (lower, upper,
                     need_lower,
                     need_upper) = self._process_M_value(M, lower, upper,
                                                         need_lower, need_upper,
                                                         bigm, None, constraint)
                if not need_lower and not need_upper:
                    return lower, upper
        return lower, upper

    def _estimate_M(self, expr, constraint):
        expr_lb, expr_ub = compute_bounds_on_expr(
            expr, ignore_fixed=not self.assume_fixed_vars_permanent)
        if expr_lb is None or expr_ub is None:
            raise GDP_Error("Cannot estimate M for unbounded "
                            "expressions.\n\t(found while processing "
                            "constraint '%s'). Please specify a value of M "
                            "or ensure all variables that appear in the "
                            "constraint are bounded." % constraint.name)
        else:
            M = (expr_lb, expr_ub)
        return tuple(M)

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

    @deprecated("The get_m_value_src function is deprecated. Use "
                "the get_M_value_src function if you need source "
                "information or the get_M_value function if you "
                "only need values.", version='5.7.1')
    def get_m_value_src(self, constraint):
        transBlock = _get_constraint_transBlock(constraint)
        ((lower_val, lower_source, lower_key),
         (upper_val, upper_source, upper_key)) = transBlock.bigm_src[constraint]

        if constraint.lower is not None and constraint.upper is not None and \
           (not lower_source is upper_source or not lower_key is upper_key):
            raise GDP_Error("This is why this method is deprecated: The lower "
                            "and upper M values for constraint %s came from "
                            "different sources, please use the get_M_value_src "
                            "method." % constraint.name)
        # if source and key are equal for the two, this is representable in the
        # old format.
        if constraint.lower is not None and lower_source is not None:
            return (lower_source, lower_key)
        if constraint.upper is not None and upper_source is not None:
            return (upper_source, upper_key)
        # else it was calculated:
        return (lower_val, upper_val)

    def get_M_value_src(self, constraint):
        """Return a tuple indicating how the M value used to transform
        constraint was specified. (In particular, this can be used to
        verify which BigM Suffixes were actually necessary to the
        transformation.)

        Return is of the form: ((lower_M_val, lower_M_source, lower_M_key),
                                (upper_M_val, upper_M_source, upper_M_key))

        If the constraint does not have a lower bound (or an upper bound),
        the first (second) element will be (None, None, None). Note that if
        a constraint is of the form a <= expr <= b or is an equality constraint,
        it is not necessarily true that the source of lower_M and upper_M
        are the same.

        If the M value came from an arg, source is the  dictionary itself and
        key is the key in that dictionary which gave us the M value.

        If the M value came from a Suffix, source is the BigM suffix used and
        key is the key in that Suffix.

        If the transformation calculated the value, both source and key are
        None.

        Parameters
        ----------
        constraint: Constraint, which must be in the subtree of a transformed
                    Disjunct
        """
        transBlock = _get_constraint_transBlock(constraint)
        # This is a KeyError if it fails, but it is also my fault if it
        # fails... (That is, it's a bug in the mapping.)
        return transBlock.bigm_src[constraint]

    def get_M_value(self, constraint):
        """Returns the M values used to transform constraint. Return is a tuple:
        (lower_M_value, upper_M_value). Either can be None if constraint does
        not have a lower or upper bound, respectively.

        Parameters
        ----------
        constraint: Constraint, which must be in the subtree of a transformed
                    Disjunct
        """
        transBlock = _get_constraint_transBlock(constraint)
        # This is a KeyError if it fails, but it is also my fault if it
        # fails... (That is, it's a bug in the mapping.)
        lower, upper = transBlock.bigm_src[constraint]
        return (lower[0], upper[0])

    def get_all_M_values_by_constraint(self, model):
        """Returns a dictionary mapping each constraint to a tuple:
        (lower_M_value, upper_M_value), where either can be None if the
        constraint does not have a lower or upper bound (respectively).

        Parameters
        ----------
        model: A GDP model that has been transformed with BigM
        """
        m_values = {}
        for disj in model.component_data_objects(
                Disjunct,
                active=None,
                descend_into=(Block, Disjunct)):
            transBlock = disj.transformation_block
            # First check if it was transformed at all.
            if transBlock is not None:
                # If it was transformed with BigM, we get the M values.
                if hasattr(transBlock, 'bigm_src'):
                    for cons in transBlock.bigm_src:
                        m_values[cons] = self.get_M_value(cons)
        return m_values

    def get_largest_M_value(self, model):
        """Returns the largest M value for any constraint on the model.

        Parameters
        ----------
        model: A GDP model that has been transformed with BigM
        """
        return max(max(abs(m) for m in m_values if m is not None) for m_values
                   in self.get_all_M_values_by_constraint(model).values())
