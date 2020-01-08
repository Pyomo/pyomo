#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Big-M Generalized Disjunctive Programming transformation module."""

import logging
import textwrap

from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
from pyomo.contrib.fbbt.interval import inf
from pyomo.core import (
    Block, Connector, Constraint, Param, Set, Suffix, Var,
    Expression, SortComponents, TraversalStrategy, Any, value,
    RangeSet)
from pyomo.core.base import Transformation, TransformationFactory
from pyomo.core.base.component import ComponentUID, ActiveComponent
from pyomo.core.base.PyomoModel import ConcreteModel, AbstractModel
from pyomo.core.kernel.component_map import ComponentMap
from pyomo.core.kernel.component_set import ComponentSet
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.disjunct import _DisjunctData, SimpleDisjunct
from pyomo.gdp.util import target_list, is_child_of
from pyomo.gdp.plugins.gdp_var_mover import HACK_GDP_Disjunct_Reclassifier
from pyomo.repn import generate_standard_repn
from pyomo.common.config import ConfigBlock, ConfigValue
from pyomo.common.modeling import unique_component_name
from pyomo.common.deprecation import deprecation_warning
from six import iterkeys, iteritems
from weakref import ref as weakref_ref

logger = logging.getLogger('pyomo.gdp.bigm')

NAME_BUFFER = {}

def _to_dict(val):
    if isinstance(val, ComponentMap):
        return val
    if isinstance(val, dict):
        return val
    return {None: val}


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
    name beginning "_pyomo_gdp_bigm_relaxation".  That Block will
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
    pointer to the corresponding OR or XOR constraint.

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

    def __init__(self):
        """Initialize transformation object."""
        super(BigM_Transformation, self).__init__()
        self.handlers = {
            Constraint:  self._transform_constraint,
            Var:         False,
            Connector:   False,
            Expression:  False,
            Suffix:      False,
            Param:       False,
            Set:         False,
            RangeSet:    False,
            Disjunction: self._warn_for_active_disjunction,
            Disjunct:    self._warn_for_active_disjunct,
            Block:       self._transform_block_on_disjunct,
        }

    def _get_bigm_suffix_list(self, block):
        # Note that you can only specify suffixes on BlockData objects or
        # SimpleBlocks. Though it is possible at this point to stick them
        # on whatever components you want, we won't pick them up.
        suffix_list = []
        while block is not None:
            bigm = block.component('BigM')
            if type(bigm) is Suffix:
                suffix_list.append(bigm)
            block = block.parent_block()
        return suffix_list


    def _apply_to(self, instance, **kwds):
        assert not NAME_BUFFER
        try:
            self._apply_to_impl(instance, **kwds)
        finally:
            # Clear the global name buffer now that we are done
            NAME_BUFFER.clear()


    def _apply_to_impl(self, instance, **kwds):
        config = self.CONFIG(kwds.pop('options', {}))

        # We will let args override suffixes and estimate as a last
        # resort. More specific args/suffixes override ones anywhere in
        # the tree. Suffixes lower down in the tree override ones higher
        # up.
        if 'default_bigM' in kwds:
            logger.warn("DEPRECATED: the 'default_bigM=' argument has been "
                        "replaced by 'bigM='")
            config.bigM = kwds.pop('default_bigM')

        config.set_value(kwds)
        bigM = config.bigM

        targets = config.targets
        if targets is None:
            targets = (instance, )
            _HACK_transform_whole_instance = True
        else:
            _HACK_transform_whole_instance = False
        # We need to check that all the targets are in fact on instance. As we
        # do this, we will use the set below to cache components we know to be
        # in the tree rooted at instance.
        knownBlocks = set()
        for t in targets:
            # [ESJ 08/22/2019] This can go away when we deprecate CUIDs. The
            # warning is in util, but we have to deal with the consequences here
            # because we need to have the instance in order to get the
            # component.
            if isinstance(t, ComponentUID):
                tmp = t
                t = t.find_component(instance)
                if t is None:
                    raise GDP_Error(
                        "Target %s is not a component on the instance!" % tmp)

            # check that t is in fact a child of instance
            if not is_child_of(parent=instance, child=t,
                               knownBlocks=knownBlocks):
                raise GDP_Error("Target %s is not a component on instance %s!"
                                % (t.name, instance.name))
            if t.type() is Disjunction:
                if t.parent_component() is t:
                    self._transform_disjunction(t, bigM)
                else:
                    self._transform_disjunctionData( t, bigM, t.index())
            elif t.type() in (Block, Disjunct):
                if t.parent_component() is t:
                    self._transform_block(t, bigM)
                else:
                    self._transform_blockData(t, bigM)
            else:
                raise GDP_Error(
                    "Target %s was not a Block, Disjunct, or Disjunction. "
                    "It was of type %s and can't be transformed."
                    % (t.name, type(t)))

        # HACK for backwards compatibility with the older GDP transformations
        #
        # Until the writers are updated to find variables on things
        # other than active blocks, we need to reclassify the Disjuncts
        # as Blocks after transformation so that the writer will pick up
        # all the variables that it needs (in this case, indicator_vars).
        if _HACK_transform_whole_instance:
            HACK_GDP_Disjunct_Reclassifier().apply_to(instance)

    def _add_transformation_block(self, instance):
        # make a transformation block on instance to put transformed disjuncts
        # on
        transBlockName = unique_component_name(
            instance,
            '_pyomo_gdp_bigm_relaxation')
        transBlock = Block()
        instance.add_component(transBlockName, transBlock)
        transBlock.relaxedDisjuncts = Block(Any)
        transBlock.lbub = Set(initialize=['lb', 'ub'])

        return transBlock

    def _transform_block(self, obj, bigM):
        for i in sorted(iterkeys(obj)):
            self._transform_blockData(obj[i], bigM)

    def _transform_blockData(self, obj, bigM):
        # Transform every (active) disjunction in the block
        for disjunction in obj.component_objects(
                Disjunction,
                active=True,
                sort=SortComponents.deterministic,
                descend_into=(Block, Disjunct),
                descent_order=TraversalStrategy.PostfixDFS):
            self._transform_disjunction(disjunction, bigM)

    def _get_xor_constraint(self, disjunction):
        # Put the disjunction constraint on its parent block and determine
        # whether it is an OR or XOR constraint. Note that we put the XOR
        # constraint on the parent block for convenience in the case of nested
        # disjunctions. Because it is the only product of the transformation
        # which we do not move off of the outer disjunct when we transform
        # it. (This differs from chull, where we move nothing for nested
        # disjunctions and so store the XOR constraint on the transformation
        # block.)

        # We never do this for just a DisjunctionData because we need to know
        # about the index set of its parent component (so that we can make the
        # index of this constraint match). So if we called this on a
        # DisjunctionData, we did something wrong.
        assert isinstance(disjunction, Disjunction)

        # first check if the constraint already exists
        if not disjunction._algebraic_constraint is None:
            return disjunction._algebraic_constraint()

        parent = disjunction.parent_block()

        # add the XOR (or OR) constraints to parent block (with unique name)
        # It's indexed if this is an IndexedDisjunction, not otherwise
        orC = Constraint(disjunction.index_set()) if \
            disjunction.is_indexed() else Constraint()
        # The name used to indicate if there were OR or XOR disjunctions,
        # however now that Disjunctions are allowed to mix the state we
        # can no longer make that distinction in the name.
        #    nm = '_xor' if xor else '_or'
        nm = '_xor'
        orCname = unique_component_name(
            parent, '_gdp_bigm_relaxation_' + disjunction.getname(
                fully_qualified=True, name_buffer=NAME_BUFFER) + nm)
        parent.add_component(orCname, orC)
        disjunction._algebraic_constraint = weakref_ref(orC)

        return orC

    def _transform_disjunction(self, obj, bigM):
        transBlock = self._add_transformation_block(obj.parent_block())

        # If this is an IndexedDisjunction, we have to create the XOR constraint
        # here because we want its index to match the disjunction. In any case,
        # we might as well.
        xorConstraint = self._get_xor_constraint(obj)

        # relax each of the disjunctionDatas
        for i in sorted(iterkeys(obj)):
            self._transform_disjunctionData(obj[i], bigM, i, xorConstraint,
                                            transBlock)

        # deactivate so the writers don't scream
        obj.deactivate()

    def _transform_disjunctionData(self, obj, bigM, index, xorConstraint=None,
                                   transBlock=None):
        if not obj.active:
            return  # Do not process a deactivated disjunction 
        # We won't have these arguments if this got called straight from
        # targets. But else, we created them earlier, and have just been passing
        # them through.
        if transBlock is None:
            transBlock = self._add_transformation_block(obj.parent_block())
        if xorConstraint is None:
            xorConstraint = self._get_xor_constraint(obj.parent_component())

        xor = obj.xor
        or_expr = 0
        # Just because it's unlikely this is what someone meant to do...    
        if len(obj.disjuncts) == 0:
            raise GDP_Error("Disjunction %s is empty. This is " 
                            "likely indicative of a modeling error."  %
                            obj.getname(fully_qualified=True,
                                        name_buffer=NAME_BUFFER))
        for disjunct in obj.disjuncts:
            or_expr += disjunct.indicator_var
            # make suffix list. (We don't need it until we are
            # transforming constraints, but it gets created at the
            # disjunct level, so more efficient to make it here and
            # pass it down.)
            suffix_list = self._get_bigm_suffix_list(disjunct)
            # relax the disjunct
            self._transform_disjunct(disjunct, transBlock, bigM, suffix_list)

        # add or (or xor) constraint
        if xor:
            xorConstraint.add(index, (or_expr, 1))
        else:
            xorConstraint.add(index, (1, or_expr, None))
        # Mark the DisjunctionData as transformed by mapping it to its XOR
        # constraint.
        obj._algebraic_constraint = weakref_ref(xorConstraint[index])
        
        # and deactivate for the writers
        obj.deactivate()

    def _transform_disjunct(self, obj, transBlock, bigM, suffix_list):
        # deactivated -> either we've already transformed or user deactivated
        if not obj.active:
            if obj.indicator_var.is_fixed():
                if value(obj.indicator_var) == 0:
                    # The user cleanly deactivated the disjunct: there
                    # is nothing for us to do here.
                    return
                else:
                    raise GDP_Error(
                        "The disjunct %s is deactivated, but the "
                        "indicator_var is fixed to %s. This makes no sense."
                        % ( obj.name, value(obj.indicator_var) ))
            if obj._transformation_block is None:
                raise GDP_Error(
                    "The disjunct %s is deactivated, but the "
                    "indicator_var is not fixed and the disjunct does not "
                    "appear to have been relaxed. This makes no sense. "
                    "(If the intent is to deactivate the disjunct, fix its "
                    "indicator_var to 0.)"
                    % ( obj.name, ))
                
        if not obj._transformation_block is None:
            # we've transformed it, which means this is the second time it's
            # appearing in a Disjunction
            raise GDP_Error(
                    "The disjunct %s has been transformed, but a disjunction "
                    "it appears in has not. Putting the same disjunct in "
                    "multiple disjunctions is not supported." % obj.name)

        # add reference to original disjunct on transformation block
        relaxedDisjuncts = transBlock.relaxedDisjuncts
        relaxationBlock = relaxedDisjuncts[len(relaxedDisjuncts)]
        obj._transformation_block = weakref_ref(relaxationBlock)
        relaxationBlock._srcDisjunct = weakref_ref(obj)

        # This is crazy, but if the disjunction has been previously
        # relaxed, the disjunct *could* be deactivated.  This is a big
        # deal for CHull, as it uses the component_objects /
        # component_data_objects generators.  For BigM, that is OK,
        # because we never use those generators with active=True.  I am
        # only noting it here for the future when someone (me?) is
        # comparing the two relaxations.
        #
        # Transform each component within this disjunct
        self._transform_block_components(obj, obj, bigM, suffix_list)

        # deactivate disjunct to keep the writers happy
        obj._deactivate_without_fixing_indicator()

    def _transform_block_components(self, block, disjunct, bigM, suffix_list):
        # We first need to find any transformed disjunctions that might be here
        # because we need to move their transformation blocks up onto the parent
        # block before we transform anything else on this block 
        destinationBlock = disjunct._transformation_block().parent_block()
        for obj in block.component_data_objects(
                Disjunction, 
                sort=SortComponents.deterministic, 
                descend_into=(Block)):
            if not obj.algebraic_constraint:
                # This could be bad if it's active since that means its
                # untransformed, but we'll wait to yell until the next loop
                continue
            # get this disjunction's relaxation block.
            transBlock = None
            for d in obj.disjuncts:
                # Check if d is transformed
                if not d._transformation_block is None:
                    transBlock = d._transformation_block().parent_block()
                    # We found it, no need to keep looking
                    break
            if transBlock is None:
                raise GDP_Error(
                    "Found transformed disjunction %s on disjunct %s, "
                    "but none of its disjuncts have been transformed. "
                    "This is very strange." % (obj.name, disjunct.name))
            # move transBlock up to parent component
            self._transfer_transBlock_data(transBlock, destinationBlock)
            # delete the entire transformed disjunct container:
            del transBlock

        # Now look through the component map of block and transform
        # everything we have a handler for. Yell if we don't know how
        # to handle it.
        for name, obj in list(iteritems(block.component_map())):
            # This means non-ActiveComponent types cannot have handlers
            if not hasattr(obj, 'active') or not obj.active:
                continue
            handler = self.handlers.get(obj.type(), None)
            if not handler:
                if handler is None:
                    raise GDP_Error(
                        "No BigM transformation handler registered "
                        "for modeling components of type %s. If your " 
                        "disjuncts contain non-GDP Pyomo components that "
                        "require transformation, please transform them first."
                        % obj.type())
                continue
            # obj is what we are transforming, we pass disjunct
            # through so that we will have access to the indicator
            # variables down the line.
            handler(obj, disjunct, bigM, suffix_list)

    def _transfer_transBlock_data(self, fromBlock, toBlock):
        # We know that we have a list of transformed disjuncts on both. We need
        # to move those over. Then there might be constraints on the block also
        # (at this point only the disaggregation constraints from chull,
        # but... I'll leave it general for now.)
        disjunctList = toBlock.relaxedDisjuncts
        for idx, disjunctBlock in iteritems(fromBlock.relaxedDisjuncts):
            # I think this should work when #1106 is resolved:
            # disjunctList[len(disjunctList)] = disjunctBlock
            # newblock = disjunctList[len(disjunctList)-1]

            # HACK in the meantime:
            newblock = disjunctList[len(disjunctList)]
            self._copy_to_block(disjunctBlock, newblock)

            # update the mappings
            original = disjunctBlock._srcDisjunct()
            original._transformation_block = weakref_ref(newblock)
            newblock._srcDisjunct = weakref_ref(original)

            # Note that we could handle other components here if we ever needed
            # to, but we control what is on the transformation block and
            # currently everything is on the blocks that we just moved...

    def _copy_to_block(self, oldblock, newblock):
        for obj in oldblock.component_objects(Constraint):
            # [ESJ 07/18/2019] This shouldn't actually matter because we are
            # deleting the whole old block anyway, but it is to keep pyomo from
            # getting upset about the same component living on multiple blocks
            oldblock.del_component(obj)
            newblock.add_component(obj.getname(fully_qualified=True,
                                               name_buffer=NAME_BUFFER), obj)

    def _warn_for_active_disjunction(self, disjunction, disjunct, bigMargs,
                                     suffix_list):
        # this should only have gotten called if the disjunction is active
        assert disjunction.active
        problemdisj = disjunction
        if disjunction.is_indexed():
            for i in sorted(iterkeys(disjunction)):
                if disjunction[i].active:
                    # a _DisjunctionData is active, we will yell about
                    # it specifically.
                    problemdisj = disjunction[i]
                    break

        parentblock = problemdisj.parent_block()
        # the disjunction should only have been active if it wasn't transformed
        assert problemdisj.algebraic_constraint is None
        _probDisjName = problemdisj.getname(
            fully_qualified=True, name_buffer=NAME_BUFFER)
        raise GDP_Error("Found untransformed disjunction %s in disjunct %s! "
                        "The disjunction must be transformed before the "
                        "disjunct. If you are using targets, put the "
                        "disjunction before the disjunct in the list."
                        % (_probDisjName, disjunct.name))

    def _warn_for_active_disjunct(self, innerdisjunct, outerdisjunct, bigMargs,
                                  suffix_list):
        assert innerdisjunct.active
        problemdisj = innerdisjunct
        if innerdisjunct.is_indexed():
            for i in sorted(iterkeys(innerdisjunct)):
                if innerdisjunct[i].active:
                    # This is shouldn't be true, we will complain about it.
                    problemdisj = innerdisjunct[i]
                    break

        raise GDP_Error("Found active disjunct {0} in disjunct {1}! "
                        "Either {0} "
                        "is not in a disjunction or the disjunction it is in "
                        "has not been transformed. "
                        "{0} needs to be deactivated "
                        "or its disjunction transformed before {1} can be "
                        "transformed.".format(problemdisj.name,
                                              outerdisjunct.name))

    def _transform_block_on_disjunct(self, block, disjunct, bigMargs,
                                     suffix_list):
        # We look through everything on the component map of the block
        # and transform it just as we would if it was on the disjunct
        # directly.  (We are passing the disjunct through so that when
        # we find constraints, _xform_constraint will have access to
        # the correct indicator variable.)
        for i in sorted(iterkeys(block)):
            self._transform_block_components( block[i], disjunct, bigMargs,
                                              suffix_list)

    def _get_constraint_map_dict(self, transBlock):
        if not hasattr(transBlock, "_constraintMap"):
            transBlock._constraintMap = {
                'srcConstraints': ComponentMap(),
                'transformedConstraints': ComponentMap()}
        return transBlock._constraintMap

    def _transform_constraint(self, obj, disjunct, bigMargs,
                              suffix_list):
        # add constraint to the transformation block, we'll transform it there.
        transBlock = disjunct._transformation_block()
        constraintMap = self._get_constraint_map_dict(transBlock)
        
        disjunctionRelaxationBlock = transBlock.parent_block()
        # Though rare, it is possible to get naming conflicts here
        # since constraints from all blocks are getting moved onto the
        # same block. So we get a unique name
        cons_name = obj.getname(fully_qualified=True, name_buffer=NAME_BUFFER)
        name = unique_component_name(transBlock, cons_name)

        if obj.is_indexed():
            try:
                newConstraint = Constraint(obj.index_set(),
                                           disjunctionRelaxationBlock.lbub)
            except TypeError:
                # The original constraint may have been indexed by a
                # non-concrete set (like an Any).  We will give up on
                # strict index verification and just blindly proceed.
                newConstraint = Constraint(Any)
        else:
            newConstraint = Constraint(disjunctionRelaxationBlock.lbub)
        transBlock.add_component(name, newConstraint)
        # add mapping of original constraint to transformed constraint
        constraintMap['srcConstraints'][newConstraint] = obj
        constraintMap['transformedConstraints'][obj] = newConstraint

        for i in sorted(iterkeys(obj)):
            c = obj[i]
            if not c.active:
                continue

            # first, we see if an M value was specified in the arguments.
            # (This returns None if not)
            M = self._get_M_from_args(c, bigMargs)

            if __debug__ and logger.isEnabledFor(logging.DEBUG):
                _name = obj.getname(
                    fully_qualified=True, name_buffer=NAME_BUFFER)
                logger.debug("GDP(BigM): The value for M for constraint %s "
                             "from the BigM argument is %s." % (cons_name,
                                                                str(M)))

            # if we didn't get something from args, try suffixes:
            if M is None:
                M = self._get_M_from_suffixes(c, suffix_list)

            if __debug__ and logger.isEnabledFor(logging.DEBUG):
                _name = obj.getname(
                    fully_qualified=True, name_buffer=NAME_BUFFER)
                logger.debug("GDP(BigM): The value for M for constraint %s "
                             "after checking suffixes is %s." % (cons_name,
                                                                 str(M)))

            if not isinstance(M, (tuple, list)):
                if M is None:
                    M = (None, None)
                else:
                    try:
                        M = (-M, M)
                    except:
                        logger.error("Error converting scalar M-value %s "
                                     "to (-M,M).  Is %s not a numeric type?"
                                     % (M, type(M)))
                        raise
            if len(M) != 2:
                raise GDP_Error("Big-M %s for constraint %s is not of "
                                "length two. "
                                "Expected either a single value or "
                                "tuple or list of length two for M."
                                % (str(M), name))

            if c.lower is not None and M[0] is None:
                M = (self._estimate_M(c.body, name)[0] - c.lower, M[1])
            if c.upper is not None and M[1] is None:
                M = (M[0], self._estimate_M(c.body, name)[1] - c.upper)

            if __debug__ and logger.isEnabledFor(logging.DEBUG):
                _name = obj.getname(
                    fully_qualified=True, name_buffer=NAME_BUFFER)
                logger.debug("GDP(BigM): The value for M for constraint %s "
                             "after estimating (if needed) is %s." %
                             (cons_name, str(M)))

            # Handle indices for both SimpleConstraint and IndexedConstraint
            if i.__class__ is tuple:
                i_lb = i + ('lb',)
                i_ub = i + ('ub',)
            elif obj.is_indexed():
                i_lb = (i, 'lb',)
                i_ub = (i, 'ub',)
            else:
                i_lb = 'lb'
                i_ub = 'ub'

            if c.lower is not None:
                if M[0] is None:
                    raise GDP_Error("Cannot relax disjunctive constraint %s "
                                    "because M is not defined." % name)
                M_expr = M[0] * (1 - disjunct.indicator_var)
                newConstraint.add(i_lb, c.lower <= c. body - M_expr)
            if c.upper is not None:
                if M[1] is None:
                    raise GDP_Error("Cannot relax disjunctive constraint %s "
                                    "because M is not defined." % name)
                M_expr = M[1] * (1 - disjunct.indicator_var)
                newConstraint.add(i_ub, c.body - M_expr <= c.upper)
            # deactivate because we relaxed
            c.deactivate()

    def _get_M_from_args(self, constraint, bigMargs):
        # check args: we only have to look for constraint, constraintdata, and
        # None
        if bigMargs is None:
            return None

        parent = constraint.parent_component()
        if constraint in bigMargs:
            return bigMargs[constraint]
        elif parent in bigMargs:
            return bigMargs[parent]

        # [ESJ 08/22/2019] We apparently never actually check what is in
        # bigMargs... So I'll just yell about CUIDs if we find them here.
        deprecation_msg = ("In the future the bigM argument will no longer "
                           "allow ComponentUIDs as keys. Keys should be "
                           "constraints (in either a dict or ComponentMap)")
        cuid = ComponentUID(constraint)
        parentcuid = ComponentUID(constraint.parent_component())
        if cuid in bigMargs:
            deprecation_warning(deprecation_msg)
            return bigMargs[cuid]
        elif parentcuid in bigMargs:
            deprecation_warning(deprecation_msg)
            return bigMargs[parentcuid]
        elif None in bigMargs:
            return bigMargs[None]
        return None

    def _get_M_from_suffixes(self, constraint, suffix_list):
        M = None
        # first we check if the constraint or its parent is a key in any of the
        # suffix lists
        for bigm in suffix_list:
            if constraint in bigm:
                M = bigm[constraint]
                break

            # if c is indexed, check for the parent component
            if constraint.parent_component() in bigm:
                M = bigm[constraint.parent_component()]
                break

        # if we didn't get an M that way, traverse upwards through the blocks
        # and see if None has a value on any of them.
        if M is None:
            for bigm in suffix_list:
                if None in bigm:
                    M = bigm[None]
                    break
        return M

    def _estimate_M(self, expr, name):
        # Calculate a best guess at M
        repn = generate_standard_repn(expr)
        M = [0, 0]

        if not repn.is_nonlinear():
            if repn.constant is not None:
                for i in (0, 1):
                    if M[i] is not None:
                        M[i] += repn.constant

            for i, coef in enumerate(repn.linear_coefs or []):
                var = repn.linear_vars[i]
                bounds = (value(var.lb), value(var.ub))
                for i in (0, 1):
                    # reverse the bounds if the coefficient is negative
                    if coef > 0:
                        j = i
                    else:
                        j = 1 - i

                    if bounds[i] is not None:
                        M[j] += value(bounds[i]) * coef
                    else:
                        raise GDP_Error(
                            "Cannot estimate M for "
                            "expressions with unbounded variables."
                            "\n\t(found unbounded var %s while processing "
                            "constraint %s)" % (var.name, name))
        else:
            # expression is nonlinear. Try using `contrib.fbbt` to estimate.
            expr_lb, expr_ub = compute_bounds_on_expr(expr)
            if expr_lb == -inf or expr_ub == inf:
                raise GDP_Error("Cannot estimate M for unbounded nonlinear "
                                "expressions.\n\t(found while processing "
                                "constraint %s)" % name)
            else:
                M = (expr_lb, expr_ub)

        return tuple(M)

    # These are all functions to retrieve transformed components from original
    # ones and vice versa.
    def get_src_disjunct(self, transBlock):
        if not hasattr(transBlock, '_srcDisjunct') or \
           not type(transBlock._srcDisjunct) is weakref_ref:
            raise GDP_Error("Block %s doesn't appear to be a transformation "
                            "block for a disjunct. No source disjunct found." 
                            % transBlock.name)
        return transBlock._srcDisjunct()

    def get_src_constraint(self, transformedConstraint):
        transBlock = transformedConstraint.parent_block()
        # This should be our block, so if it's not, the user messed up and gave
        # us the wrong thing. If they happen to also have a _constraintMap then
        # the world is really against us.
        if not hasattr(transBlock, "_constraintMap"):
            raise GDP_Error("Constraint %s is not a transformed constraint" 
                            % transformedConstraint.name)
        return transBlock._constraintMap['srcConstraints'][transformedConstraint]

    def get_transformed_constraint(self, srcConstraint):
        # We are going to have to traverse up until we find the disjunct this
        # constraint lives on
        parent = srcConstraint.parent_block()
        # [ESJ 08/06/2019] I actually don't know how to do this prettily...
        while not type(parent) in (_DisjunctData, SimpleDisjunct):
            parent = parent.parent_block()
            if parent is None:
                raise GDP_Error(
                    "Constraint %s is not on a disjunct and so was not "
                    "transformed" % srcConstraint.name)
        transBlock = parent._transformation_block
        if transBlock is None:
            raise GDP_Error("Constraint %s is on a disjunct which has not been "
                            "transformed" % srcConstraint.name)
        # if it's not None, it's the weakref we wanted.
        transBlock = transBlock()
        if hasattr(transBlock, "_constraintMap") and transBlock._constraintMap[
                'transformedConstraints'].get(srcConstraint):
            return transBlock._constraintMap['transformedConstraints'][
                srcConstraint]
        raise GDP_Error("Constraint %s has not been transformed." 
                        % srcConstraint.name)

    def get_src_disjunction(self, xor_constraint):
        m = xor_constraint.model()
        for disjunction in m.component_data_objects(Disjunction):
            if disjunction._algebraic_constraint:
                if disjunction._algebraic_constraint() is xor_constraint:
                    return disjunction
        raise GDP_Error("It appears that %s is not an XOR or OR constraint "
                        "resulting from transforming a Disjunction."
                        % xor_constraint.name)
