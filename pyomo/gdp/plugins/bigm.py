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

from pyomo.core import (
    Block, Connector, Constraint, Param, Set, Suffix, Var,
    Expression, SortComponents, TraversalStrategy, Any, value
)
from pyomo.core.base import Transformation, TransformationFactory
from pyomo.core.base.component import ComponentUID, ActiveComponent
from pyomo.core.kernel.component_map import ComponentMap
from pyomo.core.kernel.component_set import ComponentSet
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.util import target_list
from pyomo.gdp.plugins.gdp_var_mover import HACK_GDP_Disjunct_Reclassifier
from pyomo.repn import generate_standard_repn
from pyomo.common.config import ConfigBlock, ConfigValue
from pyomo.common.modeling import unique_component_name
from six import iterkeys, iteritems

logger = logging.getLogger('pyomo.gdp.bigm')


def _to_dict(val):
    if val is None:
        return val
    if isinstance(val, dict):
        return val
    return {None: val}


@TransformationFactory.register('gdp.bigm', doc="Relax disjunctive model using big-M terms.")
class BigM_Transformation(Transformation):
    """Relax disjunctive model using big-M terms.

    Relaxes a disjunctive model into an algebraic model by adding Big-M
    terms to all disjunctive constraints.

    This transformation accepts the following keyword arguments:
        bigM: A user-specified value (or dict) of M values to use (see below)
        targets: the targets to transform [default: the instance]

    M values are determined as follows:
       1) if the constraint CUID appears in the bigM argument dict
       2) if the constraint parent_component CUID appears in the bigM
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

    After transformation, every transformed disjunct will have a
    "_gdp_transformation_info" dict containing 2 entries:

        'relaxed': True,
        'bigm': {
            'relaxationBlock': <block>,
            'relaxedConstraints': ComponentMap(constraint: relaxed_constraint)
        }

    In addition, any block or disjunct containing a relaxed disjunction
    will have a "_gdp_transformation_info" dict with the following
    entry:

        'disjunction_or_constraint': <constraint>

    Finally, the transformation will create a new Block with a unique
    name beginning "_pyomo_gdp_bigm_relaxation".  That Block will
    contain an indexed Block named "relaxedDisjuncts", which will hold
    the relaxed disjuncts.  This block is indexed by an integer
    indicating the order in which the disjuncts were relaxed.  Each
    block will have a "_gdp_transformation_info" dict with the following
    entries:

        'src': <source disjunct>
        'srcConstraints': ComponentMap(relaxed_constraint: constraint)
    """

    CONFIG = ConfigBlock("gdp.bigm")
    CONFIG.declare('targets', ConfigValue(
        default=None,
        domain=target_list,
        description="target or list of targets that will be relaxed",
        doc="""

        This specifies the list of targets to relax as either a
        component, ComponentUID, or string that can be passed to a
        ComponentUID; or an iterable of these types.  If None (default),
        the entire model is transformed."""
    ))
    CONFIG.declare('bigM', ConfigValue(
        default=None,
        domain=_to_dict,
        description="Big-M value used for constraint relaxation",
        doc="""

        A user-specified value (or dict) of M values that override
        M-values found through model Suffixes or that would otherwise be
        calculated using variable domains."""
    ))

    def __init__(self):
        """Initialize transformation object."""
        super(BigM_Transformation, self).__init__()
        self.handlers = {
            Constraint:  self._xform_constraint,
            Var:         False,
            Connector:   False,
            Expression:  False,
            Suffix:      False,
            Param:       False,
            Set:         False,
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

        # make a transformation block to put transformed disjuncts on
        transBlockName = unique_component_name(
            instance,
            '_pyomo_gdp_bigm_relaxation')
        transBlock = Block()
        instance.add_component(transBlockName, transBlock)
        transBlock.relaxedDisjuncts = Block(Any)
        transBlock.lbub = Set(initialize=['lb', 'ub'])
        # this is a dictionary for keeping track of IndexedDisjuncts
        # and IndexedDisjunctions so that, at the end of the
        # transformation, we can check that the ones with no active
        # DisjstuffDatas are deactivated.
        transBlock.disjContainers = ComponentSet()

        targets = config.targets
        if targets is None:
            targets = (instance, )
            _HACK_transform_whole_instance = True
        else:
            _HACK_transform_whole_instance = False
        for _t in targets:
            t = _t.find_component(instance)
            if t is None:
                raise GDP_Error(
                    "Target %s is not a component on the instance!" % _t)

            if t.type() is Disjunction:
                if t.parent_component() is t:
                    self._transformDisjunction(t, transBlock, bigM)
                else:
                    self._transformDisjunctionData(
                        t, transBlock, bigM, t.index())
            elif t.type() in (Block, Disjunct):
                if t.parent_component() is t:
                    self._transformBlock(t, transBlock, bigM)
                else:
                    self._transformBlockData(t, transBlock, bigM)
            else:
                raise GDP_Error(
                    "Target %s was not a Block, Disjunct, or Disjunction. "
                    "It was of type %s and can't be transformed."
                    % (t.name, type(t)))
        # Go through our dictionary of indexed things and deactivate
        # the containers that don't have any active guys inside of
        # them. So the invalid component logic will tell us if we
        # missed something getting transformed.
        for obj in transBlock.disjContainers:
            if not obj.active:
                continue
            for i in obj:
                if obj[i].active:
                    break
            else:
                # HACK due to active flag implementation.
                #
                # Ideally we would not have to do any of this (an
                # ActiveIndexedComponent would get its active status by
                # querring the active status of all the contained Data
                # objects).  As a fallback, we would like to call:
                #
                #    obj._deactivate_without_fixing_indicator()
                #
                # However, the sreaightforward implementation of that
                # method would have unintended side effects (fixing the
                # contained _DisjunctData's indicator_vars!) due to our
                # class hierarchy.  Instead, we will directly call the
                # relevant base class (safe-ish since we are verifying
                # that all the contained _DisjunctionData are
                # deactivated directly above).
                ActiveComponent.deactivate(obj)

        # HACK for backwards compatibility with the older GDP transformations
        #
        # Until the writers are updated to find variables on things
        # other than active blocks, we need to reclassify the Disjuncts
        # as Blocks after transformation so that the writer will pick up
        # all the variables that it needs (in this case, indicator_vars).
        if _HACK_transform_whole_instance:
            HACK_GDP_Disjunct_Reclassifier().apply_to(instance)


    def _transformBlock(self, obj, transBlock, bigM):
        for i in sorted(iterkeys(obj)):
            self._transformBlockData(obj[i], transBlock, bigM)

    def _transformBlockData(self, obj, transBlock, bigM):
        # Transform every (active) disjunction in the block
        for disjunction in obj.component_objects(
                Disjunction,
                active=True,
                sort=SortComponents.deterministic,
                descend_into=(Block, Disjunct),
                descent_order=TraversalStrategy.PostfixDFS):
            self._transformDisjunction(disjunction, transBlock, bigM)

    def _getXorConstraint(self, disjunction):
        # Put the disjunction constraint on its parent block and
        # determine whether it is an OR or XOR constraint.

        # We never do this for just a DisjunctionData because we need
        # to know about the index set of its parent component. So if
        # we called this on a DisjunctionData, we did something wrong.
        assert isinstance(disjunction, Disjunction)
        parent = disjunction.parent_block()
        if hasattr(parent, "_gdp_transformation_info"):
            infodict = parent._gdp_transformation_info
            if type(infodict) is not dict:
                raise GDP_Error(
                    "Component %s contains an attribute named "
                    "_gdp_transformation_info. The transformation requires "
                    "that it can create this attribute!" % parent.name)
            try:
                # On the off-chance that another GDP transformation went
                # first, the infodict may exist, but the specific map we
                # want will not be present
                orConstraintMap = infodict['disjunction_or_constraint']
            except KeyError:
                orConstraintMap = infodict['disjunction_or_constraint'] \
                                  = ComponentMap()
        else:
            infodict = parent._gdp_transformation_info = {}
            orConstraintMap = infodict['disjunction_or_constraint'] \
                              = ComponentMap()

        # If the Constraint already exists, return it
        if disjunction in orConstraintMap:
            return orConstraintMap[disjunction]

        # add the XOR (or OR) constraints to parent block (with unique name)
        # It's indexed if this is an IndexedDisjunction, not otherwise
        orC = Constraint(disjunction.index_set()) if \
            disjunction.is_indexed() else Constraint()
        # The name used to indicate if thee were OR or XOR disjunctions,
        # however now that Disjunctions ae allowed to mix the state we
        # can no longer make that distinction in the name.
        #    nm = '_xor' if xor else '_or'
        nm = '_xor'
        orCname = unique_component_name(parent, '_gdp_bigm_relaxation_' +
                                        disjunction.name + nm)
        parent.add_component(orCname, orC)
        orConstraintMap[disjunction] = orC
        return orC

    def _transformDisjunction(self, obj, transBlock, bigM):
        # relax each of the disjunctionDatas
        for i in sorted(iterkeys(obj)):
            self._transformDisjunctionData(obj[i], transBlock, bigM, i)

        # deactivate so we know we relaxed
        obj.deactivate()

    def _transformDisjunctionData(self, obj, transBlock, bigM, index):
        if not obj.active:
            return  # Do not process a deactivated disjunction
        parent_component = obj.parent_component()
        transBlock.disjContainers.add(parent_component)
        orConstraint = self._getXorConstraint(parent_component)

        xor = obj.xor
        or_expr = 0
        for disjunct in obj.disjuncts:
            or_expr += disjunct.indicator_var
            # make suffix list. (We don't need it until we are
            # transforming constraints, but it gets created at the
            # disjunct level, so more efficient to make it here and
            # pass it down.
            suffix_list = self._get_bigm_suffix_list(disjunct)
            # relax the disjunct
            self._bigM_relax_disjunct(disjunct, transBlock, bigM, suffix_list)
        # add or (or xor) constraint
        if xor:
            orConstraint.add(index, (or_expr, 1))
        else:
            orConstraint.add(index, (1, or_expr, None))
        obj.deactivate()

    def _bigM_relax_disjunct(self, obj, transBlock, bigM, suffix_list):
        if hasattr(obj, "_gdp_transformation_info"):
            infodict = obj._gdp_transformation_info
            # If the user has something with our name that is not a dict, we
            # scream. If they have a dict with this name then we are just going
            # to use it...
            if type(infodict) is not dict:
                raise GDP_Error(
                    "Disjunct %s contains an attribute named "
                    "_gdp_transformation_info. The transformation requires "
                    "that it can create this attribute!" % obj.name)
        else:
            infodict = obj._gdp_transformation_info = {}

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
            if not infodict.get('relaxed', False):
                raise GDP_Error(
                    "The disjunct %s is deactivated, but the "
                    "indicator_var is not fixed and the disjunct does not "
                    "appear to have been relaxed. This makes no sense."
                    % ( obj.name, ))

        if 'bigm' in infodict:
            # we've transformed it (with BigM), so don't do it again.
            return

        # add reference to original disjunct to info dict on transformation
        # block
        relaxedDisjuncts = transBlock.relaxedDisjuncts
        relaxationBlock = relaxedDisjuncts[len(relaxedDisjuncts)]
        relaxationBlock._gdp_transformation_info = {
            'src': obj,
            'srcConstraints': ComponentMap(),
        }

        # add reference to transformation block on original disjunct
        assert 'bigm' not in infodict
        infodict['bigm'] = {
            'relaxationBlock': relaxationBlock,
            'relaxedConstraints': ComponentMap()
        }

        # if this is a disjunctData from an indexed disjunct, we are
        # going to want to check at the end that the container is
        # deactivated if everything in it is. So we save it in our
        # dictionary of things to check if it isn't there already.
        disjParent = obj.parent_component()
        if disjParent.is_indexed() and \
           disjParent not in transBlock.disjContainers:
            transBlock.disjContainers.add(disjParent)

        # This is crazy, but if the disjunction has been previously
        # relaxed, the disjunct *could* be deactivated.  This is a big
        # deal for CHull, as it uses the component_objects /
        # component_data_objects generators.  For BigM, that is OK,
        # because we never use those generators with active=True.  I am
        # only noting it here for the future when someone (me?) is
        # comparing the two relaxations.
        #
        # Transform each component within this disjunct
        self._transform_block_components(obj, obj, infodict, bigM, suffix_list)

        # deactivate disjunct so we know we've relaxed it
        obj._deactivate_without_fixing_indicator()
        infodict['relaxed'] = True

    def _transform_block_components(self, block, disjunct, infodict,
                                    bigM, suffix_list):
        # Look through the component map of block and transform
        # everything we have a handler for. Yell if we don't know how
        # to handle it.
        for name, obj in list(iteritems(block.component_map())):
            if hasattr(obj, 'active') and not obj.active:
                continue
            handler = self.handlers.get(obj.type(), None)
            if not handler:
                if handler is None:
                    raise GDP_Error(
                        "No BigM transformation handler registered "
                        "for modeling components of type %s" % obj.type())
                continue
            # obj is what we are transforming, we pass disjunct
            # through so that we will have access to the indicator
            # variables down the line.
            handler(obj, disjunct, infodict, bigM, suffix_list)

    def _warn_for_active_disjunction(self, disjunction, disjunct, infodict,
                                     bigMargs, suffix_list):
        # this should only have gotten called if the disjunction is active
        assert disjunction.active
        problemdisj = disjunction
        if disjunction.is_indexed():
            for i in disjunction:
                if disjunction[i].active:
                    # a _DisjunctionData is active, we will yell about
                    # it specifically.
                    problemdisj = disjunction[i]
                    break
            # None of the _DisjunctionDatas were actually active. We
            # are OK and we can deactivate the container.
            else:
                disjunction.deactivate()
                return
        parentblock = problemdisj.parent_block()
        # the disjunction should only have been active if it wasn't transformed
        assert (not hasattr(parentblock, "_gdp_transformation_info")) or \
            problemdisj.name not in parentblock._gdp_transformation_info
        raise GDP_Error("Found untransformed disjunction %s in disjunct %s! "
                        "The disjunction must be transformed before the "
                        "disjunct. If you are using targets, put the "
                        "disjunction before the disjunct in the list."
                        % (problemdisj.name, disjunct.name))

    def _warn_for_active_disjunct(self, innerdisjunct, outerdisjunct,
                                  infodict, bigMargs, suffix_list):
        assert innerdisjunct.active
        problemdisj = innerdisjunct
        if innerdisjunct.is_indexed():
            for i in innerdisjunct:
                if innerdisjunct[i].active:
                    # This is shouldn't be true, we will complain about it.
                    problemdisj = innerdisjunct[i]
                    break
            # None of the _DisjunctDatas were actually active, so we
            # are fine and we can deactivate the container.
            else:
                # HACK: See above about _deactivate_without_fixing_indicator
                ActiveComponent.deactivate(innerdisjunct)
                return
        raise GDP_Error("Found active disjunct {0} in disjunct {1}! "
                        "Either {0} "
                        "is not in a disjunction or the disjunction it is in "
                        "has not been transformed. "
                        "{0} needs to be deactivated "
                        "or its disjunction transformed before {1} can be "
                        "transformed.".format(problemdisj.name,
                                              outerdisjunct.name))

    def _transform_block_on_disjunct(self, block, disjunct, infodict,
                                     bigMargs, suffix_list):
        # We look through everything on the component map of the block
        # and transform it just as we would if it was on the disjunct
        # directly.  (We are passing the disjunct through so that when
        # we find constraints, _xform_constraint will have access to
        # the correct indicator variable.
        for i in sorted(iterkeys(block)):
            self._transform_block_components(
                block[i], disjunct, infodict, bigMargs, suffix_list)

    def _xform_constraint(self, obj, disjunct, infodict,
                          bigMargs, suffix_list):
        # add constraint to the transformation block, we'll transform it there.

        relaxationBlock = infodict['bigm']['relaxationBlock']
        transBlock = relaxationBlock.parent_block()
        # Though rare, it is possible to get naming conflicts here
        # since constraints from all blocks are getting moved onto the
        # same block. So we get a unique name
        name = unique_component_name(relaxationBlock, obj.name)

        if obj.is_indexed():
            try:
                newConstraint = Constraint(obj.index_set(), transBlock.lbub)
            except TypeError:
                # The original constraint may have been indexed by a
                # non-concrete set (like an Any).  We will give up on
                # strict index verification and just blindly proceed.
                newConstraint = Constraint(Any)
        else:
            newConstraint = Constraint(transBlock.lbub)
        relaxationBlock.add_component(name, newConstraint)
        # add mapping of original constraint to transformed constraint
        # in transformation info dictionary
        infodict['bigm']['relaxedConstraints'][obj] = newConstraint
        # add mapping of transformed constraint back to original constraint (we
        # know that the info dict is already created because this only got
        # called if we were transforming a disjunct...)
        relaxationBlock._gdp_transformation_info['srcConstraints'][
            newConstraint] = obj

        for i in sorted(iterkeys(obj)):
            c = obj[i]
            if not c.active:
                continue

            # first, we see if an M value was specified in the arguments.
            # (This returns None if not)
            M = self._get_M_from_args(c, bigMargs)

            if __debug__ and logger.isEnabledFor(logging.DEBUG):
                logger.debug("GDP(BigM): The value for M for constraint %s "
                             "from the BigM argument is %s." % (obj.name,
                                                                str(M)))

            # if we didn't get something from args, try suffixes:
            if M is None:
                M = self._get_M_from_suffixes(c, suffix_list)

            if __debug__ and logger.isEnabledFor(logging.DEBUG):
                logger.debug("GDP(BigM): The value for M for constraint %s "
                             "after checking suffixes is %s." % (obj.name,
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
                logger.debug("GDP(BigM): The value for M for constraint %s "
                             "after estimating (if needed) is %s." %
                             (obj.name, str(M)))

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

    def _get_M_from_args(self, constraint, bigMargs):
        # check args: we only have to look for constraint, constraintdata, and
        # None
        if bigMargs is None:
            return None

        cuid = ComponentUID(constraint)
        parentcuid = ComponentUID(constraint.parent_component())
        if cuid in bigMargs:
            return bigMargs[cuid]
        elif parentcuid in bigMargs:
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
            raise GDP_Error("Cannot estimate M for nonlinear "
                            "expressions.\n\t(found while processing "
                            "constraint %s)" % name)

        return tuple(M)
