#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import weakref
import logging
import textwrap

import pyomo.common.config as cfg
from pyomo.common.modeling import unique_component_name
from pyomo.core.expr.numvalue import native_numeric_types, ZeroConstant
from pyomo.core.expr import current as EXPR
from pyomo.core import *
from pyomo.core.base.block import SortComponents
from pyomo.core.base.component import ComponentUID, ActiveComponent
from pyomo.core.base import _ExpressionData
from pyomo.core.base.var import _VarData
from pyomo.core.kernel.component_map import ComponentMap
from pyomo.core.kernel.component_set import ComponentSet
import pyomo.core.expr.current as EXPR
from pyomo.core.base import Transformation, TransformationFactory
from pyomo.core import (
    Block, Connector, Constraint, Param, Set, Suffix, Var,
    Expression, SortComponents, TraversalStrategy,
    Any, Reals, value
)
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.util import clone_without_expression_components, target_list
from pyomo.gdp.plugins.gdp_var_mover import HACK_GDP_Disjunct_Reclassifier

from six import iteritems, iterkeys

# DEBUG
from nose.tools import set_trace

logger = logging.getLogger('pyomo.gdp.chull')


@TransformationFactory.register('gdp.chull', doc="Relax disjunctive model by forming the convex hull.")
class ConvexHull_Transformation(Transformation):
    """Relax disjunctive model by forming the convex hull.

    Relaxes a disjunctive model into an algebraic model by forming the
    convex hull of each disjunction.

    This transformation accepts the following keyword arguments:

    Parameters
    ----------
    perspective_function : str
        The perspective function used for the disaggregated variables.
        Must be one of 'FurmanSawayaGrossmann' (default),
        'LeeGrossmann', or 'GrossmannLee'
    EPS : float
        The value to use for epsilon [default: 1e-4]
    targets : (block, disjunction, ComponentUID or list of those types)
        The targets to transform. This can be a block, disjunction, or a
        list of blocks and Disjunctions [default: the instance]

    After transformation, every transformed disjunct will have a
    "_gdp_transformation_info" dict containing 2 entries:

        'relaxed': True,
        'chull': {
            'relaxationBlock': <block>,
            'relaxedConstraints': ComponentMap(constraint: relaxed_constraint)
            'disaggregatedVars': ComponentMap(var: list of disaggregated vars),
            'bigmConstraints': ComponentMap(disaggregated var: bigM constraint),
        }

    In addition, any block or disjunct containing a relaxed disjunction
    will have a "_gdp_transformation_info" dict with the following
    entry:

        'disjunction_or_constraint': <constraint>

    Finally, the transformation will create a new Block with a unique
    name beginning "_pyomo_gdp_chull_relaxation".  That Block will
    contain an indexed Block named "relaxedDisjuncts", which will hold
    the relaxed disjuncts.  This block is indexed by an integer
    indicating the order in which the disjuncts were relaxed.  Each
    block will have a "_gdp_transformation_info" dict with the following
    entries:

        'src': <source disjunct>
        'srcVars': ComponentMap(disaggregated var: original var),
        'srcConstraints': ComponentMap(relaxed_constraint: constraint)
        'boundConstraintToSrcVar': ComponentMap(bigm_constraint: orig_var),

    """


    CONFIG = cfg.ConfigBlock('gdp.chull')
    CONFIG.declare('targets', cfg.ConfigValue(
        default=None,
        domain=target_list,
        description="target or list of targets that will be relaxed",
        doc="""

        This specifies the target or list of targets to relax as either a
        component, ComponentUID, or string that can be passed to a
        ComponentUID; or an iterable of these types.  If None (default),
        the entire model is transformed."""
    ))
    CONFIG.declare('perspective function', cfg.ConfigValue(
        default='FurmanSawayaGrossmann',
        domain=cfg.In(['FurmanSawayaGrossmann','LeeGrossmann','GrossmannLee']),
        description='perspective function used for variable disaggregation',
        doc="""
        The perspective function used for variable disaggregation

        "LeeGrossmann" is the original NL convex hull from Lee &
        Grossmann (2000) [1]_, which substitutes nonlinear constraints

            h_ik(x) <= 0

        with

            x_k = sum( nu_ik )
            y_ik * h_ik( nu_ik/y_ik ) <= 0

        "GrossmannLee" is an updated formulation from Grossmann &
        Lee (2003) [2]_, which avoids divide-by-0 errors by using:

            x_k = sum( nu_ik )
            (y_ik + eps) * h_ik( nu_ik/(y_ik + eps) ) <= 0

        "FurmanSawayaGrossmann" (default) is an improved relaxation [3]_
        that is exact at 0 and 1 while avoiding numerical issues from
        the Lee & Grossmann formulation by using:

            x_k = sum( nu_ik )
            ((1-eps)*y_ik + eps) * h_ik( nu_ik/((1-eps)*y_ik + eps) ) \
                - eps * h_ki(0) * ( 1-y_ik ) <= 0

        References
        ----------
        .. [1] Lee, S., & Grossmann, I. E. (2000). New algorithms for
           nonlinear generalized disjunctive programming.  Computers and
           Chemical Engineering, 24, 2125-2141

        .. [2] Grossmann, I. E., & Lee, S. (2003). Generalized disjunctive
           programming: Nonlinear convex hull relaxation and algorithms.
           Computational Optimization and Applications, 26, 83-100.

        .. [3] Furman, K., Sawaya, N., and Grossmann, I.  A computationally
           useful algebraic representation of nonlinear disjunctive convex
           sets using the perspective function.  Optimization Online
           (2016). http://www.optimization-online.org/DB_HTML/2016/07/5544.html.
        """
    ))
    CONFIG.declare('EPS', cfg.ConfigValue(
        default=1e-4,
        domain=cfg.PositiveFloat,
        description="Epsilon value to use in perspective function",
    ))

    def __init__(self):
        super(ConvexHull_Transformation, self).__init__()
        self.handlers = {
            Constraint : self._xform_constraint,
            Var :        False,
            Connector :  False,
            Expression : False,
            Param :      False,
            Set :        False,
            Suffix :     False,
            Disjunction: self._warn_for_active_disjunction,
            Disjunct:    self._warn_for_active_disjunct,
            Block:       self._transform_block_on_disjunct,
            }


    def _apply_to(self, instance, **kwds):
        self._config = self.CONFIG(kwds.pop('options', {}))
        self._config.set_value(kwds)

        # make a transformation block
        transBlockName = unique_component_name(
            instance,
            '_pyomo_gdp_chull_relaxation')
        transBlock = Block()
        instance.add_component(transBlockName, transBlock)
        transBlock.relaxedDisjuncts = Block(Any)
        transBlock.lbub = Set(initialize = ['lb','ub','eq'])
        transBlock.disjContainers = ComponentSet()

        targets = self._config.targets
        if targets is None:
            targets = ( instance, )
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
                    self._transformDisjunction(t, transBlock)
                else:
                    self._transformDisjunctionData(t, transBlock, t.index())
            elif t.type() in (Block, Disjunct):
                if t.parent_component() is t:
                    self._transformBlock(t, transBlock)
                else:
                    self._transformBlockData(t, transBlock)
            else:
                raise GDP_Error(
                    "Target %s was not a Block, Disjunct, or Disjunction. "
                    "It was of type %s and can't be transformed"
                    % (t.name, type(t)) )

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

    def _contained_in(self, var, block):
        "Return True if a var is in the subtree rooted at block"
        while var is not None:
            if var.parent_component() is block:
                return True
            var = var.parent_block()
            if var is block:
                return True
        return False

    def _transformBlock(self, obj, transBlock):
        for i in sorted(iterkeys(obj)):
            self._transformBlockData(obj[i], transBlock)


    def _transformBlockData(self, obj, transBlock):
        # Transform every (active) disjunction in the block
        for disjunction in obj.component_objects(
                Disjunction,
                active=True,
                sort=SortComponents.deterministic,
                descend_into=(Block,Disjunct),
                descent_order=TraversalStrategy.PostfixDFS):
            self._transformDisjunction(disjunction, transBlock)


    def _getDisjunctionConstraints(self, disjunction):
        # Put the disjunction constraint on its parent block

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
            try:
                disaggregationConstraintMap = infodict[
                    'disjunction_disaggregation_constraints']
            except KeyError:
                disaggregationConstraintMap = infodict[
                    'disjunction_disaggregation_constraints'] \
                    = ComponentMap()
        else:
            infodict = parent._gdp_transformation_info = {}
            orConstraintMap = infodict['disjunction_or_constraint'] \
                              = ComponentMap()
            disaggregationConstraintMap = infodict[
                'disjunction_disaggregation_constraints'] \
                = ComponentMap()

        if disjunction in disaggregationConstraintMap:
            disaggregationConstraint = disaggregationConstraintMap[disjunction]
        else:
            # add the disaggregation constraint
            disaggregationConstraint \
                = disaggregationConstraintMap[disjunction] = Constraint(Any)
            parent.add_component(
                unique_component_name(parent, '_gdp_chull_relaxation_' + \
                                      disjunction.name + '_disaggregation'),
                disaggregationConstraint)

        # If the Constraint already exists, return it
        if disjunction in orConstraintMap:
            orC = orConstraintMap[disjunction]
        else:
            # add the XOR (or OR) constraints to parent block (with
            # unique name) It's indexed if this is an
            # IndexedDisjunction, not otherwise
            orC = Constraint(disjunction.index_set()) if \
                  disjunction.is_indexed() else Constraint()
            parent.add_component(
                unique_component_name(parent, '_gdp_chull_relaxation_' +
                                      disjunction.name + '_xor'),
                orC)
            orConstraintMap[disjunction] = orC

        return orC, disaggregationConstraint


    def _transformDisjunction(self, obj, transBlock):
        # create the disjunction constraint and disaggregation
        # constraints and then relax each of the disjunctionDatas
        for i in sorted(iterkeys(obj)):
            self._transformDisjunctionData(obj[i], transBlock, i)

        # deactivate so we know we relaxed
        obj.deactivate()


    def _transformDisjunctionData(self, obj, transBlock, index):
        # Convex hull doesn't work if this is an or constraint. So if
        # xor is false, give up
        if not obj.xor:
            raise GDP_Error("Cannot do convex hull transformation for "
                            "disjunction %s with or constraint. Must be an xor!"
                            % obj.name)

        parent_component = obj.parent_component()
        transBlock.disjContainers.add(parent_component)
        orConstraint, disaggregationConstraint \
            = self._getDisjunctionConstraints(parent_component)

        # We first go through and collect all the variables that we
        # are going to disaggregate.
        varOrder_set = ComponentSet()
        varOrder = []
        varsByDisjunct = ComponentMap()
        for disjunct in obj.disjuncts:
            # This is crazy, but if the disjunct has been previously
            # relaxed, the disjunct *could* be deactivated.
            not_active = not disjunct.active
            if not_active:
                disjunct._activate_without_unfixing_indicator()
            try:
                disjunctVars = varsByDisjunct[disjunct] = ComponentSet()
                for cons in disjunct.component_data_objects(
                        Constraint,
                        active = True,
                        sort=SortComponents.deterministic,
                        descend_into=Block):
                    # we aren't going to disaggregate fixed
                    # variables. This means there is trouble if they are
                    # unfixed later...
                    for var in EXPR.identify_variables(
                            cons.body, include_fixed=False):
                        # Note the use of a list so that we will
                        # eventually disaggregate the vars in a
                        # deterministic order (the order that we found
                        # them)
                        disjunctVars.add(var)
                        if var not in varOrder_set:
                            varOrder.append(var)
                            varOrder_set.add(var)
            finally:
                if not_active:
                    disjunct._deactivate_without_fixing_indicator()

        # We will only disaggregate variables that
        #  1) appear in multiple disjuncts, or
        #  2) are not contained in this disjunct, or
        #  3) are not themselves disaggregated variables
        varSet = []
        localVars = ComponentMap((d,[]) for d in obj.disjuncts)
        for var in varOrder:
            disjuncts = [d for d in varsByDisjunct if var in varsByDisjunct[d]]
            if len(disjuncts) > 1:
                varSet.append(var)
            elif self._contained_in(var, disjuncts[0]):
                localVars[disjuncts[0]].append(var)
            elif self._contained_in(var, transBlock):
                # There is nothing to do here: these are already
                # disaggregated vars that can/will be forced to 0 when
                # their disjunct is not active.
                pass
            else:
                varSet.append(var)

        # Now that we know who we need to disaggregate, we will do it
        # while we also transform the disjuncts.
        or_expr = 0
        for disjunct in obj.disjuncts:
            or_expr += disjunct.indicator_var
            self._transform_disjunct(disjunct, transBlock, varSet,
                                     localVars[disjunct])
        orConstraint.add(index, (or_expr, 1))

        for i, var in enumerate(varSet):
            disaggregatedExpr = 0
            for disjunct in obj.disjuncts:
                if 'chull' not in disjunct._gdp_transformation_info:
                    if not disjunct.indicator_var.is_fixed() \
                            or value(disjunct.indicator_var) != 0:
                        raise RuntimeError(
                            "GDP chull: disjunct was not relaxed, but "
                            "does not appear to be correctly deactivated.")
                    continue
                disaggregatedVar = disjunct._gdp_transformation_info['chull'][
                    'disaggregatedVars'][var]
                disaggregatedExpr += disaggregatedVar
            if type(index) is tuple:
                consIdx = index + (i,)
            elif parent_component.is_indexed():
                consIdx = (index,) + (i,)
            else:
                consIdx = i

            disaggregationConstraint.add(
                consIdx,
                var == disaggregatedExpr)


    def _transform_disjunct(self, obj, transBlock, varSet, localVars):
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
        # deactivated means either we've already transformed or user deactivated
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

        if 'chull' in infodict:
            # we've transformed it (with CHull), so don't do it again.
            return

        # add reference to original disjunct to info dict on
        # transformation block
        relaxedDisjuncts = transBlock.relaxedDisjuncts
        relaxationBlock = relaxedDisjuncts[len(relaxedDisjuncts)]
        relaxationBlockInfo = relaxationBlock._gdp_transformation_info = {
            'src': obj,
            'srcVars': ComponentMap(),
            'srcConstraints': ComponentMap(),
            'boundConstraintToSrcVar': ComponentMap(),
        }
        infodict['chull'] = chull = {
            'relaxationBlock': relaxationBlock,
            'relaxedConstraints': ComponentMap(),
            'disaggregatedVars': ComponentMap(),
            'bigmConstraints': ComponentMap(),
        }

        # if this is a disjunctData from an indexed disjunct, we are
        # going to want to check at the end that the container is
        # deactivated if everything in it is. So we save it in our
        # dictionary of things to check if it isn't there already.
        disjParent = obj.parent_component()
        if disjParent.is_indexed() and \
           disjParent not in transBlock.disjContainers:
            transBlock.disjContainers.add(disjParent)

        # add the disaggregated variables and their bigm constraints
        # to the relaxationBlock
        for var in varSet:
            lb = var.lb
            ub = var.ub
            if lb is None or ub is None:
                raise GDP_Error("Variables that appear in disjuncts must be "
                                "bounded in order to use the chull "
                                "transformation! Missing bound for %s."
                                % (var.name))

            disaggregatedVar = Var(within=Reals,
                                   bounds=(min(0, lb), max(0, ub)),
                                   initialize=var.value)
            # naming conflicts are possible here since this is a bunch
            # of variables from different blocks coming together, so we
            # get a unique name
            disaggregatedVarName = unique_component_name(
                relaxationBlock, var.local_name)
            relaxationBlock.add_component(
                disaggregatedVarName, disaggregatedVar)
            chull['disaggregatedVars'][var] = disaggregatedVar
            relaxationBlockInfo['srcVars'][disaggregatedVar] = var

            bigmConstraint = Constraint(transBlock.lbub)
            relaxationBlock.add_component(
                disaggregatedVarName + "_bounds", bigmConstraint)
            if lb:
                bigmConstraint.add(
                    'lb', obj.indicator_var*lb <= disaggregatedVar)
            if ub:
                bigmConstraint.add(
                    'ub', disaggregatedVar <= obj.indicator_var*ub)
            chull['bigmConstraints'][var] = bigmConstraint
            relaxationBlockInfo['boundConstraintToSrcVar'][bigmConstraint] = var

        for var in localVars:
            lb = var.lb
            ub = var.ub
            if lb is None or ub is None:
                raise GDP_Error("Variables that appear in disjuncts must be "
                                "bounded in order to use the chull "
                                "transformation! Missing bound for %s."
                                % (var.name))
            if value(lb) > 0:
                var.setlb(0)
            if value(ub) < 0:
                var.setub(0)

            # naming conflicts are possible here since this is a bunch
            # of variables from different blocks coming together, so we
            # get a unique name
            conName = unique_component_name(
                relaxationBlock, var.local_name+"_bounds")
            bigmConstraint = Constraint(transBlock.lbub)
            relaxationBlock.add_component(conName, bigmConstraint)
            bigmConstraint.add('lb', obj.indicator_var*lb <= var)
            bigmConstraint.add('ub', var <= obj.indicator_var*ub)
            chull['bigmConstraints'][var] = bigmConstraint
            relaxationBlockInfo['boundConstraintToSrcVar'][bigmConstraint] = var

        var_substitute_map = dict((id(v), newV) for v, newV in
                                  iteritems(chull['disaggregatedVars']))
        zero_substitute_map = dict((id(v), ZeroConstant) for v, newV in
                                   iteritems(chull['disaggregatedVars']))
        zero_substitute_map.update((id(v), ZeroConstant)
                                   for v in localVars)

        # Transform each component within this disjunct
        self._transform_block_components(obj, obj, infodict, var_substitute_map,
                                         zero_substitute_map)

        # deactivate disjunct so we know we've relaxed it
        obj._deactivate_without_fixing_indicator()
        infodict['relaxed'] = True


    def _transform_block_components(
            self, block, disjunct, infodict,
            var_substitute_map, zero_substitute_map):
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
                        "No chull transformation handler registered "
                        "for modeling components of type %s" % obj.type() )
                continue
            # obj is what we are transforming, we pass disjunct
            # through so that we will have access to the indicator
            # variables down the line.
            handler(obj, disjunct, infodict, var_substitute_map,
                    zero_substitute_map)


    def _warn_for_active_disjunction(
            self, disjunction, disjunct, infodict, var_substitute_map,
            zero_substitute_map):
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
                        "disjunction before the disjunct in the list." \
                        % (problemdisj.name, disjunct.name))


    def _warn_for_active_disjunct(
            self, innerdisjunct, outerdisjunct, infodict, var_substitute_map,
            zero_substitute_map):
        assert innerdisjunct.active
        problemdisj = innerdisjunct
        if innerdisjunct.is_indexed():
            for i in sorted(iterkeys(innerdisjunct)):
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
        raise GDP_Error("Found active disjunct {0} in disjunct {1}! Either {0} "
                        "is not in a disjunction or the disjunction it is in "
                        "has not been transformed. {0} needs to be deactivated "
                        "or its disjunction transformed before {1} can be "
                        "transformed.".format(problemdisj.name,
                                              outerdisjunct.name))


    def _transform_block_on_disjunct(
            self, block, disjunct, infodict, var_substitute_map,
            zero_substitute_map):
        # We look through everything on the component map of the block
        # and transform it just as we would if it was on the disjunct
        # directly.  (We are passing the disjunct through so that when
        # we find constraints, _xform_constraint will have access to
        # the correct indicator variable.
        self._transform_block_components(
            block, disjunct, infodict, var_substitute_map, zero_substitute_map)


    def _xform_constraint(self, obj, disjunct, infodict, var_substitute_map,
                          zero_substitute_map):
        # we will put a new transformed constraint on the relaxation block.
        relaxationBlock = infodict['chull']['relaxationBlock']
        transBlock = relaxationBlock.parent_block()
        varMap = infodict['chull']['disaggregatedVars']

        # Though rare, it is possible to get naming conflicts here
        # since constraints from all blocks are getting moved onto the
        # same block. So we get a unique name
        name = unique_component_name(relaxationBlock, obj.name)

        if obj.is_indexed():
            try:
                newConstraint = Constraint(obj.index_set(), transBlock.lbub)
            except:
                # The original constraint may have been indexed by a
                # non-concrete set (like an Any).  We will give up on
                # strict index verification and just blindly proceed.
                newConstraint = Constraint(Any)
        else:
            newConstraint = Constraint(transBlock.lbub)
        relaxationBlock.add_component(name, newConstraint)
        # add mapping of original constraint to transformed constraint
        # in transformation info dictionary
        infodict['chull']['relaxedConstraints'][obj] = newConstraint
        # add mapping of transformed constraint back to original constraint (we
        # know that the info dict is already created because this only got
        # called if we were transforming a disjunct...)
        relaxationBlock._gdp_transformation_info['srcConstraints'][
            newConstraint] = obj

        for i in sorted(iterkeys(obj)):
            c = obj[i]
            if not c.active:
                continue

            NL = c.body.polynomial_degree() not in (0,1)
            EPS = self._config.EPS
            mode = self._config.perspective_function

            # We need to evaluate the expression at the origin *before*
            # we substitute the expression variables with the
            # disaggregated variables
            if not NL or mode == "FurmanSawayaGrossmann":
                h_0 = clone_without_expression_components(
                    c.body, substitute=zero_substitute_map)

            y = disjunct.indicator_var
            if NL:
                if mode == "LeeGrossmann":
                    sub_expr = clone_without_expression_components(
                        c.body,
                        substitute=dict(
                            (var,  subs/y)
                            for var, subs in iteritems(var_substitute_map) )
                    )
                    expr = sub_expr * y
                elif mode == "GrossmannLee":
                    sub_expr = clone_without_expression_components(
                        c.body,
                        substitute=dict(
                            (var, subs/(y + EPS))
                            for var, subs in iteritems(var_substitute_map) )
                    )
                    expr = (y + EPS) * sub_expr
                elif mode == "FurmanSawayaGrossmann":
                    sub_expr = clone_without_expression_components(
                        c.body,
                        substitute=dict(
                            (var, subs/((1 - EPS)*y + EPS))
                            for var, subs in iteritems(var_substitute_map) )
                    )
                    expr = ((1-EPS)*y + EPS)*sub_expr - EPS*h_0*(1-y)
                else:
                    raise RuntimeError("Unknown NL CHull mode")
            else:
                expr = clone_without_expression_components(
                    c.body, substitute=var_substitute_map)

            if c.equality:
                if NL:
                    newConsExpr = expr == c.lower*y
                else:
                    v = list(EXPR.identify_variables(expr))
                    if len(v) == 1 and not c.lower:
                        # Setting a variable to 0 in a disjunct is
                        # *very* common.  We should recognize that in
                        # that structure, the disaggregated variable
                        # will also be fixed to 0.
                        v[0].fix(0)
                        continue
                    newConsExpr = expr - (1-y)*h_0 == c.lower*y

                if obj.is_indexed():
                    newConstraint.add((i, 'eq'), newConsExpr)
                else:
                    newConstraint.add('eq', newConsExpr)
                continue

            if c.lower is not None:
                # TODO: At the moment there is no reason for this to be in both
                # lower and upper... I think there could be though if I say what
                # the new constraint is going to be or something.
                if __debug__ and logger.isEnabledFor(logging.DEBUG):
                    logger.debug("GDP(cHull): Transforming constraint " +
                                 "'%s'", c.name)
                if NL:
                    newConsExpr = expr >= c.lower*y
                else:
                    newConsExpr = expr - (1-y)*h_0 >= c.lower*y

                if obj.is_indexed():
                    newConstraint.add((i, 'lb'), newConsExpr)
                else:
                    newConstraint.add('lb', newConsExpr)

            if c.upper is not None:
                if __debug__ and logger.isEnabledFor(logging.DEBUG):
                    logger.debug("GDP(cHull): Transforming constraint " +
                                 "'%s'", c.name)
                if NL:
                    newConsExpr = expr <= c.upper*y
                else:
                    newConsExpr = expr - (1-y)*h_0 <= c.upper*y

                if obj.is_indexed():
                    newConstraint.add((i, 'ub'), newConsExpr)
                else:
                    newConstraint.add('ub', newConsExpr)
