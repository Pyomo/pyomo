#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import logging

import pyomo.common.config as cfg
from pyomo.common.modeling import unique_component_name
from pyomo.core.expr.numvalue import ZeroConstant
from pyomo.core.base.component import ActiveComponent, ComponentUID
from pyomo.core.kernel.component_map import ComponentMap
from pyomo.core.kernel.component_set import ComponentSet
import pyomo.core.expr.current as EXPR
from pyomo.core.base import Transformation, TransformationFactory
from pyomo.core import (
    Block, Connector, Constraint, Param, Set, Suffix, Var,
    Expression, SortComponents, TraversalStrategy,
    Any, RangeSet, Reals, value
)
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.disjunct import _DisjunctData, SimpleDisjunct
from pyomo.gdp.util import clone_without_expression_components, target_list, \
    is_child_of
from pyomo.gdp.plugins.gdp_var_mover import HACK_GDP_Disjunct_Reclassifier

from six import iteritems, iterkeys
from weakref import ref as weakref_ref

# TODO: DEBUG
from nose.tools import set_trace

logger = logging.getLogger('pyomo.gdp.chull')

NAME_BUFFER = {}

@TransformationFactory.register('gdp.chull', 
                                doc="Relax disjunctive model by forming "
                                "the convex hull.")

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
    targets : (block, disjunction, or list of those types)
        The targets to transform. This can be a block, disjunction, or a
        list of blocks and Disjunctions [default: the instance]

    The transformation will create a new Block with a unique
    name beginning "_pyomo_gdp_chull_relaxation".  That Block will
    contain an indexed Block named "relaxedDisjuncts", which will hold
    the relaxed disjuncts.  This block is indexed by an integer
    indicating the order in which the disjuncts were relaxed.
    Each block has a dictionary "_constraintMap":
    
        'srcConstraints': ComponentMap(<transformed constraint>:
                                       <src constraint>),
        'transformedConstraints': ComponentMap(<src constraint>:
                                               <transformed constraint>)

    It will have a dictionary "_disaggregatedVarMap:
        'srcVar': ComponentMap(<src var>:<disaggregated var>),
        'disaggregatedVar': ComponentMap(<disaggregated var>:<src var>)

    And, last, it will have a ComponentMap "_bigMConstraintMap":

        <disaggregated var>:<bounds constraint>

    All transformed Disjuncts will have a pointer to the block their transformed
    constraints are on, and all transformed Disjunctions will have a 
    pointer to the corresponding OR or XOR constraint.

    The _pyomo_gdp_chull_relaxation block will have a ComponentMap
    "_disaggregationConstraintMap":
        <src var>:ComponentMap(<srcDisjunction>: <disaggregation constraint>)

    TODO: I wish:
        (<src var>, <srcDisjunction>): <disaggregation constraint>

    """


    CONFIG = cfg.ConfigBlock('gdp.chull')
    CONFIG.declare('targets', cfg.ConfigValue(
        default=None,
        domain=target_list,
        description="target or list of targets that will be relaxed",
        doc="""

        This specifies the target or list of targets to relax as either a
        component or a list of components. If None (default), the entire model
        is transformed. Note that if the transformation is done out of place,
        the list of targets should be attached to the model before it is cloned,
        and the list will specify the targets on the cloned instance."""
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
            Constraint : self._transform_constraint,
            Var :        False,
            Connector :  False,
            Expression : False,
            Param :      False,
            Set :        False,
            RangeSet:    False,
            Suffix :     False,
            Disjunction: self._warn_for_active_disjunction,
            Disjunct:    self._warn_for_active_disjunct,
            Block:       self._transform_block_on_disjunct,
            }

    def _apply_to(self, instance, **kwds):
        assert not NAME_BUFFER
        try:
            self._apply_to_impl(instance, **kwds)
        finally:
            # Clear the global name buffer now that we are done
            NAME_BUFFER.clear()


    def _apply_to_impl(self, instance, **kwds):
        self._config = self.CONFIG(kwds.pop('options', {}))
        self._config.set_value(kwds)

        targets = self._config.targets
        if targets is None:
            targets = ( instance, )
            _HACK_transform_whole_instance = True
        else:
            _HACK_transform_whole_instance = False
        knownBlocks = set()
        for t in targets:
            # [ESJ 10/18/2019] This can go away when we deprecate using CUIDs as
            # targets. The warning is issued in util, but we need to make sure
            # that we do the right thing here.
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
                    self._transform_disjunction(t)
                else:
                    self._transform_disjunctionData(t, t.index())
            elif t.type() in (Block, Disjunct):
                if t.parent_component() is t:
                    self._transform_block(t)
                else:
                    self._transform_blockData(t)
            else:
                raise GDP_Error(
                    "Target %s was not a Block, Disjunct, or Disjunction. "
                    "It was of type %s and can't be transformed"
                    % (t.name, type(t)) )

        # HACK for backwards compatibility with the older GDP transformations
        #
        # Until the writers are updated to find variables on things
        # other than active blocks, we need to reclassify the Disjuncts
        # as Blocks after transformation so that the writer will pick up
        # all the variables that it needs (in this case, indicator_vars).
        if _HACK_transform_whole_instance:
            HACK_GDP_Disjunct_Reclassifier().apply_to(instance)

    def _add_transformation_block(self, instance):
         # make a transformation block on instance where we will store
         # transformed components
        transBlockName = unique_component_name(
            instance,
            '_pyomo_gdp_chull_relaxation')
        transBlock = Block()
        instance.add_component(transBlockName, transBlock)
        transBlock.relaxedDisjuncts = Block(Any)
        transBlock.lbub = Set(initialize = ['lb','ub','eq'])
        # TODO: This is wrong!!
        # We will store all of the disaggregation constraints for any
        # Disjunctions we transform onto this block here. Note that this
        # (correctly) means that we will move them up off of the Disjunct in the
        # case of nested disjunctions
        transBlock.disaggregationConstraints = Constraint(Any)

        # This will map from srcVar to a map of srcDisjunction to the
        # disaggregation constraint corresponding to srcDisjunction
        transBlock._disaggregationConstraintMap = ComponentMap()

        return transBlock

    def _contained_in(self, var, block):
        "Return True if a var is in the subtree rooted at block"
        while var is not None:
            if var.parent_component() is block:
                return True
            var = var.parent_block()
            if var is block:
                return True
        return False

    def _transform_block(self, obj):
        for i in sorted(iterkeys(obj)):
            self._transform_blockData(obj[i])

    def _transform_blockData(self, obj):
        # Transform every (active) disjunction in the block
        for disjunction in obj.component_objects(
                Disjunction,
                active=True,
                sort=SortComponents.deterministic,
                descend_into=(Block,Disjunct),
                descent_order=TraversalStrategy.PostfixDFS):
            self._transform_disjunction(disjunction)

    def _get_xor_constraint(self, disjunction, transBlock):
        # NOTE: This differs from bigm because in chull there is no reason to
        # but the XOR constraint on the parent block of the Disjunction. We
        # don't move anything in the case of nested disjunctions, so to avoid
        # name conflicts, we put everything together on the transformation
        # block.

        # We never do this for just a DisjunctionData because we need
        # to know about the index set of its parent component. So if
        # we called this on a DisjunctionData, we did something wrong.
        assert isinstance(disjunction, Disjunction)

        # check if the constraint already exists
        if not disjunction._algebraic_constraint is None:
            return disjunction._algebraic_constraint()

        # add the XOR (or OR) constraints to parent block (with
        # unique name) It's indexed if this is an
        # IndexedDisjunction, not otherwise
        orC = Constraint(disjunction.index_set()) if \
              disjunction.is_indexed() else Constraint()
        transBlock.add_component( 
            unique_component_name(transBlock, disjunction.name + '_xor'),
            orC)
        disjunction._algebraic_constraint = weakref_ref(orC)

        return orC

    def _transform_disjunction(self, obj):
        # put the transformation block on the parent block of the Disjunction
        transBlock = self._add_transformation_block(obj.parent_block())
        # and create the xor constraint
        xorConstraint = self._get_xor_constraint(obj, transBlock)

        # create the disjunction constraint and disaggregation
        # constraints and then relax each of the disjunctionDatas
        for i in sorted(iterkeys(obj)):
            self._transform_disjunctionData(obj[i], i, transBlock)

        # deactivate so we know we relaxed
        obj.deactivate()

    def _transform_disjunctionData(self, obj, index, transBlock=None):
        # TODO: This should've been a bug, I think?? Make sure it's tested...
        if not obj.active:
            return
        # Convex hull doesn't work if this is an or constraint. So if
        # xor is false, give up
        if not obj.xor:
            raise GDP_Error("Cannot do convex hull transformation for "
                            "disjunction %s with OR constraint. Must be an XOR!"
                            % obj.name)
        
        if transBlock is None:
            transBlock = self._add_transformation_block(obj.parent_block())

        parent_component = obj.parent_component()
        
        orConstraint = self._get_xor_constraint(parent_component, transBlock)
        disaggregationConstraint = transBlock.disaggregationConstraints
        disaggregationConstraintMap = transBlock._disaggregationConstraintMap

        # We first go through and collect all the variables that we
        # are going to disaggregate.
        varOrder_set = ComponentSet()
        varOrder = []
        varsByDisjunct = ComponentMap()
        for disjunct in obj.disjuncts:
            disjunctVars = varsByDisjunct[disjunct] = ComponentSet()
            for cons in disjunct.component_data_objects(
                    Constraint,
                    active = True,
                    sort=SortComponents.deterministic,
                    descend_into=Block):
                # [ESJ 12/10/2019] I don't think I agree with this... Fixing is
                # not a promise for the future. And especially since this is
                # undocumented, we are asking for trouble with silent failures
                # later...
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
                    if not var in varOrder_set:
                        varOrder.append(var)
                        varOrder_set.add(var)

        # We will only disaggregate variables that
        #  1) appear in multiple disjuncts, or
        #  2) are not contained in this disjunct, or
        # [ESJ 10/18/2019]: I think this is going to happen implicitly 
        # because we will just move them out of the danger zone:
        #  3) are not themselves disaggregated variables
        varSet = []
        localVars = ComponentMap((d,[]) for d in obj.disjuncts)
        for var in varOrder:
            disjuncts = [d for d in varsByDisjunct if var in varsByDisjunct[d]]
            if len(disjuncts) > 1:
                varSet.append(var)
            elif self._contained_in(var, disjuncts[0]):
                localVars[disjuncts[0]].append(var)
            # [ESJ 10/18/2019] TODO: This is strange though because it means we
            # shouldn't have a bug with double-disaggregating right now... And I
            # thought we did. But this is also not my code.
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
                if disjunct._transformation_block is None:
                    if not disjunct.indicator_var.is_fixed() \
                            or value(disjunct.indicator_var) != 0:
                        raise RuntimeError(
                            "GDP chull: disjunct was not relaxed, but "
                            "does not appear to be correctly deactivated.")
                    continue

                disaggregatedVar = disjunct._transformation_block().\
                                   _disaggregatedVarMap['disaggregatedVar'][var]
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
            # and update the map so that we can find this later. We index by
            # variable and the particular disjunction because there is a
            # different one for each disjunction
            if not disaggregationConstraintMap.get(var) is None:
                disaggregationConstraintMap[var][obj] = disaggregationConstraint[
                    consIdx]
            else:
                thismap = disaggregationConstraintMap[var] = ComponentMap()
                thismap[obj] = disaggregationConstraint[consIdx]
            # I wish:
            # disaggregationConstraintMap[
            #     (var, obj)] = disaggregationConstraint[consIdx]

    def _transform_disjunct(self, obj, transBlock, varSet, localVars):
        # deactivated should only come from the user
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

        # create a relaxation block for this disjunct
        relaxedDisjuncts = transBlock.relaxedDisjuncts
        relaxationBlock = relaxedDisjuncts[len(relaxedDisjuncts)]

        # add the map that will link back and forth between transformed
        # constraints and their originals.
        relaxationBlock._constraintMap = {
            'srcConstraints': ComponentMap(),
            'transformedConstraints': ComponentMap()
        }
        # Map between disaggregated variables for this disjunct and their
        # originals
        relaxationBlock._disaggregatedVarMap = {
            'srcVar': ComponentMap(),
            'disaggregatedVar': ComponentMap(),
        }
        # Map between disaggregated variables and their lb*indicator <= var <=
        # ub*indicator constraints
        relaxationBlock._bigMConstraintMap = ComponentMap()

        # add mappings to source disjunct (so we'll know we've relaxed)
        obj._transformation_block = weakref_ref(relaxationBlock)
        relaxationBlock._srcDisjunct = weakref_ref(obj)

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
                relaxationBlock, 
                var.getname(fully_qualified=False, name_buffer=NAME_BUFFER),
            )
            relaxationBlock.add_component( disaggregatedVarName,
                                           disaggregatedVar)
            # store the mappings from variables to their disaggregated selves on
            # the transformation block.
            relaxationBlock._disaggregatedVarMap['disaggregatedVar'][
                var] = disaggregatedVar
            relaxationBlock._disaggregatedVarMap['srcVar'][
                disaggregatedVar] = var

            bigmConstraint = Constraint(transBlock.lbub)
            relaxationBlock.add_component(
                disaggregatedVarName + "_bounds", bigmConstraint)
            if lb:
                bigmConstraint.add(
                    'lb', obj.indicator_var*lb <= disaggregatedVar)
            if ub:
                bigmConstraint.add(
                    'ub', disaggregatedVar <= obj.indicator_var*ub)

            relaxationBlock._bigMConstraintMap[disaggregatedVar] = bigmConstraint
            
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
                relaxationBlock,
                var.getname(fully_qualified=False, name_buffer=NAME_BUFFER
                        ) + "_bounds"
            )
            bigmConstraint = Constraint(transBlock.lbub)
            relaxationBlock.add_component(conName, bigmConstraint)
            
            # These are the constraints for the variables we didn't disaggregate
            # since they are local to a single disjunct
            bigmConstraint.add('lb', obj.indicator_var*lb <= var)
            bigmConstraint.add('ub', var <= obj.indicator_var*ub)

            relaxationBlock._bigMConstraintMap[var] = bigmConstraint
            
        var_substitute_map = dict((id(v), newV) for v, newV in iteritems(
            relaxationBlock._disaggregatedVarMap['disaggregatedVar']))
        zero_substitute_map = dict((id(v), ZeroConstant) for v, newV in \
                                   iteritems(
                                       relaxationBlock._disaggregatedVarMap[
                                           'disaggregatedVar']))
        zero_substitute_map.update((id(v), ZeroConstant)
                                   for v in localVars)

        # Transform each component within this disjunct
        self._transform_block_components(obj, obj, var_substitute_map,
                                         zero_substitute_map)

        # deactivate disjunct so writers can be happy
        obj._deactivate_without_fixing_indicator()

    def _transform_block_components( self, block, disjunct, var_substitute_map,
                                     zero_substitute_map):
        # As opposed to bigm, in chull we do not need to do anything special for
        # nested disjunctions. The indicator variables and disaggregated
        # variables of the inner disjunction will need to be disaggregated again
        # anyway, and nothing will get double-bigm-ed. (If an untransformed
        # disjunction is lurking here, we will catch it below).
        
        # Look through the component map of block and transform
        # everything we have a handler for. Yell if we don't know how
        # to handle it.
        for name, obj in list(iteritems(block.component_map())):
            # Note: This means non-ActiveComponent types cannot have handlers
            if not hasattr(obj, 'active') or not obj.active:
                continue
            handler = self.handlers.get(obj.type(), None)
            if not handler:
                if handler is None:
                    raise GDP_Error(
                        "No chull transformation handler registered "
                        "for modeling components of type %s. If your " 
                        "disjuncts contain non-GDP Pyomo components that "
                        "require transformation, please transform them first."
                        % obj.type() )
                continue
            # obj is what we are transforming, we pass disjunct
            # through so that we will have access to the indicator
            # variables down the line.
            handler(obj, disjunct, var_substitute_map, zero_substitute_map)

    def _warn_for_active_disjunction( self, disjunction, disjunct,
                                      var_substitute_map, zero_substitute_map):
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
        _probDisjName = problemdisj.getname(
            fully_qualified=True, name_buffer=NAME_BUFFER)
        
        assert problemdisj.algebraic_constraint is None
        raise GDP_Error("Found untransformed disjunction %s in disjunct %s! "
                        "The disjunction must be transformed before the "
                        "disjunct. If you are using targets, put the "
                        "disjunction before the disjunct in the list." \
                        % (_probDisjName, disjunct.name))

    def _warn_for_active_disjunct( self, innerdisjunct, outerdisjunct,
                                   var_substitute_map, zero_substitute_map):
        assert innerdisjunct.active
        problemdisj = innerdisjunct
        if innerdisjunct.is_indexed():
            # ESJ: This is different from bigm... Probably this one is right...
            for i in sorted(iterkeys(innerdisjunct)):
                if innerdisjunct[i].active:
                    # This is shouldn't be true, we will complain about it.
                    problemdisj = innerdisjunct[i]
                    break
            
        raise GDP_Error("Found active disjunct {0} in disjunct {1}! Either {0} "
                        "is not in a disjunction or the disjunction it is in "
                        "has not been transformed. {0} needs to be deactivated "
                        "or its disjunction transformed before {1} can be "
                        "transformed.".format(problemdisj.name,
                                              outerdisjunct.name))

    def _transform_block_on_disjunct( self, block, disjunct, var_substitute_map,
                                      zero_substitute_map):
        # We look through everything on the component map of the block
        # and transform it just as we would if it was on the disjunct
        # directly.  (We are passing the disjunct through so that when
        # we find constraints, _transform_constraint will have access to
        # the correct indicator variable.
        for i in sorted(iterkeys(block)):
            self._transform_block_components( block[i], disjunct,
                                              var_substitute_map,
                                              zero_substitute_map)


    def _transform_constraint(self, obj, disjunct, var_substitute_map,
                          zero_substitute_map):
        # we will put a new transformed constraint on the relaxation block.
        relaxationBlock = disjunct._transformation_block()
        transBlock = relaxationBlock.parent_block()
        varMap = relaxationBlock._disaggregatedVarMap['disaggregatedVar']
        constraintMap = relaxationBlock._constraintMap

        # Though rare, it is possible to get naming conflicts here
        # since constraints from all blocks are getting moved onto the
        # same block. So we get a unique name
        name = unique_component_name(relaxationBlock, obj.getname(
            fully_qualified=True, name_buffer=NAME_BUFFER))

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
        constraintMap['transformedConstraints'][obj] = newConstraint
        # add mapping of transformed constraint back to original constraint
        constraintMap['srcConstraints'][newConstraint] = obj

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
                    _name = c.getname(
                        fully_qualified=True, name_buffer=NAME_BUFFER)
                    logger.debug("GDP(cHull): Transforming constraint " +
                                 "'%s'", _name)
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
                    _name = c.getname(
                        fully_qualified=True, name_buffer=NAME_BUFFER)
                    logger.debug("GDP(cHull): Transforming constraint " +
                                 "'%s'", _name)
                if NL:
                    newConsExpr = expr <= c.upper*y
                else:
                    newConsExpr = expr - (1-y)*h_0 <= c.upper*y

                if obj.is_indexed():
                    newConstraint.add((i, 'ub'), newConsExpr)
                else:
                    newConstraint.add('ub', newConsExpr)

    # TODO: Oh... I think these methods should be in util, some of them. They
    # are the same as bigm
    def get_src_disjunct(self, transBlock):
        if not hasattr(transBlock, '_srcDisjunct') or \
           not type(transBlock._srcDisjunct) is weakref_ref:
            raise GDP_Error("Block %s doesn't appear to be a transformation "
                            "block for a disjunct. No source disjunct found." 
                            % transBlock.name)
        return transBlock._srcDisjunct()

    def get_src_disjunction(self, xor_constraint):
        m = xor_constraint.model()
        for disjunction in m.component_data_objects(Disjunction):
            if disjunction._algebraic_constraint:
                if disjunction._algebraic_constraint() is xor_constraint:
                    return disjunction
        raise GDP_Error("It appears that %s is not an XOR or OR constraint "
                        "resulting from transforming a Disjunction."
                        % xor_constraint.name)

    def get_src_constraint(self, transformedConstraint):
        transBlock = transformedConstraint.parent_block()
        # This should be our block, so if it's not, the user messed up and gave
        # us the wrong thing. If they happen to also have a _constraintMap then
        # the world is really against us.
        if not hasattr(transBlock, "_constraintMap"):
            raise GDP_Error("Constraint %s is not a transformed constraint" 
                            % transformedConstraint.name)
        return transBlock._constraintMap['srcConstraints'][transformedConstraint]

    # TODO: This needs to go to util because I think it gets used in bigm too
    def _get_parent_disjunct(self, obj, err_message):
        # We are going to have to traverse up until we find the disjunct that
        # obj lives on
        parent = obj.parent_block()
        while not type(parent) in (_DisjunctData, SimpleDisjunct):
            parent = parent.parent_block()
            if parent is None:
                raise GDP_Error(err_message)
        return parent

    def get_transformed_constraint(self, srcConstraint):
        disjunct = self._get_parent_disjunct(
            srcConstraint,                     
            "Constraint %s is not on a disjunct and so was not "
            "transformed" % srcConstraint.name)
        transBlock = disjunct._transformation_block
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

    ## Beginning here, these are unique to chull

    def get_disaggregated_var(self, v, disjunct):
        # Retrieve the disaggregated var corresponding to the specified disjunct
        if disjunct._transformation_block is None:
            raise GDP_Error("Disjunct %s has not been transformed" 
                            % disjunct.name)
        transBlock = disjunct._transformation_block()
        return transBlock._disaggregatedVarMap['disaggregatedVar'][v]
        
    def get_src_var(self, disaggregated_var):
        transBlock = disaggregated_var.parent_block()
        try:
            src_disjunct = transBlock._srcDisjunct()
        except:
            raise GDP_Error("%s does not appear to be a disaggregated variable"
                            % disaggregated_var.name)
        return transBlock._disaggregatedVarMap['srcVar'][disaggregated_var]

    # retrieves the disaggregation constraint for original_var resulting from
    # transforming disjunction
    def get_disaggregation_constraint(self, original_var, disjunction):
        for disjunct in disjunction.disjuncts:
            transBlock = disjunct._transformation_block
            if not transBlock is None:
                break
        if transBlock is None:
            raise GDP_Error("Disjunction %s has not been properly transformed: "
                            "None of its disjuncts are transformed." 
                            % disjunction.name)
        try:
            return transBlock().parent_block()._disaggregationConstraintMap[
                original_var][disjunction]
        except:
            raise GDP_Error("It doesn't appear that %s is a variable that was "
                            "disaggregated by Disjunction %s" % 
                            (original_var.name, disjunction.name))

    def get_var_bounds_constraint(self, v):
        # v is a disaggregated variable: get the indicator*lb <= it <=
        # indicator*ub constraint for it
        transBlock = v.parent_block()
        try:
            return transBlock._bigMConstraintMap[v]
        except:
            raise GDP_Error("Either %s is not a disaggregated variable, or "
                            "the disjunction that disaggregates it has not "
                            "been properly transformed." % v.name)

    # I don't think we need this. look for the only variable not named
    # 'indicator_var'! If we ever need to be efficient, we can do the reverse
    # map.
    # def get_var_from_bounds_constraint(self, cons):
    #     transBlock = cons.parent_block()
    #     return transBlock._bigMConstraintMap['srcVar'][cons]

    # TODO: These maps actually get used in cuttingplanes. It will be worth
    # making sure that the ones that are called there are on the more efficient
    # side...

    # TODO: This is not a relaxation, I would love to not be using that word in
    # the code... And I need a convention for distinguishing between the
    # disjunct transBlocks and the parent blocks of those.
