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
from pyomo.common import deprecated
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import unique_component_name
from pyomo.core.expr.numvalue import ZeroConstant
from pyomo.core.base.component import ActiveComponent
import pyomo.core.expr.current as EXPR
from pyomo.core.base import Transformation, TransformationFactory, Reference
from pyomo.core import (
    Block, BooleanVar, Connector, Constraint, Param, Set, SetOf, Suffix, Var,
    Expression, SortComponents, TraversalStrategy,
    Any, RangeSet, Reals, value, NonNegativeIntegers, LogicalConstraint,
)
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.util import ( _warn_for_active_logical_constraint,
                             clone_without_expression_components, target_list,
                             is_child_of, get_src_disjunction,
                             get_src_constraint, get_transformed_constraints,
                             get_src_disjunct, _warn_for_active_disjunction,
                             _warn_for_active_disjunct, )
from functools import wraps
from six import iteritems, iterkeys
from weakref import ref as weakref_ref

logger = logging.getLogger('pyomo.gdp.hull')

NAME_BUFFER = {}

@TransformationFactory.register(
    'gdp.hull',
    doc="Relax disjunctive model by forming the hull reformulation.")
class Hull_Reformulation(Transformation):
    """Relax disjunctive model by forming the hull reformulation.

    Relaxes a disjunctive model into an algebraic model by forming the
    hull reformulation of each disjunction.

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
    name beginning "_pyomo_gdp_hull_reformulation".  That Block will
    contain an indexed Block named "relaxedDisjuncts", which will hold
    the relaxed disjuncts.  This block is indexed by an integer
    indicating the order in which the disjuncts were relaxed.
    Each block has a dictionary "_constraintMap":

        'srcConstraints': ComponentMap(<transformed constraint>:
                                       <src constraint>),
        'transformedConstraints':ComponentMap(<src constraint container>:
                                              <transformed constraint container>,
                                              <src constraintData>:
                                              [<transformed constraintDatas>]
                                             )

    It will have a dictionary "_disaggregatedVarMap:
        'srcVar': ComponentMap(<src var>:<disaggregated var>),
        'disaggregatedVar': ComponentMap(<disaggregated var>:<src var>)

    And, last, it will have a ComponentMap "_bigMConstraintMap":

        <disaggregated var>:<bounds constraint>

    All transformed Disjuncts will have a pointer to the block their transformed
    constraints are on, and all transformed Disjunctions will have a
    pointer to the corresponding OR or XOR constraint.

    The _pyomo_gdp_hull_reformulation block will have a ComponentMap
    "_disaggregationConstraintMap":
        <src var>:ComponentMap(<srcDisjunction>: <disaggregation constraint>)

    """


    CONFIG = cfg.ConfigBlock('gdp.hull')
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
    CONFIG.declare('assume_fixed_vars_permanent', cfg.ConfigValue(
        default=False,
        domain=bool,
        description="Boolean indicating whether or not to transform so that the "
        "the transformed model will still be valid when fixed Vars are unfixed.",
        doc="""
        If True, the transformation will not disaggregate fixed variables.
        This means that if a fixed variable is unfixed after transformation,
        the transformed model is no longer valid. By default, the transformation
        will disagregate fixed variables so that any later fixing and unfixing
        will be valid in the transformed model.
        """
    ))

    def __init__(self):
        super(Hull_Reformulation, self).__init__()
        self.handlers = {
            Constraint : self._transform_constraint,
            Var :        False,
            BooleanVar:  False,
            Connector :  False,
            Expression : False,
            Param :      False,
            Set :        False,
            SetOf :      False,
            RangeSet:    False,
            Suffix :     False,
            Disjunction: self._warn_for_active_disjunction,
            Disjunct:    self._warn_for_active_disjunct,
            Block:       self._transform_block_on_disjunct,
            LogicalConstraint: self._warn_for_active_logical_statement,
            }
        self._generate_debug_messages = False

    def _add_local_vars(self, block, local_var_dict):
        localVars = block.component('LocalVars')
        if type(localVars) is Suffix:
            for disj, var_list in iteritems(localVars):
                if local_var_dict.get(disj) is None:
                    local_var_dict[disj] = ComponentSet(var_list)
                else:
                    local_var_dict[disj].update(var_list)

    def _get_local_var_suffixes(self, block, local_var_dict):
        # You can specify suffixes on any block (disjuncts included). This method
        # starts from a Disjunct (presumably) and checks for a LocalVar suffixes
        # going both up and down the tree, adding them into the dictionary that
        # is the second argument.

        # first look beneath where we are (there could be Blocks on this
        # disjunct)
        for b in block.component_data_objects(Block, descend_into=(Block),
                                              active=True,
                                              sort=SortComponents.deterministic):
            self._add_local_vars(b, local_var_dict)
        # now traverse upwards and get what's above
        while block is not None:
            self._add_local_vars(block, local_var_dict)
            block = block.parent_block()

        return local_var_dict

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
        self._generate_debug_messages = is_debug_set(logger)

        targets = self._config.targets
        if targets is None:
            targets = ( instance, )
        knownBlocks = {}
        for t in targets:
            # check that t is in fact a child of instance
            if not is_child_of(parent=instance, child=t,
                               knownBlocks=knownBlocks):
                raise GDP_Error(
                    "Target '%s' is not a component on instance '%s'!"
                    % (t.name, instance.name))
            elif t.ctype is Disjunction:
                if t.is_indexed():
                    self._transform_disjunction(t)
                else:
                    self._transform_disjunctionData(t, t.index())
            elif t.ctype in (Block, Disjunct):
                if t.is_indexed():
                    self._transform_block(t)
                else:
                    self._transform_blockData(t)
            else:
                raise GDP_Error(
                    "Target '%s' was not a Block, Disjunct, or Disjunction. "
                    "It was of type %s and can't be transformed."
                    % (t.name, type(t)) )

    def _add_transformation_block(self, instance):
        # make a transformation block on instance where we will store
        # transformed components
        transBlockName = unique_component_name(
            instance,
            '_pyomo_gdp_hull_reformulation')
        transBlock = Block()
        instance.add_component(transBlockName, transBlock)
        transBlock.relaxedDisjuncts = Block(NonNegativeIntegers)
        transBlock.lbub = Set(initialize = ['lb','ub','eq'])
        # We will store all of the disaggregation constraints for any
        # Disjunctions we transform onto this block here.
        transBlock.disaggregationConstraints = Constraint(NonNegativeIntegers,
                                                          Any)

        # This will map from srcVar to a map of srcDisjunction to the
        # disaggregation constraint corresponding to srcDisjunction
        transBlock._disaggregationConstraintMap = ComponentMap()

        return transBlock

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

    def _add_xor_constraint(self, disjunction, transBlock):
        # Put XOR constraint on the transformation block

        # We never do this for just a DisjunctionData because we need
        # to know about the index set of its parent component. So if
        # we called this on a DisjunctionData, we did something wrong.
        assert isinstance(disjunction, Disjunction)

        # check if the constraint already exists
        if disjunction._algebraic_constraint is not None:
            return disjunction._algebraic_constraint()

        # add the XOR (or OR) constraints to parent block (with
        # unique name) It's indexed if this is an
        # IndexedDisjunction, not otherwise
        orC = Constraint(disjunction.index_set())
        transBlock.add_component(
            unique_component_name(transBlock,
                                  disjunction.getname(fully_qualified=True,
                                                      name_buffer=NAME_BUFFER) +\
                                  '_xor'), orC)
        disjunction._algebraic_constraint = weakref_ref(orC)

        return orC

    def _transform_disjunction(self, obj):
        # NOTE: this check is actually necessary because it's possible we go
        # straight to this function when we use targets.
        if not obj.active:
            return

        # put the transformation block on the parent block of the Disjunction,
        # unless this is a disjunction we have seen in a prior call to hull, in
        # which case we will use the same transformation block we created
        # before.
        if obj._algebraic_constraint is not None:
            transBlock = obj._algebraic_constraint().parent_block()
        else:
            transBlock = self._add_transformation_block(obj.parent_block())
        # and create the xor constraint
        xorConstraint = self._add_xor_constraint(obj, transBlock)

        # create the disjunction constraint and disaggregation
        # constraints and then relax each of the disjunctionDatas
        for i in sorted(iterkeys(obj)):
            self._transform_disjunctionData(obj[i], i, transBlock)

        # deactivate so the writers will be happy
        obj.deactivate()

    def _transform_disjunctionData(self, obj, index, transBlock=None):
        if not obj.active:
            return
        # Hull reformulation doesn't work if this is an OR constraint. So if
        # xor is false, give up
        if not obj.xor:
            raise GDP_Error("Cannot do hull reformulation for "
                            "Disjunction '%s' with OR constraint.  "
                            "Must be an XOR!" % obj.name)

        if transBlock is None:
            # It's possible that we have already created a transformation block
            # for another disjunctionData from this same container. If that's
            # the case, let's use the same transformation block. (Else it will
            # be really confusing that the XOR constraint goes to that old block
            # but we create a new one here.)
            if obj.parent_component()._algebraic_constraint is not None:
                transBlock = obj.parent_component()._algebraic_constraint().\
                             parent_block()
            else:
                transBlock = self._add_transformation_block(obj.parent_block())

        parent_component = obj.parent_component()

        orConstraint = self._add_xor_constraint(parent_component, transBlock)
        disaggregationConstraint = transBlock.disaggregationConstraints
        disaggregationConstraintMap = transBlock._disaggregationConstraintMap

        # Just because it's unlikely this is what someone meant to do...
        if len(obj.disjuncts) == 0:
            raise GDP_Error("Disjunction '%s' is empty. This is "
                            "likely indicative of a modeling error."  %
                            obj.getname(fully_qualified=True,
                                        name_buffer=NAME_BUFFER))

        # We first go through and collect all the variables that we
        # are going to disaggregate.
        varOrder_set = ComponentSet()
        varOrder = []
        varsByDisjunct = ComponentMap()
        localVarsByDisjunct = ComponentMap()
        include_fixed_vars = not self._config.assume_fixed_vars_permanent
        for disjunct in obj.disjuncts:
            disjunctVars = varsByDisjunct[disjunct] = ComponentSet()
            for cons in disjunct.component_data_objects(
                    Constraint,
                    active = True,
                    sort=SortComponents.deterministic,
                    descend_into=Block):
                # [ESJ 02/14/2020] By default, we disaggregate fixed variables
                # on the philosophy that fixing is not a promise for the future
                # and we are mathematically wrong if we don't transform these
                # correctly and someone later unfixes them and keeps playing
                # with their transformed model. However, the user may have set
                # assume_fixed_vars_permanent to True in which case we will skip
                # them
                for var in EXPR.identify_variables(
                        cons.body, include_fixed=include_fixed_vars):
                    # Note the use of a list so that we will
                    # eventually disaggregate the vars in a
                    # deterministic order (the order that we found
                    # them)
                    disjunctVars.add(var)
                    if not var in varOrder_set:
                        varOrder.append(var)
                        varOrder_set.add(var)

            # check for LocalVars Suffix
            localVarsByDisjunct = self._get_local_var_suffixes(
                disjunct, localVarsByDisjunct)

        # We will disaggregate all variables which are not explicitly declared
        # as being local. Note however, that we do declare our own disaggregated
        # variables as local, so they will not be re-disaggregated.
        varSet = []
        # Note that variables are local with respect to a Disjunct. We deal with
        # them here to do some error checking (if something is obviously not
        # local since it is used in multiple Disjuncts in this Disjunction) and
        # also to get a deterministic order in which to process them when we
        # transform the Disjuncts: Values of localVarsByDisjunct are
        # ComponentSets, so we need this for determinism (we iterate through the
        # localVars of a Disjunct later)
        localVars = ComponentMap()
        for var in varOrder:
            disjuncts = [d for d in varsByDisjunct if var in varsByDisjunct[d]]
            # clearly not local if used in more than one disjunct
            if len(disjuncts) > 1:
                if self._generate_debug_messages:
                    logger.debug("Assuming '%s' is not a local var since it is"
                                 "used in multiple disjuncts." %
                                 var.getname(fully_qualified=True,
                                             name_buffer=NAME_BUFFER))
                varSet.append(var)
            # disjuncts is a list of length 1
            elif localVarsByDisjunct.get(disjuncts[0]) is not None:
                if var in localVarsByDisjunct[disjuncts[0]]:
                    localVars_thisDisjunct = localVars.get(disjuncts[0])
                    if localVars_thisDisjunct is not None:
                        localVars[disjuncts[0]].append(var)
                    else:
                        localVars[disjuncts[0]] = [var]
                else:
                    # It's not local to this Disjunct
                    varSet.append(var)
            else:
                # We don't even have have any local vars for this Disjunct.
                varSet.append(var)

        # Now that we know who we need to disaggregate, we will do it
        # while we also transform the disjuncts.
        or_expr = 0
        for disjunct in obj.disjuncts:
            or_expr += disjunct.indicator_var
            self._transform_disjunct(disjunct, transBlock, varSet,
                                     localVars.get(disjunct, []))
        orConstraint.add(index, (or_expr, 1))
        # map the DisjunctionData to its XOR constraint to mark it as
        # transformed
        obj._algebraic_constraint = weakref_ref(orConstraint[index])

        # add the reaggregation constraints
        for i, var in enumerate(varSet):
            disaggregatedExpr = 0
            for disjunct in obj.disjuncts:
                if disjunct._transformation_block is None:
                    # Because we called _transform_disjunct in the loop above,
                    # we know that if this isn't transformed it is because it
                    # was cleanly deactivated, and we can just skip it.
                    continue

                disaggregatedVar = disjunct._transformation_block().\
                                   _disaggregatedVarMap['disaggregatedVar'][var]
                disaggregatedExpr += disaggregatedVar

            disaggregationConstraint.add((i, index), var == disaggregatedExpr)
            # and update the map so that we can find this later. We index by
            # variable and the particular disjunction because there is a
            # different one for each disjunction
            if disaggregationConstraintMap.get(var) is not None:
                disaggregationConstraintMap[var][obj] = disaggregationConstraint[
                    (i, index)]
            else:
                thismap = disaggregationConstraintMap[var] = ComponentMap()
                thismap[obj] = disaggregationConstraint[(i, index)]

        # deactivate for the writers
        obj.deactivate()

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
                        "The disjunct '%s' is deactivated, but the "
                        "indicator_var is fixed to %s. This makes no sense."
                        % ( obj.name, value(obj.indicator_var) ))
            if obj._transformation_block is None:
                raise GDP_Error(
                    "The disjunct '%s' is deactivated, but the "
                    "indicator_var is not fixed and the disjunct does not "
                    "appear to have been relaxed. This makes no sense. "
                    "(If the intent is to deactivate the disjunct, fix its "
                    "indicator_var to 0.)"
                    % ( obj.name, ))

        if obj._transformation_block is not None:
            # we've transformed it, which means this is the second time it's
            # appearing in a Disjunction
            raise GDP_Error(
                    "The disjunct '%s' has been transformed, but a disjunction "
                    "it appears in has not. Putting the same disjunct in "
                    "multiple disjunctions is not supported." % obj.name)

        # create a relaxation block for this disjunct
        relaxedDisjuncts = transBlock.relaxedDisjuncts
        relaxationBlock = relaxedDisjuncts[len(relaxedDisjuncts)]

        relaxationBlock.localVarReferences = Block()

        # Put the disaggregated variables all on their own block so that we can
        # isolate the name collisions and still have complete control over the
        # names on this block. (This is for peace of mind now, but will matter
        # in the future for adding the binaries corresponding to Boolean
        # indicator vars.)
        relaxationBlock.disaggregatedVars = Block()

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

        # add Suffix to the relaxation block that disaggregated variables are
        # local (in case this is nested in another Disjunct)
        local_var_set = None
        parent_disjunct = obj.parent_block()
        while parent_disjunct is not None:
            if parent_disjunct.ctype is Disjunct:
                break
            parent_disjunct = parent_disjunct.parent_block()
        if parent_disjunct is not None:
            # This limits the cases that a user is allowed to name something
            # (other than a Suffix) 'LocalVars' on a Disjunct. But I am assuming
            # that the Suffix has to be somewhere above the disjunct in the
            # tree, so I can't put it on a Block that I own. And if I'm coopting
            # something of theirs, it may as well be here.
            self._add_local_var_suffix(parent_disjunct)
            if parent_disjunct.LocalVars.get(parent_disjunct) is None:
                parent_disjunct.LocalVars[parent_disjunct] = []
            local_var_set = parent_disjunct.LocalVars[parent_disjunct]

        # add the disaggregated variables and their bigm constraints
        # to the relaxationBlock
        for var in varSet:
            lb = var.lb
            ub = var.ub
            if lb is None or ub is None:
                raise GDP_Error("Variables that appear in disjuncts must be "
                                "bounded in order to use the hull "
                                "transformation! Missing bound for %s."
                                % (var.name))

            disaggregatedVar = Var(within=Reals,
                                   bounds=(min(0, lb), max(0, ub)),
                                   initialize=var.value)
            # naming conflicts are possible here since this is a bunch
            # of variables from different blocks coming together, so we
            # get a unique name
            disaggregatedVarName = unique_component_name(
                relaxationBlock.disaggregatedVars,
                var.getname(fully_qualified=False, name_buffer=NAME_BUFFER),
            )
            relaxationBlock.disaggregatedVars.add_component(
                disaggregatedVarName, disaggregatedVar)
            # mark this as local because we won't re-disaggregate if this is a
            # nested disjunction
            if local_var_set is not None:
                local_var_set.append(disaggregatedVar)
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
                                "bounded in order to use the hull "
                                "transformation! Missing bound for %s."
                                % (var.name))
            if value(lb) > 0:
                var.setlb(0)
            if value(ub) < 0:
                var.setub(0)

            # map it to itself
            relaxationBlock._disaggregatedVarMap['disaggregatedVar'][var] = var
            relaxationBlock._disaggregatedVarMap['srcVar'][var] = var

            # naming conflicts are possible here since this is a bunch
            # of variables from different blocks coming together, so we
            # get a unique name
            conName = unique_component_name(
                relaxationBlock,
                var.getname(fully_qualified=False, name_buffer=NAME_BUFFER) + \
                "_bounds")
            bigmConstraint = Constraint(transBlock.lbub)
            relaxationBlock.add_component(conName, bigmConstraint)
            if lb:
                bigmConstraint.add('lb', obj.indicator_var*lb <= var)
            if ub:
                bigmConstraint.add('ub', var <= obj.indicator_var*ub)
            relaxationBlock._bigMConstraintMap[var] = bigmConstraint

        var_substitute_map = dict((id(v), newV) for v, newV in iteritems(
            relaxationBlock._disaggregatedVarMap['disaggregatedVar']))
        zero_substitute_map = dict((id(v), ZeroConstant) for v, newV in \
                                   iteritems(
                                       relaxationBlock._disaggregatedVarMap[
                                           'disaggregatedVar']))
        zero_substitute_map.update((id(v), ZeroConstant) for v in localVars)

        # Transform each component within this disjunct
        self._transform_block_components(obj, obj, var_substitute_map,
                                         zero_substitute_map)

        # deactivate disjunct so writers can be happy
        obj._deactivate_without_fixing_indicator()

    def _transform_block_components( self, block, disjunct, var_substitute_map,
                                     zero_substitute_map):
        # As opposed to bigm, in hull the only special thing we need to do for
        # nested Disjunctions is to make sure that we move up local var
        # references and also references to the disaggregated variables so that
        # all will be accessible after we transform this Disjunct.The indicator
        # variables and disaggregated variables of the inner disjunction will
        # need to be disaggregated again, but the transformed constraints will
        # not be. But this way nothing will get double-bigm-ed. (If an
        # untransformed disjunction is lurking here, we will catch it below).

        # add references to all local variables on block (including the
        # indicator_var)
        disjunctBlock = disjunct._transformation_block()
        varRefBlock = disjunctBlock.localVarReferences
        for v in block.component_objects(Var, descend_into=Block, active=None):
            varRefBlock.add_component(unique_component_name(
                varRefBlock, v.getname(fully_qualified=True,
                                       name_buffer=NAME_BUFFER)), Reference(v))

        destinationBlock = disjunctBlock.parent_block()
        for obj in block.component_data_objects(
                Disjunction,
                sort=SortComponents.deterministic,
                descend_into=(Block)):
            if obj.algebraic_constraint is None:
                # This could be bad if it's active since that means its
                # untransformed, but we'll wait to yell until the next loop
                continue
            # get this disjunction's relaxation block.
            transBlock = obj.algebraic_constraint().parent_block()

            self._transfer_var_references(transBlock, destinationBlock)

        # Look through the component map of block and transform everything we
        # have a handler for. Yell if we don't know how to handle it. (Note that
        # because we only iterate through active components, this means
        # non-ActiveComponent types cannot have handlers.)
        for obj in block.component_objects(active=True, descend_into=False):
            handler = self.handlers.get(obj.ctype, None)
            if not handler:
                if handler is None:
                    raise GDP_Error(
                        "No hull transformation handler registered "
                        "for modeling components of type %s. If your "
                        "disjuncts contain non-GDP Pyomo components that "
                        "require transformation, please transform them first."
                        % obj.ctype )
                continue
            # obj is what we are transforming, we pass disjunct
            # through so that we will have access to the indicator
            # variables down the line.
            handler(obj, disjunct, var_substitute_map, zero_substitute_map)

    def _transfer_var_references(self, fromBlock, toBlock):
        disjunctList = toBlock.relaxedDisjuncts
        for idx, disjunctBlock in iteritems(fromBlock.relaxedDisjuncts):
            # move all the of the local var references
            newblock = disjunctList[len(disjunctList)]
            newblock.localVarReferences = Block()
            newblock.localVarReferences.transfer_attributes_from(
                disjunctBlock.localVarReferences)

    def _warn_for_active_disjunction( self, disjunction, disjunct,
                                      var_substitute_map, zero_substitute_map):
        _warn_for_active_disjunction(disjunction, disjunct, NAME_BUFFER)

    def _warn_for_active_disjunct( self, innerdisjunct, outerdisjunct,
                                   var_substitute_map, zero_substitute_map):
        _warn_for_active_disjunct(innerdisjunct, outerdisjunct, NAME_BUFFER)

    def _warn_for_active_logical_statement(
            self, logical_statment, disjunct, var_substitute_map,
            zero_substitute_map):
        _warn_for_active_logical_constraint(logical_statment, disjunct,
                                            NAME_BUFFER)

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
            newConstraint = Constraint(obj.index_set(), transBlock.lbub)
        else:
            newConstraint = Constraint(transBlock.lbub)
        relaxationBlock.add_component(name, newConstraint)
        # map the containers:
        # add mapping of original constraint to transformed constraint
        if obj.is_indexed():
            constraintMap['transformedConstraints'][obj] = newConstraint
        # add mapping of transformed constraint container back to original
        # constraint container (or SimpleConstraint)
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
                    raise RuntimeError("Unknown NL Hull mode")
            else:
                expr = clone_without_expression_components(
                    c.body, substitute=var_substitute_map)

            if c.equality:
                if NL:
                    # ESJ TODO: This can't happen right? This is the only
                    # obvious case where someone has messed up, but this has to
                    # be nonconvex, right? Shouldn't we tell them?
                    newConsExpr = expr == c.lower*y
                else:
                    v = list(EXPR.identify_variables(expr))
                    if len(v) == 1 and not c.lower:
                        # Setting a variable to 0 in a disjunct is
                        # *very* common.  We should recognize that in
                        # that structure, the disaggregated variable
                        # will also be fixed to 0.
                        v[0].fix(0)
                        # ESJ: If you ask where the transformed constraint is,
                        # the answer is nowhere. Really, it is in the bounds of
                        # this variable, so I'm going to return
                        # it. Alternatively we could return an empty list, but I
                        # think I like this better.
                        constraintMap['transformedConstraints'][c] = [v[0]]
                        # Reverse map also (this is strange)
                        constraintMap['srcConstraints'][v[0]] = c
                        continue
                    newConsExpr = expr - (1-y)*h_0 == c.lower*y

                if obj.is_indexed():
                    newConstraint.add((i, 'eq'), newConsExpr)
                    # map the _ConstraintDatas (we mapped the container above)
                    constraintMap[
                        'transformedConstraints'][c] = [newConstraint[i,'eq']]
                    constraintMap['srcConstraints'][newConstraint[i,'eq']] = c
                else:
                    newConstraint.add('eq', newConsExpr)
                    # map to the _ConstraintData (And yes, for
                    # SimpleConstraints, this is overwriting the map to the
                    # container we made above, and that is what I want to
                    # happen. SimpleConstraints will map to lists. For
                    # IndexedConstraints, we can map the container to the
                    # container, but more importantly, we are mapping the
                    # _ConstraintDatas to each other above)
                    constraintMap[
                        'transformedConstraints'][c] = [newConstraint['eq']]
                    constraintMap['srcConstraints'][newConstraint['eq']] = c

                continue

            if c.lower is not None:
                if self._generate_debug_messages:
                    _name = c.getname(
                        fully_qualified=True, name_buffer=NAME_BUFFER)
                    logger.debug("GDP(Hull): Transforming constraint " +
                                 "'%s'", _name)
                if NL:
                    newConsExpr = expr >= c.lower*y
                else:
                    newConsExpr = expr - (1-y)*h_0 >= c.lower*y

                if obj.is_indexed():
                    newConstraint.add((i, 'lb'), newConsExpr)
                    constraintMap[
                        'transformedConstraints'][c] = [newConstraint[i,'lb']]
                    constraintMap['srcConstraints'][newConstraint[i,'lb']] = c
                else:
                    newConstraint.add('lb', newConsExpr)
                    constraintMap[
                        'transformedConstraints'][c] = [newConstraint['lb']]
                    constraintMap['srcConstraints'][newConstraint['lb']] = c

            if c.upper is not None:
                if self._generate_debug_messages:
                    _name = c.getname(
                        fully_qualified=True, name_buffer=NAME_BUFFER)
                    logger.debug("GDP(Hull): Transforming constraint " +
                                 "'%s'", _name)
                if NL:
                    newConsExpr = expr <= c.upper*y
                else:
                    newConsExpr = expr - (1-y)*h_0 <= c.upper*y

                if obj.is_indexed():
                    newConstraint.add((i, 'ub'), newConsExpr)
                    # map (have to account for fact we might have created list
                    # above
                    transformed = constraintMap['transformedConstraints'].get(c)
                    if transformed is not None:
                        transformed.append(newConstraint[i,'ub'])
                    else:
                        constraintMap['transformedConstraints'][
                            c] = [newConstraint[i,'ub']]
                    constraintMap['srcConstraints'][newConstraint[i,'ub']] = c
                else:
                    newConstraint.add('ub', newConsExpr)
                    transformed = constraintMap['transformedConstraints'].get(c)
                    if transformed is not None:
                        transformed.append(newConstraint['ub'])
                    else:
                        constraintMap['transformedConstraints'][
                            c] = [newConstraint['ub']]
                    constraintMap['srcConstraints'][newConstraint['ub']] = c

        # deactivate now that we have transformed
        obj.deactivate()

    def _add_local_var_suffix(self, disjunct):
        # If the Suffix is there, we will borrow it. If not, we make it. If it's
        # something else, we complain.
        localSuffix = disjunct.component("LocalVars")
        if localSuffix is None:
            disjunct.LocalVars = Suffix(direction=Suffix.LOCAL)
        else:
            if localSuffix.ctype is Suffix:
                return
            raise GDP_Error("A component called 'LocalVars' is declared on "
                            "Disjunct %s, but it is of type %s, not Suffix."  
                            % (disjunct.getname(fully_qualified=True,
                                                name_buffer=NAME_BUFFER), 
                               localSuffix.ctype))

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

    def get_disaggregated_var(self, v, disjunct):
        """
        Returns the disaggregated variable corresponding to the Var v and the
        Disjunct disjunct.

        If v is a local variable, this method will return v.

        Parameters
        ----------
        v: a Var which appears in a constraint in a transformed Disjunct
        disjunct: a transformed Disjunct in which v appears
        """
        if disjunct._transformation_block is None:
            raise GDP_Error("Disjunct '%s' has not been transformed"
                            % disjunct.name)
        transBlock = disjunct._transformation_block()
        try:
            return transBlock._disaggregatedVarMap['disaggregatedVar'][v]
        except:
            logger.error("It does not appear '%s' is a "
                         "variable which appears in disjunct '%s'"
                         % (v.name, disjunct.name))
            raise

    def get_src_var(self, disaggregated_var):
        """
        Returns the original model variable to which disaggregated_var
        corresponds.

        Parameters
        ----------
        disaggregated_var: a Var which was created by the hull
                           transformation as a disaggregated variable
                           (and so appears on a transformation block
                           of some Disjunct)
        """
        transBlock = disaggregated_var.parent_block()
        try:
            return transBlock.parent_block()._disaggregatedVarMap[
                'srcVar'][disaggregated_var]
        except:
            logger.error("'%s' does not appear to be a disaggregated variable"
                         % disaggregated_var.name)
            raise

    # retrieves the disaggregation constraint for original_var resulting from
    # transforming disjunction
    def get_disaggregation_constraint(self, original_var, disjunction):
        """
        Returns the disaggregation (re-aggregation?) constraint
        (which links the disaggregated variables to their original)
        corresponding to original_var and the transformation of disjunction.

        Parameters
        ----------
        original_var: a Var which was disaggregated in the transformation
                      of Disjunction disjunction
        disjunction: a transformed Disjunction containing original_var
        """
        for disjunct in disjunction.disjuncts:
            transBlock = disjunct._transformation_block
            if transBlock is not None:
                break
        if transBlock is None:
            raise GDP_Error("Disjunction '%s' has not been properly transformed:"
                            " None of its disjuncts are transformed."
                            % disjunction.name)

        try:
            return transBlock().parent_block()._disaggregationConstraintMap[
                original_var][disjunction]
        except:
            logger.error("It doesn't appear that '%s' is a variable that was "
                         "disaggregated by Disjunction '%s'" %
                         (original_var.name, disjunction.name))
            raise

    def get_var_bounds_constraint(self, v):
        """
        Returns the IndexedConstraint which sets a disaggregated
        variable to be within its bounds when its Disjunct is active and to
        be 0 otherwise. (It is always an IndexedConstraint because each
        bound becomes a separate constraint.)

        Parameters
        ----------
        v: a Var which was created by the hull  transformation as a
           disaggregated variable (and so appears on a transformation
           block of some Disjunct)
        """
        # This can only go well if v is a disaggregated var
        transBlock = v.parent_block().parent_block()
        try:
            return transBlock._bigMConstraintMap[v]
        except:
            logger.error("Either '%s' is not a disaggregated variable, or "
                         "the disjunction that disaggregates it has not "
                         "been properly transformed." % v.name)
            raise


@TransformationFactory.register(
    'gdp.chull',
    doc="Deprecated name for the hull reformulation. Please use 'gdp.hull'.")
class _Deprecated_Name_Hull(Hull_Reformulation):
    @deprecated("The 'gdp.chull' name is deprecated. Please use the more apt 'gdp.hull' instead.",
                logger='pyomo.gdp',
                version="5.7")
    def __init__(self):
        super(_Deprecated_Name_Hull, self).__init__()
