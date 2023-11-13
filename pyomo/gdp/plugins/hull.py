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

import logging

from collections import defaultdict

import pyomo.common.config as cfg
from pyomo.common import deprecated
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.modeling import unique_component_name
from pyomo.core.expr.numvalue import ZeroConstant
import pyomo.core.expr as EXPR
from pyomo.core.base import TransformationFactory, Reference
from pyomo.core import (
    Block,
    BooleanVar,
    Connector,
    Constraint,
    Param,
    Set,
    SetOf,
    Suffix,
    Var,
    Expression,
    SortComponents,
    TraversalStrategy,
    Any,
    RangeSet,
    Reals,
    value,
    NonNegativeIntegers,
    Binary,
)
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.disjunct import _DisjunctData
from pyomo.gdp.plugins.gdp_to_mip_transformation import GDP_to_MIP_Transformation
from pyomo.gdp.transformed_disjunct import _TransformedDisjunct
from pyomo.gdp.util import (
    clone_without_expression_components,
    is_child_of,
    _warn_for_active_disjunct,
)
from pyomo.core.util import target_list
from weakref import ref as weakref_ref

logger = logging.getLogger('pyomo.gdp.hull')

from pytest import set_trace

@TransformationFactory.register(
    'gdp.hull', doc="Relax disjunctive model by forming the hull reformulation."
)
class Hull_Reformulation(GDP_to_MIP_Transformation):
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
    name beginning "_pyomo_gdp_hull_reformulation".
    The block will have a dictionary "_disaggregatedVarMap:
        'srcVar': ComponentMap(<src var>:<disaggregated var>),
        'disaggregatedVar': ComponentMap(<disaggregated var>:<src var>)

    It will also have a ComponentMap "_bigMConstraintMap":

        <disaggregated var>:<bounds constraint>

    Last, it will contain an indexed Block named "relaxedDisjuncts",
    which will hold the relaxed disjuncts.  This block is indexed by
    an integer indicating the order in which the disjuncts were relaxed.
    Each block has a dictionary "_constraintMap":

        'srcConstraints': ComponentMap(<transformed constraint>:
                                       <src constraint>),
        'transformedConstraints':
            ComponentMap(<src constraint container> :
                         <transformed constraint container>,
                         <src constraintData> : [<transformed constraintDatas>])

    All transformed Disjuncts will have a pointer to the block their transformed
    constraints are on, and all transformed Disjunctions will have a
    pointer to the corresponding OR or XOR constraint.

    The _pyomo_gdp_hull_reformulation block will have a ComponentMap
    "_disaggregationConstraintMap":
        <src var>:ComponentMap(<srcDisjunction>: <disaggregation constraint>)

    """

    CONFIG = cfg.ConfigDict('gdp.hull')
    CONFIG.declare(
        'targets',
        cfg.ConfigValue(
            default=None,
            domain=target_list,
            description="target or list of targets that will be relaxed",
            doc="""

        This specifies the target or list of targets to relax as either a
        component or a list of components. If None (default), the entire model
        is transformed. Note that if the transformation is done out of place,
        the list of targets should be attached to the model before it is cloned,
        and the list will specify the targets on the cloned instance.""",
        ),
    )
    CONFIG.declare(
        'perspective function',
        cfg.ConfigValue(
            default='FurmanSawayaGrossmann',
            domain=cfg.In(['FurmanSawayaGrossmann', 'LeeGrossmann', 'GrossmannLee']),
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
        """,
        ),
    )
    CONFIG.declare(
        'EPS',
        cfg.ConfigValue(
            default=1e-4,
            domain=cfg.PositiveFloat,
            description="Epsilon value to use in perspective function",
        ),
    )
    CONFIG.declare(
        'assume_fixed_vars_permanent',
        cfg.ConfigValue(
            default=False,
            domain=bool,
            description="Boolean indicating whether or not to transform so that "
            "the transformed model will still be valid when fixed Vars are "
            "unfixed.",
            doc="""
        If True, the transformation will not disaggregate fixed variables.
        This means that if a fixed variable is unfixed after transformation,
        the transformed model is no longer valid. By default, the transformation
        will disagregate fixed variables so that any later fixing and unfixing
        will be valid in the transformed model.
        """,
        ),
    )
    transformation_name = 'hull'

    def __init__(self):
        super().__init__(logger)
        self._targets = set()

    def _collect_local_vars_from_block(self, block, local_var_dict):
        localVars = block.component('LocalVars')
        if localVars is not None and localVars.ctype is Suffix:
            for disj, var_list in localVars.items():
                local_var_dict[disj].update(var_list)

    def _get_user_defined_local_vars(self, targets):
        user_defined_local_vars = defaultdict(lambda: ComponentSet())
        seen_blocks = set()
        # we go through the targets looking both up and down the hierarchy, but
        # we cache what Blocks/Disjuncts we've already looked on so that we
        # don't duplicate effort.
        for t in targets:
            if t.ctype is Disjunct or isinstance(t, _DisjunctData):
                # first look beneath where we are (there could be Blocks on this
                # disjunct)
                for b in t.component_data_objects(Block, descend_into=Block,
                                                  active=True,
                                                  sort=SortComponents.deterministic
                                              ):
                    if b not in seen_blocks:
                        self._collect_local_vars_from_block(b, user_defined_local_vars)
                        seen_blocks.add(b)
                # now look up in the tree
                blk = t
                while blk is not None:
                    if blk not in seen_blocks:
                        self._collect_local_vars_from_block(blk,
                                                            user_defined_local_vars)
                        seen_blocks.add(blk)
                    blk = blk.parent_block()
        return user_defined_local_vars

    # def _get_local_vars_from_suffixes(self, block, local_var_dict):
    #     # You can specify suffixes on any block (disjuncts included). This
    #     # method starts from a Disjunct (presumably) and checks for a LocalVar
    #     # suffixes going both up and down the tree, adding them into the
    #     # dictionary that is the second argument.

    #     # first look beneath where we are (there could be Blocks on this
    #     # disjunct)
    #     for b in block.component_data_objects(
    #         Block, descend_into=Block, active=True, sort=SortComponents.deterministic
    #     ):
    #         self._collect_local_vars_from_block(b, local_var_dict)
    #     # now traverse upwards and get what's above
    #     while block is not None:
    #         self._collect_local_vars_from_block(block, local_var_dict)
    #         block = block.parent_block()

    def _apply_to(self, instance, **kwds):
        try:
            self._apply_to_impl(instance, **kwds)
        finally:
            self._restore_state()
            self._transformation_blocks.clear()
            self._algebraic_constraints.clear()

    def _apply_to_impl(self, instance, **kwds):
        self._process_arguments(instance, **kwds)

        # filter out inactive targets and handle case where targets aren't
        # specified.
        targets = self._filter_targets(instance)
        # transform logical constraints based on targets
        self._transform_logical_constraints(instance, targets)

        # Preprocess in order to find what disjunctive components need
        # transformation
        gdp_tree = self._get_gdp_tree_from_targets(instance, targets)
        # Transform from leaf to root: This is important for hull because for
        # nested GDPs, we will introduce variables that need disaggregating into
        # parent Disjuncts as we transform their child Disjunctions.
        preprocessed_targets = gdp_tree.reverse_topological_sort()
        local_vars_by_disjunct = self._get_user_defined_local_vars(
            preprocessed_targets)

        for t in preprocessed_targets:
            if t.ctype is Disjunction:
                self._transform_disjunctionData(
                    t,
                    t.index(),
                    gdp_tree.parent(t),
                    local_vars_by_disjunct
                )
            # We skip disjuncts now, because we need information from the
            # disjunctions to transform them (which variables to disaggregate),
            # so for hull's purposes, they need not be in the tree.

    def _add_transformation_block(self, to_block):
        transBlock, new_block = super()._add_transformation_block(to_block)
        if not new_block:
            return transBlock, new_block

        transBlock.lbub = Set(initialize=['lb', 'ub', 'eq'])
        # Map between disaggregated variables and their
        # originals
        transBlock._disaggregatedVarMap = {
            'srcVar': ComponentMap(),
            'disaggregatedVar': ComponentMap(),
        }
        # Map between disaggregated variables and their lb*indicator <= var <=
        # ub*indicator constraints
        transBlock._bigMConstraintMap = ComponentMap()
        # We will store all of the disaggregation constraints for any
        # Disjunctions we transform onto this block here.
        transBlock.disaggregationConstraints = Constraint(NonNegativeIntegers)

        # This will map from srcVar to a map of srcDisjunction to the
        # disaggregation constraint corresponding to srcDisjunction
        transBlock._disaggregationConstraintMap = ComponentMap()

        # we are going to store some of the disaggregated vars directly here
        # when we have vars that don't appear in every disjunct
        transBlock._disaggregatedVars = Var(NonNegativeIntegers, dense=False)
        transBlock._boundsConstraints = Constraint(NonNegativeIntegers, transBlock.lbub)

        return transBlock, True

    def _transform_disjunctionData(self, obj, index, parent_disjunct,
                                   local_vars_by_disjunct):
        # Hull reformulation doesn't work if this is an OR constraint. So if
        # xor is false, give up
        if not obj.xor:
            raise GDP_Error(
                "Cannot do hull reformulation for "
                "Disjunction '%s' with OR constraint.  "
                "Must be an XOR!" % obj.name
            )
        # collect the Disjuncts we are going to transform now because we will
        # change their active status when we transform them, but still need this
        # list after the fact.
        active_disjuncts = [disj for disj in obj.disjuncts if disj.active]

        # We put *all* transformed things on the parent Block of this
        # disjunction. We'll mark the disaggregated Vars as local, but beyond
        # that, we actually need everything to get transformed again as we go up
        # the nested hierarchy (if there is one)
        transBlock, xorConstraint = self._setup_transform_disjunctionData(
            obj, root_disjunct=None
        )

        disaggregationConstraint = transBlock.disaggregationConstraints
        disaggregationConstraintMap = transBlock._disaggregationConstraintMap
        disaggregatedVars = transBlock._disaggregatedVars
        disaggregated_var_bounds = transBlock._boundsConstraints

        # We first go through and collect all the variables that we are going to
        # disaggregate. We do this in its own pass because we want to know all
        # the Disjuncts that each Var appears in.
        var_order = ComponentSet()
        disjuncts_var_appears_in = ComponentMap()
        for disjunct in active_disjuncts:
            # create the key for each disjunct now
            transBlock._disaggregatedVarMap['disaggregatedVar'][
                disjunct
            ] = ComponentMap()
            for cons in disjunct.component_data_objects(
                Constraint,
                active=True,
                sort=SortComponents.deterministic,
                descend_into=Block,
            ):
                # [ESJ 02/14/2020] By default, we disaggregate fixed variables
                # on the philosophy that fixing is not a promise for the future
                # and we are mathematically wrong if we don't transform these
                # correctly and someone later unfixes them and keeps playing
                # with their transformed model. However, the user may have set
                # assume_fixed_vars_permanent to True in which case we will skip
                # them
                for var in EXPR.identify_variables(
                        cons.body, include_fixed=not
                        self._config.assume_fixed_vars_permanent):
                    # Note that, because ComponentSets are ordered, we will
                    # eventually disaggregate the vars in a deterministic order
                    # (the order that we found them)
                    if var not in var_order:
                        var_order.add(var)
                        disjuncts_var_appears_in[var] = ComponentSet([disjunct])
                    else:
                        disjuncts_var_appears_in[var].add(disjunct)

        # We will disaggregate all variables that are not explicitly declared as
        # being local. We have marked our own disaggregated variables as local,
        # so they will not be re-disaggregated.
        vars_to_disaggregate = {disj: ComponentSet() for disj in obj.disjuncts}
        all_vars_to_disaggregate = ComponentSet()
        for var in var_order:
            disjuncts = disjuncts_var_appears_in[var]
            # clearly not local if used in more than one disjunct
            if len(disjuncts) > 1:
                if self._generate_debug_messages:
                    logger.debug(
                        "Assuming '%s' is not a local var since it is"
                        "used in multiple disjuncts."
                        % var.getname(fully_qualified=True)
                    )
                for disj in disjuncts:
                    vars_to_disaggregate[disj].add(var)
                    all_vars_to_disaggregate.add(var)
            else: # disjuncts is a set of length 1
                disjunct = next(iter(disjuncts))
                if disjunct in local_vars_by_disjunct:
                    if var not in local_vars_by_disjunct[disjunct]:
                        # It's not declared local to this Disjunct, so we
                        # disaggregate
                        vars_to_disaggregate[disjunct].add(var)
                        all_vars_to_disaggregate.add(var)
                else:
                    # The user didn't declare any local vars for this
                    # Disjunct, so we know we're disaggregating it
                    vars_to_disaggregate[disjunct].add(var)
                    all_vars_to_disaggregate.add(var)

        # Now that we know who we need to disaggregate, we will do it
        # while we also transform the disjuncts.
        parent_local_var_list = self._get_local_var_list(parent_disjunct)
        or_expr = 0
        for disjunct in obj.disjuncts:
            or_expr += disjunct.indicator_var.get_associated_binary()
            if obj.active:
                self._transform_disjunct(
                    disjunct,
                    transBlock,
                    vars_to_disaggregate[disjunct],
                    local_vars_by_disjunct.get(disjunct, []),
                    parent_local_var_list,
                    local_vars_by_disjunct[parent_disjunct]
                )
        xorConstraint.add(index, (or_expr, 1))
        # map the DisjunctionData to its XOR constraint to mark it as
        # transformed
        obj._algebraic_constraint = weakref_ref(xorConstraint[index])

        # add the reaggregation constraints
        i = 0
        for var in all_vars_to_disaggregate:
            # There are two cases here: Either the var appeared in every
            # disjunct in the disjunction, or it didn't. If it did, there's
            # nothing special to do: All of the disaggregated variables have
            # been created, and we can just proceed and make this constraint. If
            # it didn't, we need one more disaggregated variable, correctly
            # defined. And then we can make the constraint.
            if len(disjuncts_var_appears_in[var]) < len(active_disjuncts):
                # create one more disaggregated var
                idx = len(disaggregatedVars)
                disaggregated_var = disaggregatedVars[idx]
                # mark this as local because we won't re-disaggregate if this is
                # a nested disjunction
                if parent_local_var_list is not None:
                    parent_local_var_list.append(disaggregated_var)
                local_vars_by_disjunct[parent_disjunct].add(disaggregated_var)
                var_free = 1 - sum(
                    disj.indicator_var.get_associated_binary()
                    for disj in disjuncts_var_appears_in[var]
                )
                self._declare_disaggregated_var_bounds(
                    var,
                    disaggregated_var,
                    obj,
                    disaggregated_var_bounds,
                    (idx, 'lb'),
                    (idx, 'ub'),
                    var_free,
                )
                # For every Disjunct the Var does not appear in, we want to map
                # that this new variable is its disaggreggated variable.
                for disj in active_disjuncts:
                    # Because we called _transform_disjunct above, we know that
                    # if this isn't transformed it is because it was cleanly
                    # deactivated, and we can just skip it.
                    if (
                        disj._transformation_block is not None
                        and disj not in disjuncts_var_appears_in[var]
                    ):
                        relaxationBlock = disj._transformation_block().\
                                          parent_block()
                        relaxationBlock._bigMConstraintMap[
                            disaggregated_var
                        ] = Reference(disaggregated_var_bounds[idx, :])
                        relaxationBlock._disaggregatedVarMap['srcVar'][
                            disaggregated_var
                        ] = var
                        relaxationBlock._disaggregatedVarMap[
                            'disaggregatedVar'][disj][
                            var
                        ] = disaggregated_var

                disaggregatedExpr = disaggregated_var
            else:
                disaggregatedExpr = 0
            for disjunct in disjuncts_var_appears_in[var]:
                # We know this Disjunct was active, so it has been transformed now.
                disaggregatedVar = (
                    disjunct._transformation_block()
                    .parent_block()
                    ._disaggregatedVarMap['disaggregatedVar'][disjunct][var]
                )
                disaggregatedExpr += disaggregatedVar

            cons_idx = len(disaggregationConstraint)
            # We always aggregate to the original var. If this is nested, this
            # constraint will be transformed again.
            disaggregationConstraint.add(cons_idx, var == disaggregatedExpr)
            # and update the map so that we can find this later. We index by
            # variable and the particular disjunction because there is a
            # different one for each disjunction
            if disaggregationConstraintMap.get(var) is not None:
                disaggregationConstraintMap[var][obj] = disaggregationConstraint[
                    cons_idx
                ]
            else:
                thismap = disaggregationConstraintMap[var] = ComponentMap()
                thismap[obj] = disaggregationConstraint[cons_idx]

            i += 1

        # deactivate for the writers
        obj.deactivate()

    def _transform_disjunct(self, obj, transBlock, vars_to_disaggregate, local_vars,
                            parent_local_var_suffix, parent_disjunct_local_vars):
        print("\nTransforming '%s'" % obj.name)
        relaxationBlock = self._get_disjunct_transformation_block(obj, transBlock)

        # Put the disaggregated variables all on their own block so that we can
        # isolate the name collisions and still have complete control over the
        # names on this block.
        relaxationBlock.disaggregatedVars = Block()

        # add the disaggregated variables and their bigm constraints
        # to the relaxationBlock
        for var in vars_to_disaggregate:
            disaggregatedVar = Var(within=Reals, initialize=var.value)
            # naming conflicts are possible here since this is a bunch
            # of variables from different blocks coming together, so we
            # get a unique name
            disaggregatedVarName = unique_component_name(
                relaxationBlock.disaggregatedVars, var.getname(fully_qualified=True)
            )
            relaxationBlock.disaggregatedVars.add_component(
                disaggregatedVarName, disaggregatedVar
            )
            # mark this as local via the Suffix in case this is a partial
            # transformation:
            if parent_local_var_suffix is not None:
                parent_local_var_suffix.append(disaggregatedVar)
            # Record that it's local for our own bookkeeping in case we're in a
            # nested situation in *this* transformation
            parent_disjunct_local_vars.add(disaggregatedVar)

            # add the bigm constraint
            bigmConstraint = Constraint(transBlock.lbub)
            relaxationBlock.add_component(
                disaggregatedVarName + "_bounds", bigmConstraint
            )

            print("Adding bounds constraints for '%s'" % var)
            self._declare_disaggregated_var_bounds(
                var,
                disaggregatedVar,
                obj,
                bigmConstraint,
                'lb',
                'ub',
                obj.indicator_var.get_associated_binary(),
                transBlock,
            )

        for var in local_vars:
            if var in vars_to_disaggregate:
                logger.warning(
                    "Var '%s' was declared as a local Var for Disjunct '%s', "
                    "but it appeared in multiple Disjuncts, so it will be "
                    "disaggregated." % (var.name, obj.name))
                continue
            # we don't need to disaggregate, i.e., we can use this Var, but we
            # do need to set up its bounds constraints.

            # naming conflicts are possible here since this is a bunch
            # of variables from different blocks coming together, so we
            # get a unique name
            conName = unique_component_name(
                relaxationBlock, var.getname(fully_qualified=False) + "_bounds"
            )
            bigmConstraint = Constraint(transBlock.lbub)
            relaxationBlock.add_component(conName, bigmConstraint)

            print("Adding bounds constraints for local var '%s'" % var)
            # TODO: This gets mapped in a place where we can't find it if we ask
            # for it from the local var itself.
            self._declare_disaggregated_var_bounds(
                var,
                var,
                obj,
                bigmConstraint,
                'lb',
                'ub',
                obj.indicator_var.get_associated_binary(),
                transBlock,
            )

        var_substitute_map = dict(
            (id(v), newV)
            for v, newV in transBlock._disaggregatedVarMap['disaggregatedVar'][
                obj
            ].items()
        )
        zero_substitute_map = dict(
            (id(v), ZeroConstant)
            for v, newV in transBlock._disaggregatedVarMap['disaggregatedVar'][
                obj
            ].items()
        )
        zero_substitute_map.update((id(v), ZeroConstant) for v in local_vars)

        # Transform each component within this disjunct
        self._transform_block_components(
            obj, obj, var_substitute_map, zero_substitute_map
        )

        # Anything that was local to this Disjunct is also local to the parent,
        # and just got "promoted" up there, so to speak.
        parent_disjunct_local_vars.update(local_vars)
        # deactivate disjunct so writers can be happy
        obj._deactivate_without_fixing_indicator()

    def _declare_disaggregated_var_bounds(
        self,
        original_var,
        disaggregatedVar,
        disjunct,
        bigmConstraint,
        lb_idx,
        ub_idx,
        var_free_indicator,
        transBlock=None,
    ):
        # If transBlock is None then this is a disaggregated variable for
        # multiple Disjuncts and we will handle the mappings separately.
        lb = original_var.lb
        ub = original_var.ub
        if lb is None or ub is None:
            raise GDP_Error(
                "Variables that appear in disjuncts must be "
                "bounded in order to use the hull "
                "transformation! Missing bound for %s." % (original_var.name)
            )

        disaggregatedVar.setlb(min(0, lb))
        disaggregatedVar.setub(max(0, ub))

        if lb:
            bigmConstraint.add(lb_idx, var_free_indicator * lb <= disaggregatedVar)
        if ub:
            bigmConstraint.add(ub_idx, disaggregatedVar <= ub * var_free_indicator)

        # store the mappings from variables to their disaggregated selves on
        # the transformation block.
        if transBlock is not None:
            transBlock._disaggregatedVarMap['disaggregatedVar'][disjunct][
                original_var
            ] = disaggregatedVar
            transBlock._disaggregatedVarMap['srcVar'][disaggregatedVar] = original_var
            transBlock._bigMConstraintMap[disaggregatedVar] = bigmConstraint

    def _get_local_var_list(self, parent_disjunct):
        # Add or retrieve Suffix from parent_disjunct so that, if this is
        # nested, we can use it to declare that the disaggregated variables are
        # local. We return the list so that we can add to it.
        local_var_list = None
        if parent_disjunct is not None:
            # This limits the cases that a user is allowed to name something
            # (other than a Suffix) 'LocalVars' on a Disjunct. But I am assuming
            # that the Suffix has to be somewhere above the disjunct in the
            # tree, so I can't put it on a Block that I own. And if I'm coopting
            # something of theirs, it may as well be here.
            self._get_local_var_suffix(parent_disjunct)
            if parent_disjunct.LocalVars.get(parent_disjunct) is None:
                parent_disjunct.LocalVars[parent_disjunct] = []
            local_var_list = parent_disjunct.LocalVars[parent_disjunct]

        return local_var_list

    def _transform_constraint(
        self, obj, disjunct, var_substitute_map, zero_substitute_map
    ):
        # we will put a new transformed constraint on the relaxation block.
        relaxationBlock = disjunct._transformation_block()
        constraintMap = relaxationBlock._constraintMap

        # We will make indexes from ({obj.local_name} x obj.index_set() x ['lb',
        # 'ub']), but don't bother construct that set here, as taking Cartesian
        # products is kind of expensive (and redundant since we have the
        # original model)
        newConstraint = relaxationBlock.transformedConstraints

        for i in sorted(obj.keys()):
            c = obj[i]
            if not c.active:
                continue

            unique = len(newConstraint)
            name = c.local_name + "_%s" % unique

            NL = c.body.polynomial_degree() not in (0, 1)
            EPS = self._config.EPS
            mode = self._config.perspective_function

            # We need to evaluate the expression at the origin *before*
            # we substitute the expression variables with the
            # disaggregated variables
            if not NL or mode == "FurmanSawayaGrossmann":
                h_0 = clone_without_expression_components(
                    c.body, substitute=zero_substitute_map
                )

            y = disjunct.binary_indicator_var
            if NL:
                if mode == "LeeGrossmann":
                    sub_expr = clone_without_expression_components(
                        c.body,
                        substitute=dict(
                            (var, subs / y) for var, subs in var_substitute_map.items()
                        ),
                    )
                    expr = sub_expr * y
                elif mode == "GrossmannLee":
                    sub_expr = clone_without_expression_components(
                        c.body,
                        substitute=dict(
                            (var, subs / (y + EPS))
                            for var, subs in var_substitute_map.items()
                        ),
                    )
                    expr = (y + EPS) * sub_expr
                elif mode == "FurmanSawayaGrossmann":
                    sub_expr = clone_without_expression_components(
                        c.body,
                        substitute=dict(
                            (var, subs / ((1 - EPS) * y + EPS))
                            for var, subs in var_substitute_map.items()
                        ),
                    )
                    expr = ((1 - EPS) * y + EPS) * sub_expr - EPS * h_0 * (1 - y)
                else:
                    raise RuntimeError("Unknown NL Hull mode")
            else:
                expr = clone_without_expression_components(
                    c.body, substitute=var_substitute_map
                )

            if c.equality:
                if NL:
                    # ESJ TODO: This can't happen right? This is the only
                    # obvious case where someone has messed up, but this has to
                    # be nonconvex, right? Shouldn't we tell them?
                    newConsExpr = expr == c.lower * y
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
                    newConsExpr = expr - (1 - y) * h_0 == c.lower * y

                if obj.is_indexed():
                    newConstraint.add((name, i, 'eq'), newConsExpr)
                    # map the _ConstraintDatas (we mapped the container above)
                    constraintMap['transformedConstraints'][c] = [
                        newConstraint[name, i, 'eq']
                    ]
                    constraintMap['srcConstraints'][newConstraint[name, i, 'eq']] = c
                else:
                    newConstraint.add((name, 'eq'), newConsExpr)
                    # map to the _ConstraintData (And yes, for
                    # ScalarConstraints, this is overwriting the map to the
                    # container we made above, and that is what I want to
                    # happen. ScalarConstraints will map to lists. For
                    # IndexedConstraints, we can map the container to the
                    # container, but more importantly, we are mapping the
                    # _ConstraintDatas to each other above)
                    constraintMap['transformedConstraints'][c] = [
                        newConstraint[name, 'eq']
                    ]
                    constraintMap['srcConstraints'][newConstraint[name, 'eq']] = c

                continue

            if c.lower is not None:
                if self._generate_debug_messages:
                    _name = c.getname(fully_qualified=True)
                    logger.debug("GDP(Hull): Transforming constraint " + "'%s'", _name)
                if NL:
                    newConsExpr = expr >= c.lower * y
                else:
                    newConsExpr = expr - (1 - y) * h_0 >= c.lower * y

                if obj.is_indexed():
                    newConstraint.add((name, i, 'lb'), newConsExpr)
                    constraintMap['transformedConstraints'][c] = [
                        newConstraint[name, i, 'lb']
                    ]
                    constraintMap['srcConstraints'][newConstraint[name, i, 'lb']] = c
                else:
                    newConstraint.add((name, 'lb'), newConsExpr)
                    constraintMap['transformedConstraints'][c] = [
                        newConstraint[name, 'lb']
                    ]
                    constraintMap['srcConstraints'][newConstraint[name, 'lb']] = c

            if c.upper is not None:
                if self._generate_debug_messages:
                    _name = c.getname(fully_qualified=True)
                    logger.debug("GDP(Hull): Transforming constraint " + "'%s'", _name)
                if NL:
                    newConsExpr = expr <= c.upper * y
                else:
                    newConsExpr = expr - (1 - y) * h_0 <= c.upper * y

                if obj.is_indexed():
                    newConstraint.add((name, i, 'ub'), newConsExpr)
                    # map (have to account for fact we might have created list
                    # above
                    transformed = constraintMap['transformedConstraints'].get(c)
                    if transformed is not None:
                        transformed.append(newConstraint[name, i, 'ub'])
                    else:
                        constraintMap['transformedConstraints'][c] = [
                            newConstraint[name, i, 'ub']
                        ]
                    constraintMap['srcConstraints'][newConstraint[name, i, 'ub']] = c
                else:
                    newConstraint.add((name, 'ub'), newConsExpr)
                    transformed = constraintMap['transformedConstraints'].get(c)
                    if transformed is not None:
                        transformed.append(newConstraint[name, 'ub'])
                    else:
                        constraintMap['transformedConstraints'][c] = [
                            newConstraint[name, 'ub']
                        ]
                    constraintMap['srcConstraints'][newConstraint[name, 'ub']] = c

        # deactivate now that we have transformed
        obj.deactivate()

    def _get_local_var_suffix(self, disjunct):
        # If the Suffix is there, we will borrow it. If not, we make it. If it's
        # something else, we complain.
        localSuffix = disjunct.component("LocalVars")
        if localSuffix is None:
            disjunct.LocalVars = Suffix(direction=Suffix.LOCAL)
        else:
            if localSuffix.ctype is Suffix:
                return
            raise GDP_Error(
                "A component called 'LocalVars' is declared on "
                "Disjunct %s, but it is of type %s, not Suffix."
                % (disjunct.getname(fully_qualified=True), localSuffix.ctype)
            )

    def get_disaggregated_var(self, v, disjunct, raise_exception=True):
        """
        Returns the disaggregated variable corresponding to the Var v and the
        Disjunct disjunct.

        If v is a local variable, this method will return v.

        Parameters
        ----------
        v: a Var that appears in a constraint in a transformed Disjunct
        disjunct: a transformed Disjunct in which v appears
        """
        if disjunct._transformation_block is None:
            raise GDP_Error("Disjunct '%s' has not been transformed" % disjunct.name)
        transBlock = disjunct._transformation_block().parent_block()
        try:
            return transBlock._disaggregatedVarMap['disaggregatedVar'][disjunct][v]
        except:
            if raise_exception:
                logger.error(
                    "It does not appear '%s' is a "
                    "variable that appears in disjunct '%s'" % (v.name, disjunct.name)
                )
                raise
            return none

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
        msg = (
            "'%s' does not appear to be a "
            "disaggregated variable" % disaggregated_var.name
        )
        # There are two possibilities: It is declared on a Disjunct
        # transformation Block, or it is declared on the parent of a Disjunct
        # transformation block (if it is a single variable for multiple
        # Disjuncts the original doesn't appear in)
        transBlock = disaggregated_var.parent_block()
        if not hasattr(transBlock, '_disaggregatedVarMap'):
            try:
                transBlock = transBlock.parent_block().parent_block()
            except:
                logger.error(msg)
                raise
        try:
            return transBlock._disaggregatedVarMap['srcVar'][disaggregated_var]
        except:
            logger.error(msg)
            raise

    # retrieves the disaggregation constraint for original_var resulting from
    # transforming disjunction
    def get_disaggregation_constraint(self, original_var, disjunction,
                                      raise_exception=True):
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
            transBlock = disjunct.transformation_block
            if transBlock is not None:
                break
        if transBlock is None:
            raise GDP_Error(
                "Disjunction '%s' has not been properly "
                "transformed:"
                " None of its disjuncts are transformed." % disjunction.name
            )

        try:
            cons = (
                transBlock
                .parent_block()
                ._disaggregationConstraintMap[original_var][disjunction]
            )
        except:
            if raise_exception:
                logger.error(
                    "It doesn't appear that '%s' is a variable that was "
                    "disaggregated by Disjunction '%s'"
                    % (original_var.name, disjunction.name)
                )
                raise
            return None
        while not cons.active:
            cons = self.get_transformed_constraints(cons)[0]
        return cons

    def get_var_bounds_constraint(self, v):
        """
        Returns the IndexedConstraint which sets a disaggregated
        variable to be within its bounds when its Disjunct is active and to
        be 0 otherwise. (It is always an IndexedConstraint because each
        bound becomes a separate constraint.)

        Parameters
        ----------
        v: a Var that was created by the hull transformation as a
           disaggregated variable (and so appears on a transformation
           block of some Disjunct)
        """
        msg = (
            "Either '%s' is not a disaggregated variable, or "
            "the disjunction that disaggregates it has not "
            "been properly transformed." % v.name
        )
        # This can only go well if v is a disaggregated var
        transBlock = v.parent_block()
        if not hasattr(transBlock, '_bigMConstraintMap'):
            try:
                transBlock = transBlock.parent_block().parent_block()
            except:
                logger.error(msg)
                raise
        try:
            cons = transBlock._bigMConstraintMap[v]
        except:
            logger.error(msg)
            raise
        transformed_cons = {key: con for key, con in cons.items()}
        def is_active(cons):
            return all(c.active for c in cons.values())
        while not is_active(transformed_cons):
            if 'lb' in transformed_cons:
                transformed_cons['lb'] = self.get_transformed_constraints(
                    transformed_cons['lb'])[0]
            if 'ub' in transformed_cons:
                transformed_cons['ub'] = self.get_transformed_constraints(
                    transformed_cons['ub'])[0]
        return transformed_cons

    def get_transformed_constraints(self, cons):
        cons = super().get_transformed_constraints(cons)
        while not cons[0].active:
            transformed_cons = []
            for con in cons:
                transformed_cons += super().get_transformed_constraints(con)
            cons = transformed_cons
        return cons


@TransformationFactory.register(
    'gdp.chull',
    doc="[DEPRECATED] please use 'gdp.hull' to get the Hull transformation.",
)
@deprecated(
    "The 'gdp.chull' name is deprecated. "
    "Please use the more apt 'gdp.hull' instead.",
    logger='pyomo.gdp',
    version="5.7",
)
class _Deprecated_Name_Hull(Hull_Reformulation):
    def __init__(self):
        super(_Deprecated_Name_Hull, self).__init__()
