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

    def _add_local_vars(self, block, local_var_dict):
        localVars = block.component('LocalVars')
        if type(localVars) is Suffix:
            for disj, var_list in localVars.items():
                if local_var_dict.get(disj) is None:
                    local_var_dict[disj] = ComponentSet(var_list)
                else:
                    local_var_dict[disj].update(var_list)

    def _get_local_var_suffixes(self, block, local_var_dict):
        # You can specify suffixes on any block (disjuncts included). This
        # method starts from a Disjunct (presumably) and checks for a LocalVar
        # suffixes going both up and down the tree, adding them into the
        # dictionary that is the second argument.

        # first look beneath where we are (there could be Blocks on this
        # disjunct)
        for b in block.component_data_objects(
            Block, descend_into=(Block), active=True, sort=SortComponents.deterministic
        ):
            self._add_local_vars(b, local_var_dict)
        # now traverse upwards and get what's above
        while block is not None:
            self._add_local_vars(block, local_var_dict)
            block = block.parent_block()

        return local_var_dict

    def _apply_to(self, instance, **kwds):
        try:
            self._apply_to_impl(instance, **kwds)
        finally:
            self._restore_state()
            self._transformation_blocks.clear()
            self._algebraic_constraints.clear()
            self._targets_set = set()

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
        preprocessed_targets = gdp_tree.topological_sort()
        self._targets_set = set(preprocessed_targets)

        for t in preprocessed_targets:
            if t.ctype is Disjunction:
                self._transform_disjunctionData(
                    t,
                    t.index(),
                    parent_disjunct=gdp_tree.parent(t),
                    root_disjunct=gdp_tree.root_disjunct(t),
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

    def _transform_disjunctionData(
        self, obj, index, parent_disjunct=None, root_disjunct=None
    ):
        # Hull reformulation doesn't work if this is an OR constraint. So if
        # xor is false, give up
        if not obj.xor:
            raise GDP_Error(
                "Cannot do hull reformulation for "
                "Disjunction '%s' with OR constraint.  "
                "Must be an XOR!" % obj.name
            )

        transBlock, xorConstraint = self._setup_transform_disjunctionData(
            obj, root_disjunct
        )

        disaggregationConstraint = transBlock.disaggregationConstraints
        disaggregationConstraintMap = transBlock._disaggregationConstraintMap
        disaggregatedVars = transBlock._disaggregatedVars
        disaggregated_var_bounds = transBlock._boundsConstraints

        # We first go through and collect all the variables that we
        # are going to disaggregate.
        varOrder_set = ComponentSet()
        varOrder = []
        varsByDisjunct = ComponentMap()
        localVarsByDisjunct = ComponentMap()
        include_fixed_vars = not self._config.assume_fixed_vars_permanent
        for disjunct in obj.disjuncts:
            if not disjunct.active:
                continue
            disjunctVars = varsByDisjunct[disjunct] = ComponentSet()
            # create the key for each disjunct now
            transBlock._disaggregatedVarMap['disaggregatedVar'][
                disjunct
            ] = ComponentMap()
            for cons in disjunct.component_data_objects(
                Constraint,
                active=True,
                sort=SortComponents.deterministic,
                descend_into=(Block, Disjunct),
            ):
                # [ESJ 02/14/2020] By default, we disaggregate fixed variables
                # on the philosophy that fixing is not a promise for the future
                # and we are mathematically wrong if we don't transform these
                # correctly and someone later unfixes them and keeps playing
                # with their transformed model. However, the user may have set
                # assume_fixed_vars_permanent to True in which case we will skip
                # them
                for var in EXPR.identify_variables(
                    cons.body, include_fixed=include_fixed_vars
                ):
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
                disjunct, localVarsByDisjunct
            )

        # We will disaggregate all variables that are not explicitly declared as
        # being local. Since we transform from leaf to root, we are implicitly
        # treating our own disaggregated variables as local, so they will not be
        # re-disaggregated.
        varSet = []
        varSet = {disj: [] for disj in obj.disjuncts}
        # Note that variables are local with respect to a Disjunct. We deal with
        # them here to do some error checking (if something is obviously not
        # local since it is used in multiple Disjuncts in this Disjunction) and
        # also to get a deterministic order in which to process them when we
        # transform the Disjuncts: Values of localVarsByDisjunct are
        # ComponentSets, so we need this for determinism (we iterate through the
        # localVars of a Disjunct later)
        localVars = ComponentMap()
        varsToDisaggregate = []
        disjunctsVarAppearsIn = ComponentMap()
        for var in varOrder:
            disjuncts = disjunctsVarAppearsIn[var] = [
                d for d in varsByDisjunct if var in varsByDisjunct[d]
            ]
            # clearly not local if used in more than one disjunct
            if len(disjuncts) > 1:
                if self._generate_debug_messages:
                    logger.debug(
                        "Assuming '%s' is not a local var since it is"
                        "used in multiple disjuncts."
                        % var.getname(fully_qualified=True)
                    )
                for disj in disjuncts:
                    varSet[disj].append(var)
                varsToDisaggregate.append(var)
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
                    varSet[disjuncts[0]].append(var)
                    varsToDisaggregate.append(var)
            else:
                # We don't even have have any local vars for this Disjunct.
                varSet[disjuncts[0]].append(var)
                varsToDisaggregate.append(var)

        # Now that we know who we need to disaggregate, we will do it
        # while we also transform the disjuncts.
        local_var_set = self._get_local_var_set(obj)
        or_expr = 0
        for disjunct in obj.disjuncts:
            or_expr += disjunct.indicator_var.get_associated_binary()
            self._transform_disjunct(
                disjunct,
                transBlock,
                varSet[disjunct],
                localVars.get(disjunct, []),
                local_var_set,
            )
        rhs = 1 if parent_disjunct is None else parent_disjunct.binary_indicator_var
        xorConstraint.add(index, (or_expr, rhs))
        # map the DisjunctionData to its XOR constraint to mark it as
        # transformed
        obj._algebraic_constraint = weakref_ref(xorConstraint[index])

        # add the reaggregation constraints
        for i, var in enumerate(varsToDisaggregate):
            # There are two cases here: Either the var appeared in every
            # disjunct in the disjunction, or it didn't. If it did, there's
            # nothing special to do: All of the disaggregated variables have
            # been created, and we can just proceed and make this constraint. If
            # it didn't, we need one more disaggregated variable, correctly
            # defined. And then we can make the constraint.
            if len(disjunctsVarAppearsIn[var]) < len(obj.disjuncts):
                # create one more disaggregated var
                idx = len(disaggregatedVars)
                disaggregated_var = disaggregatedVars[idx]
                # mark this as local because we won't re-disaggregate if this is
                # a nested disjunction
                if local_var_set is not None:
                    local_var_set.append(disaggregated_var)
                var_free = 1 - sum(
                    disj.indicator_var.get_associated_binary()
                    for disj in disjunctsVarAppearsIn[var]
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
                # maintain the mappings
                for disj in obj.disjuncts:
                    # Because we called _transform_disjunct above, we know that
                    # if this isn't transformed it is because it was cleanly
                    # deactivated, and we can just skip it.
                    if (
                        disj._transformation_block is not None
                        and disj not in disjunctsVarAppearsIn[var]
                    ):
                        relaxationBlock = disj._transformation_block().parent_block()
                        relaxationBlock._bigMConstraintMap[
                            disaggregated_var
                        ] = Reference(disaggregated_var_bounds[idx, :])
                        relaxationBlock._disaggregatedVarMap['srcVar'][
                            disaggregated_var
                        ] = var
                        relaxationBlock._disaggregatedVarMap['disaggregatedVar'][disj][
                            var
                        ] = disaggregated_var

                disaggregatedExpr = disaggregated_var
            else:
                disaggregatedExpr = 0
            for disjunct in disjunctsVarAppearsIn[var]:
                if disjunct._transformation_block is None:
                    # Because we called _transform_disjunct above, we know that
                    # if this isn't transformed it is because it was cleanly
                    # deactivated, and we can just skip it.
                    continue

                disaggregatedVar = (
                    disjunct._transformation_block()
                    .parent_block()
                    ._disaggregatedVarMap['disaggregatedVar'][disjunct][var]
                )
                disaggregatedExpr += disaggregatedVar

            # We equate the sum of the disaggregated vars to var (the original)
            # if parent_disjunct is None, else it needs to be the disaggregated
            # var corresponding to var on the parent disjunct. This is the
            # reason we transform from root to leaf: This constraint is now
            # correct regardless of how nested something may have been.
            parent_var = (
                var
                if parent_disjunct is None
                else self.get_disaggregated_var(var, parent_disjunct)
            )
            cons_idx = len(disaggregationConstraint)
            disaggregationConstraint.add(cons_idx, parent_var == disaggregatedExpr)
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

        # deactivate for the writers
        obj.deactivate()

    def _transform_disjunct(self, obj, transBlock, varSet, localVars, local_var_set):
        # We're not using the preprocessed list here, so this could be
        # inactive. We've already done the error checking in preprocessing, so
        # we just skip it here.
        if not obj.active:
            return

        relaxationBlock = self._get_disjunct_transformation_block(obj, transBlock)

        # Put the disaggregated variables all on their own block so that we can
        # isolate the name collisions and still have complete control over the
        # names on this block.
        relaxationBlock.disaggregatedVars = Block()

        # add the disaggregated variables and their bigm constraints
        # to the relaxationBlock
        for var in varSet:
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
            # mark this as local because we won't re-disaggregate if this is a
            # nested disjunction
            if local_var_set is not None:
                local_var_set.append(disaggregatedVar)

            # add the bigm constraint
            bigmConstraint = Constraint(transBlock.lbub)
            relaxationBlock.add_component(
                disaggregatedVarName + "_bounds", bigmConstraint
            )

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

        for var in localVars:
            # we don't need to disaggregated, we can use this Var, but we do
            # need to set up its bounds constraints.

            # naming conflicts are possible here since this is a bunch
            # of variables from different blocks coming together, so we
            # get a unique name
            conName = unique_component_name(
                relaxationBlock, var.getname(fully_qualified=False) + "_bounds"
            )
            bigmConstraint = Constraint(transBlock.lbub)
            relaxationBlock.add_component(conName, bigmConstraint)

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
        zero_substitute_map.update((id(v), ZeroConstant) for v in localVars)

        # Transform each component within this disjunct
        self._transform_block_components(
            obj, obj, var_substitute_map, zero_substitute_map
        )

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

    def _get_local_var_set(self, disjunction):
        # add Suffix to the relaxation block that disaggregated variables are
        # local (in case this is nested in another Disjunct)
        local_var_set = None
        parent_disjunct = disjunction.parent_block()
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

        return local_var_set

    def _warn_for_active_disjunct(
        self, innerdisjunct, outerdisjunct, var_substitute_map, zero_substitute_map
    ):
        # We override the base class method because in hull, it might just be
        # that we haven't gotten here yet.
        disjuncts = (
            innerdisjunct.values() if innerdisjunct.is_indexed() else (innerdisjunct,)
        )
        for disj in disjuncts:
            if disj in self._targets_set:
                # We're getting to this, have some patience.
                continue
            else:
                # But if it wasn't in the targets after preprocessing, it
                # doesn't belong in an active Disjunction that we are
                # transforming and we should be confused.
                _warn_for_active_disjunct(innerdisjunct, outerdisjunct)

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

    def _add_local_var_suffix(self, disjunct):
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

    def get_disaggregated_var(self, v, disjunct):
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
            logger.error(
                "It does not appear '%s' is a "
                "variable that appears in disjunct '%s'" % (v.name, disjunct.name)
            )
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
            raise GDP_Error(
                "Disjunction '%s' has not been properly "
                "transformed:"
                " None of its disjuncts are transformed." % disjunction.name
            )

        try:
            return (
                transBlock()
                .parent_block()
                ._disaggregationConstraintMap[original_var][disjunction]
            )
        except:
            logger.error(
                "It doesn't appear that '%s' is a variable that was "
                "disaggregated by Disjunction '%s'"
                % (original_var.name, disjunction.name)
            )
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
            return transBlock._bigMConstraintMap[v]
        except:
            logger.error(msg)
            raise


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
