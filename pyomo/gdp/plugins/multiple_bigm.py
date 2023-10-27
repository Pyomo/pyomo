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

import itertools
import logging

from pyomo.common.collections import ComponentMap
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.common.gc_manager import PauseGC
from pyomo.common.modeling import unique_component_name

from pyomo.core import (
    Any,
    Binary,
    Block,
    BooleanVar,
    Connector,
    Constraint,
    Expression,
    ExternalFunction,
    maximize,
    minimize,
    NonNegativeIntegers,
    Objective,
    Param,
    RangeSet,
    Set,
    SetOf,
    SortComponents,
    Suffix,
    value,
    Var,
)
from pyomo.core.base import Reference, TransformationFactory
import pyomo.core.expr as EXPR
from pyomo.core.util import target_list

from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.plugins.bigm_mixin import (
    _BigM_MixIn,
    _convert_M_to_tuple,
    _warn_for_unused_bigM_args,
)
from pyomo.gdp.plugins.gdp_to_mip_transformation import GDP_to_MIP_Transformation
from pyomo.gdp.transformed_disjunct import _TransformedDisjunct
from pyomo.gdp.util import get_gdp_tree, _to_dict
from pyomo.network import Port
from pyomo.opt import SolverFactory, TerminationCondition
from pyomo.repn import generate_standard_repn

from weakref import ref as weakref_ref

logger = logging.getLogger('pyomo.gdp.mbigm')


@TransformationFactory.register(
    'gdp.mbigm',
    doc="Relax disjunctive model using big-M terms specific to each disjunct",
)
class MultipleBigMTransformation(GDP_to_MIP_Transformation, _BigM_MixIn):
    """
    Implements the multiple big-M transformation from [1]. Note that this
    transformation is no different than the big-M transformation for two-
    term disjunctions, but that it may provide a tighter relaxation for
    models containing some disjunctions with three or more terms.

    [1] Francisco Trespalaios and Ignacio E. Grossmann, "Improved Big-M
        reformulation for generalized disjunctive programs," Computers and
        Chemical Engineering, vol. 76, 2015, pp. 98-103.
    """

    CONFIG = ConfigDict('gdp.mbigm')
    CONFIG.declare(
        'targets',
        ConfigValue(
            default=None,
            domain=target_list,
            description="target or list of targets that will be relaxed",
            doc="""
        This specifies the list of components to relax. If None (default), the
        entire model is transformed. Note that if the transformation is done out
        of place, the list of targets should be attached to the model before it
        is cloned, and the list will specify the targets on the cloned
        instance.""",
        ),
    )
    CONFIG.declare(
        'assume_fixed_vars_permanent',
        ConfigValue(
            default=False,
            domain=bool,
            description="Boolean indicating whether or not to transform so that "
            "the transformed model will still be valid when fixed Vars are "
            "unfixed.",
            doc="""
        This is only relevant when the transformation will be calculating M
        values. If True, the transformation will calculate M values assuming
        that fixed variables will always be fixed to their current values. This
        means that if a fixed variable is unfixed after transformation, the
        transformed model is potentially no longer valid. By default, the
        transformation will assume fixed variables could be unfixed in the
        future and will use their bounds to calculate the M value rather than
        their value. Note that this could make for a weaker LP relaxation
        while the variables remain fixed.
        """,
        ),
    )
    CONFIG.declare(
        'solver',
        ConfigValue(
            default=SolverFactory('gurobi'),
            description="A solver to use to solve the continuous subproblems for "
            "calculating the M values",
        ),
    )
    CONFIG.declare(
        'bigM',
        ConfigValue(
            default=None,
            domain=_to_dict,
            description="Big-M values to use while relaxing constraints",
            doc="""
        A user-specified dict or ComponentMap mapping tuples of Constraints
        and Disjuncts to Big-M values valid for relaxing the constraint if
        the Disjunct is chosen.

        Note: Unlike in the bigm transformation, we require the keys in this
        mapping specify the components the M value applies to exactly in order
        to avoid ambiguity. However, if the 'only_mbigm_bound_constraints'
        option is True, this argument can be used as it would be in the
        traditional bigm transformation for the non-bound constraints.
        """,
        ),
    )
    CONFIG.declare(
        'reduce_bound_constraints',
        ConfigValue(
            default=True,
            domain=bool,
            description="Flag indicating whether or not to handle disjunctive "
            "constraints that bound a single variable in a single (tighter) "
            "constraint, rather than one per Disjunct.",
            doc="""
        Given the not-uncommon special structure:

        [l_1 <= x <= u_1] v [l_2 <= x <= u_2] v ... v [l_K <= x <= u_K],

        instead of applying the rote transformation that would create 2*K
        different constraints in the relaxation, we can write two constraints:

        x >= l_1*y_1 + l_2*y_2 + ... + l_K*y_k
        x <= u_1*y_1 + u_2*y_2 + ... + u_K*y_K.

        This relaxation is as tight and has fewer constraints. This option is
        a flag to tell the mbigm transformation to detect this structure and
        handle it specially. Note that this is a special case of the 'Hybrid
        Big-M Formulation' from [2] that takes advantage of the common left-
        hand side matrix for disjunctive constraints that bound a single
        variable.

        Note that we do not use user-specified M values for these constraints
        when this flag is set to True: If tighter bounds exist then they
        they should be put in the constraints.

        [2] Juan Pablo Vielma, "Mixed Integer Linear Programming Formluation
            Techniques," SIAM Review, vol. 57, no. 1, 2015, pp. 3-57.
        """,
        ),
    )
    CONFIG.declare(
        'only_mbigm_bound_constraints',
        ConfigValue(
            default=False,
            domain=bool,
            description="Flag indicating if only bound constraints should be "
            "transformed with multiple-bigm, or if all the disjunctive "
            "constraints should.",
            doc="""
        Sometimes it is only computationally advantageous to apply multiple-
        bigm to disjunctive constraints with the special structure:

        [l_1 <= x <= u_1] v [l_2 <= x <= u_2] v ... v [l_K <= x <= u_K],

        and transform other disjunctive constraints with the traditional
        big-M transformation. This flag is used to set the above behavior.

        Note that the reduce_bound_constraints flag must also be True when
        this flag is set to True.
        """,
        ),
    )
    transformation_name = 'mbigm'

    def __init__(self):
        super().__init__(logger)
        self.handlers[Suffix] = self._warn_for_active_suffix
        self._arg_list = {}
        self._set_up_expr_bound_visitor()

    def _apply_to(self, instance, **kwds):
        self.used_args = ComponentMap()
        with PauseGC():
            try:
                self._apply_to_impl(instance, **kwds)
            finally:
                self._restore_state()
                self.used_args.clear()
                self._arg_list.clear()
                self._expr_bound_visitor.leaf_bounds.clear()
                self._expr_bound_visitor.use_fixed_var_values_as_bounds = False

    def _apply_to_impl(self, instance, **kwds):
        self._process_arguments(instance, **kwds)
        if self._config.assume_fixed_vars_permanent:
            self._bound_visitor.use_fixed_var_values_as_bounds = True

        if (
            self._config.only_mbigm_bound_constraints
            and not self._config.reduce_bound_constraints
        ):
            raise GDP_Error(
                "The 'only_mbigm_bound_constraints' option is set "
                "to True, but the 'reduce_bound_constraints' "
                "option is not. This is not supported--please also "
                "set 'reduce_bound_constraints' to True if you "
                "only wish to transform the bound constraints with "
                "multiple bigm."
            )

        # filter out inactive targets and handle case where targets aren't
        # specified.
        targets = self._filter_targets(instance)
        # transform any logical constraints that might be anywhere on the stuff
        # we're about to transform. We do this before we preprocess targets
        # because we will likely create more disjunctive components that will
        # need transformation.
        self._transform_logical_constraints(instance, targets)
        # We don't allow nested, so it doesn't much matter which way we sort
        # this. But transforming from leaf to root makes the error checking for
        # complaining about nested smoother, so we do that. We have to transform
        # a Disjunction at a time because, more similarly to hull than bigm, we
        # need information from the other Disjuncts in the Disjunction.
        gdp_tree = self._get_gdp_tree_from_targets(instance, targets)
        preprocessed_targets = gdp_tree.reverse_topological_sort()

        for t in preprocessed_targets:
            if t.ctype is Disjunction:
                self._transform_disjunctionData(
                    t,
                    t.index(),
                    parent_disjunct=gdp_tree.parent(t),
                    root_disjunct=gdp_tree.root_disjunct(t),
                )

        # issue warnings about anything that was in the bigM args dict that we
        # didn't use
        _warn_for_unused_bigM_args(self._config.bigM, self.used_args, logger)

    def _transform_disjunctionData(self, obj, index, parent_disjunct, root_disjunct):
        if root_disjunct is not None:
            # We do not support nested because, unlike in regular bigM, the
            # constraints are not fully relaxed when the exactly-one constraint
            # is not enforced. (For example, in this model: [1 <= x <= 3, [1 <=
            # y <= 5] v [6 <= y <= 10]] v [5 <= x <= 10, 15 <= y <= 20]), we
            # would need to put the relaxed inner-disjunction constraints on the
            # parent Disjunct and process them again. This means the order in
            # which we transformed Disjuncts would change the calculated M
            # values. This is crazy, so we skip it.
            raise GDP_Error(
                "Found nested Disjunction '%s'. The multiple bigm "
                "transformation does not support nested GDPs. "
                "Please flatten the model before calling the "
                "transformation" % obj.name
            )

        if not obj.xor:
            # This transformation assumes it can relax constraints assuming that
            # another Disjunct is chosen. If it could be possible to choose both
            # then that logic might fail.
            raise GDP_Error(
                "Cannot do multiple big-M reformulation for "
                "Disjunction '%s' with OR constraint.  "
                "Must be an XOR!" % obj.name
            )

        (transBlock, algebraic_constraint) = self._setup_transform_disjunctionData(
            obj, root_disjunct
        )

        ## Here's the logic for the actual transformation

        arg_Ms = self._config.bigM if self._config.bigM is not None else {}

        # First handle the bound constraints if we are dealing with them
        # separately
        active_disjuncts = [disj for disj in obj.disjuncts if disj.active]
        transformed_constraints = set()
        if self._config.reduce_bound_constraints:
            transformed_constraints = self._transform_bound_constraints(
                active_disjuncts, transBlock, arg_Ms
            )

        Ms = arg_Ms
        if not self._config.only_mbigm_bound_constraints:
            Ms = (
                transBlock.calculated_missing_m_values
            ) = self._calculate_missing_M_values(
                active_disjuncts, arg_Ms, transBlock, transformed_constraints
            )

        # Now we can deactivate the constraints we deferred, so that we don't
        # re-transform them
        for cons in transformed_constraints:
            cons.deactivate()

        or_expr = 0
        for disjunct in active_disjuncts:
            or_expr += disjunct.indicator_var.get_associated_binary()
            self._transform_disjunct(disjunct, transBlock, active_disjuncts, Ms)
        rhs = 1 if parent_disjunct is None else parent_disjunct.binary_indicator_var
        algebraic_constraint.add(index, (or_expr, rhs))
        # map the DisjunctionData to its XOR constraint to mark it as
        # transformed
        obj._algebraic_constraint = weakref_ref(algebraic_constraint[index])

        obj.deactivate()

    def _transform_disjunct(self, obj, transBlock, active_disjuncts, Ms):
        # We've already filtered out deactivated disjuncts, so we know obj is
        # active.

        # Make a relaxation block if we haven't already.
        relaxationBlock = self._get_disjunct_transformation_block(obj, transBlock)

        # Transform everything on the disjunct
        self._transform_block_components(obj, obj, active_disjuncts, Ms)

        # deactivate disjunct so writers can be happy
        obj._deactivate_without_fixing_indicator()

    def _warn_for_active_suffix(self, obj, disjunct, active_disjuncts, Ms):
        raise GDP_Error(
            "Found active Suffix '{0}' on Disjunct '{1}'. "
            "The multiple bigM transformation does not currently "
            "support Suffixes.".format(obj.name, disjunct.name)
        )

    def _transform_constraint(self, obj, disjunct, active_disjuncts, Ms):
        # we will put a new transformed constraint on the relaxation block.
        relaxationBlock = disjunct._transformation_block()
        constraintMap = relaxationBlock._constraintMap
        transBlock = relaxationBlock.parent_block()

        # Though rare, it is possible to get naming conflicts here
        # since constraints from all blocks are getting moved onto the
        # same block. So we get a unique name
        name = unique_component_name(
            relaxationBlock, obj.getname(fully_qualified=False)
        )

        newConstraint = Constraint(Any)
        relaxationBlock.add_component(name, newConstraint)

        for i in sorted(obj.keys()):
            c = obj[i]
            if not c.active:
                continue

            if not self._config.only_mbigm_bound_constraints:
                transformed = []
                if c.lower is not None:
                    rhs = sum(
                        Ms[c, disj][0] * disj.indicator_var.get_associated_binary()
                        for disj in active_disjuncts
                        if disj is not disjunct
                    )
                    newConstraint.add((i, 'lb'), c.body - c.lower >= rhs)
                    transformed.append(newConstraint[i, 'lb'])

                if c.upper is not None:
                    rhs = sum(
                        Ms[c, disj][1] * disj.indicator_var.get_associated_binary()
                        for disj in active_disjuncts
                        if disj is not disjunct
                    )
                    newConstraint.add((i, 'ub'), c.body - c.upper <= rhs)
                    transformed.append(newConstraint[i, 'ub'])
                for c_new in transformed:
                    constraintMap['srcConstraints'][c_new] = [c]
                constraintMap['transformedConstraints'][c] = transformed
            else:
                lower = (None, None, None)
                upper = (None, None, None)

                if disjunct not in self._arg_list:
                    self._arg_list[disjunct] = self._get_bigM_arg_list(
                        self._config.bigM, disjunct
                    )
                arg_list = self._arg_list[disjunct]

                # first, we see if an M value was specified in the arguments.
                # (This returns None if not)
                lower, upper = self._get_M_from_args(c, Ms, arg_list, lower, upper)
                M = (lower[0], upper[0])

                # estimate if we don't have what we need
                if c.lower is not None and M[0] is None:
                    M = (self._estimate_M(c.body, c)[0] - c.lower, M[1])
                    lower = (M[0], None, None)
                if c.upper is not None and M[1] is None:
                    M = (M[0], self._estimate_M(c.body, c)[1] - c.upper)
                    upper = (M[1], None, None)
                self._add_constraint_expressions(
                    c,
                    i,
                    M,
                    disjunct.indicator_var.get_associated_binary(),
                    newConstraint,
                    constraintMap,
                )

        # deactivate now that we have transformed
        c.deactivate()

    def _transform_bound_constraints(self, active_disjuncts, transBlock, Ms):
        # first we're just going to find all of them
        bounds_cons = ComponentMap()
        lower_bound_constraints_by_var = ComponentMap()
        upper_bound_constraints_by_var = ComponentMap()
        transformed_constraints = set()
        for disj in active_disjuncts:
            for c in disj.component_data_objects(
                Constraint,
                active=True,
                descend_into=Block,
                sort=SortComponents.deterministic,
            ):
                repn = generate_standard_repn(c.body)
                if repn.is_linear() and len(repn.linear_vars) == 1:
                    # We can treat this as a bounds constraint
                    v = repn.linear_vars[0]
                    if v not in bounds_cons:
                        bounds_cons[v] = [{}, {}]
                    M = [None, None]
                    if c.lower is not None:
                        M[0] = (c.lower - repn.constant) / repn.linear_coefs[0]
                        if disj in bounds_cons[v][0]:
                            # this is a redundant bound, we need to keep the
                            # better one
                            M[0] = max(M[0], bounds_cons[v][0][disj])
                        bounds_cons[v][0][disj] = M[0]
                        if v in lower_bound_constraints_by_var:
                            lower_bound_constraints_by_var[v].add((c, disj))
                        else:
                            lower_bound_constraints_by_var[v] = {(c, disj)}
                    if c.upper is not None:
                        M[1] = (c.upper - repn.constant) / repn.linear_coefs[0]
                        if disj in bounds_cons[v][1]:
                            # this is a redundant bound, we need to keep the
                            # better one
                            M[1] = min(M[1], bounds_cons[v][1][disj])
                        bounds_cons[v][1][disj] = M[1]
                        if v in upper_bound_constraints_by_var:
                            upper_bound_constraints_by_var[v].add((c, disj))
                        else:
                            upper_bound_constraints_by_var[v] = {(c, disj)}
                    # Add the M values to the dictionary
                    transBlock._mbm_values[c, disj] = M

                    # We can't deactivate yet because we will still be solving
                    # this Disjunct when we calculate M values for non-bounds
                    # constraints. We track that it is transformed instead by
                    # adding it to this set.
                    transformed_constraints.add(c)

        # Now we actually construct the constraints. We do this separately so
        # that we can make sure that we have a term for every active disjunct in
        # the disjunction (falling back on the variable bounds if they are there
        transformed = transBlock.transformed_bound_constraints
        offset = len(transformed)
        for i, (v, (lower_dict, upper_dict)) in enumerate(bounds_cons.items()):
            lower_rhs = 0
            upper_rhs = 0
            for disj in active_disjuncts:
                relaxationBlock = self._get_disjunct_transformation_block(
                    disj, transBlock
                )
                if len(lower_dict) > 0:
                    M = lower_dict.get(disj, None)
                    if M is None:
                        # substitute the lower bound if it has one
                        M = v.lb
                    if M is None:
                        raise GDP_Error(
                            "There is no lower bound for variable '%s', and "
                            "Disjunct '%s' does not specify one in its "
                            "constraints. The transformation cannot construct "
                            "the special bound constraint relaxation without "
                            "one of these." % (v.name, disj.name)
                        )
                    lower_rhs += M * disj.indicator_var.get_associated_binary()
                if len(upper_dict) > 0:
                    M = upper_dict.get(disj, None)
                    if M is None:
                        # substitute the upper bound if it has one
                        M = v.ub
                    if M is None:
                        raise GDP_Error(
                            "There is no upper bound for variable '%s', and "
                            "Disjunct '%s' does not specify one in its "
                            "constraints. The transformation cannot construct "
                            "the special bound constraint relaxation without "
                            "one of these." % (v.name, disj.name)
                        )
                    upper_rhs += M * disj.indicator_var.get_associated_binary()
            idx = i + offset
            if len(lower_dict) > 0:
                transformed.add((idx, 'lb'), v >= lower_rhs)
                relaxationBlock._constraintMap['srcConstraints'][
                    transformed[idx, 'lb']
                ] = []
                for c, disj in lower_bound_constraints_by_var[v]:
                    relaxationBlock._constraintMap['srcConstraints'][
                        transformed[idx, 'lb']
                    ].append(c)
                    disj.transformation_block._constraintMap['transformedConstraints'][
                        c
                    ] = [transformed[idx, 'lb']]
            if len(upper_dict) > 0:
                transformed.add((idx, 'ub'), v <= upper_rhs)
                relaxationBlock._constraintMap['srcConstraints'][
                    transformed[idx, 'ub']
                ] = []
                for c, disj in upper_bound_constraints_by_var[v]:
                    relaxationBlock._constraintMap['srcConstraints'][
                        transformed[idx, 'ub']
                    ].append(c)
                    # might already be here if it had an upper bound
                    if (
                        c
                        in disj.transformation_block._constraintMap[
                            'transformedConstraints'
                        ]
                    ):
                        disj.transformation_block._constraintMap[
                            'transformedConstraints'
                        ][c].append(transformed[idx, 'ub'])
                    else:
                        disj.transformation_block._constraintMap[
                            'transformedConstraints'
                        ][c] = [transformed[idx, 'ub']]

        return transformed_constraints

    def _add_transformation_block(self, to_block):
        transBlock, new_block = super()._add_transformation_block(to_block)

        if new_block:
            # Will store M values as we transform
            transBlock._mbm_values = {}
            transBlock.transformed_bound_constraints = Constraint(
                NonNegativeIntegers, ['lb', 'ub']
            )
        return transBlock, new_block

    def _get_all_var_objects(self, active_disjuncts):
        # This is actually a general utility for getting all Vars that appear in
        # active Disjuncts in a Disjunction.
        seen = set()
        for disj in active_disjuncts:
            for constraint in disj.component_data_objects(
                Constraint,
                active=True,
                sort=SortComponents.deterministic,
                descend_into=Block,
            ):
                for var in EXPR.identify_variables(constraint.expr, include_fixed=True):
                    if id(var) not in seen:
                        seen.add(id(var))
                        yield var

    def _calculate_missing_M_values(
        self, active_disjuncts, arg_Ms, transBlock, transformed_constraints
    ):
        scratch_blocks = {}
        all_vars = list(self._get_all_var_objects(active_disjuncts))
        for disjunct, other_disjunct in itertools.product(
            active_disjuncts, active_disjuncts
        ):
            if disjunct is other_disjunct:
                continue
            if id(other_disjunct) in scratch_blocks:
                scratch = scratch_blocks[id(other_disjunct)]
            else:
                scratch = scratch_blocks[id(other_disjunct)] = Block()
                other_disjunct.add_component(
                    unique_component_name(other_disjunct, "scratch"), scratch
                )
                scratch.obj = Objective(expr=0)  # placeholder, but I want to
                # take the name before I add a
                # bunch of random reference
                # objects.

                # If the writers don't assume Vars are declared on the Block
                # being solved, we won't need this!
                for v in all_vars:
                    ref = Reference(v)
                    scratch.add_component(unique_component_name(scratch, v.name), ref)

            for constraint in disjunct.component_data_objects(
                Constraint,
                active=True,
                descend_into=Block,
                sort=SortComponents.deterministic,
            ):
                if constraint in transformed_constraints:
                    continue
                # First check args
                if (constraint, other_disjunct) in arg_Ms:
                    (lower_M, upper_M) = _convert_M_to_tuple(
                        arg_Ms[constraint, other_disjunct], constraint, other_disjunct
                    )
                    self.used_args[constraint, other_disjunct] = (lower_M, upper_M)
                else:
                    (lower_M, upper_M) = (None, None)
                if constraint.lower is not None and lower_M is None:
                    # last resort: calculate
                    if lower_M is None:
                        scratch.obj.expr = constraint.body - constraint.lower
                        scratch.obj.sense = minimize
                        results = self._config.solver.solve(other_disjunct)
                        if (
                            results.solver.termination_condition
                            is not TerminationCondition.optimal
                        ):
                            raise GDP_Error(
                                "Unsuccessful solve to calculate M value to "
                                "relax constraint '%s' on Disjunct '%s' when "
                                "Disjunct '%s' is selected."
                                % (constraint.name, disjunct.name, other_disjunct.name)
                            )
                        lower_M = value(scratch.obj.expr)
                if constraint.upper is not None and upper_M is None:
                    # last resort: calculate
                    if upper_M is None:
                        scratch.obj.expr = constraint.body - constraint.upper
                        scratch.obj.sense = maximize
                        results = self._config.solver.solve(other_disjunct)
                        if (
                            results.solver.termination_condition
                            is not TerminationCondition.optimal
                        ):
                            raise GDP_Error(
                                "Unsuccessful solve to calculate M value to "
                                "relax constraint '%s' on Disjunct '%s' when "
                                "Disjunct '%s' is selected."
                                % (constraint.name, disjunct.name, other_disjunct.name)
                            )
                        upper_M = value(scratch.obj.expr)
                arg_Ms[constraint, other_disjunct] = (lower_M, upper_M)
                transBlock._mbm_values[constraint, other_disjunct] = (lower_M, upper_M)

        # clean up the scratch blocks
        for blk in scratch_blocks.values():
            blk.parent_block().del_component(blk)

        return arg_Ms

    # These are all functions to retrieve transformed components from
    # original ones and vice versa.

    def get_src_constraints(self, transformedConstraint):
        """Return the original Constraints whose transformed counterpart is
        transformedConstraint

        Parameters
        ----------
        transformedConstraint: Constraint, which must be a component on one of
        the BlockDatas in the relaxedDisjuncts Block of
        a transformation block
        """
        # This is silly, but we rename this function for multiple bigm because
        # transformed constraints have multiple source constraints.
        return super().get_src_constraint(transformedConstraint)

    def get_all_M_values(self, model):
        """Returns a dictionary mapping each constraint, disjunct pair (where
        the constraint is on a disjunct and the disjunct is in the same
        disjunction as that disjunct) to a tuple: (lower_M_value,
        upper_M_value), where either can be None if the constraint does not
        have a lower or upper bound (respectively).

        Parameters
        ----------
        model: A GDP model that has been transformed with multiple-BigM
        """
        all_ms = {}
        for disjunction in model.component_data_objects(
            Disjunction,
            active=None,
            descend_into=(Block, Disjunct),
            sort=SortComponents.deterministic,
        ):
            if disjunction.algebraic_constraint is not None:
                transBlock = disjunction.algebraic_constraint.parent_block()
                # Don't necessarily assume all disjunctions were transformed
                # with multiple bigm...
                if hasattr(transBlock, "_mbm_values"):
                    all_ms.update(transBlock._mbm_values)

        return all_ms
