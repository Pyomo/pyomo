#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Big-M Generalized Disjunctive Programming transformation module."""

import logging

from pyomo.common.autoslots import AutoSlots
from pyomo.common.collections import ComponentMap
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.common.gc_manager import PauseGC
from pyomo.common.modeling import unique_component_name
from pyomo.common.deprecation import deprecated, deprecation_warning
from pyomo.contrib.cp.transform.logical_to_disjunctive_program import (
    LogicalToDisjunctive,
)
from pyomo.core import (
    Block,
    BooleanVar,
    Connector,
    Constraint,
    Param,
    Set,
    SetOf,
    Var,
    Expression,
    SortComponents,
    TraversalStrategy,
    value,
    RangeSet,
    NonNegativeIntegers,
    Binary,
    Any,
)
from pyomo.core.base import TransformationFactory, Reference
import pyomo.core.expr as EXPR
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.plugins.bigm_mixin import (
    _BigM_MixIn,
    _get_bigM_suffix_list,
    _warn_for_unused_bigM_args,
)
from pyomo.gdp.plugins.gdp_to_mip_transformation import GDP_to_MIP_Transformation
from pyomo.gdp.transformed_disjunct import _TransformedDisjunct
from pyomo.gdp.util import is_child_of, _get_constraint_transBlock, _to_dict
from pyomo.core.util import target_list
from pyomo.network import Port
from pyomo.repn import generate_standard_repn
from weakref import ref as weakref_ref, ReferenceType

logger = logging.getLogger('pyomo.gdp.bigm')


class _BigMData(AutoSlots.Mixin):
    __slots__ = ('bigm_src',)

    def __init__(self):
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
        self.bigm_src = {}


Block.register_private_data_initializer(_BigMData)


@TransformationFactory.register(
    'gdp.bigm', doc="Relax disjunctive model using big-M terms."
)
class BigM_Transformation(GDP_to_MIP_Transformation, _BigM_MixIn):
    """Relax disjunctive model using big-M terms.

    Relaxes a disjunctive model into an algebraic model by adding Big-M
    terms to all disjunctive constraints.

    This transformation accepts the following keyword arguments:
        bigM: A user-specified value (or dict) of M values to use (see below)
        targets: the targets to transform [default: the instance]

    M values are determined as follows:
       1. if the constraint appears in the bigM argument dict
       2. if the constraint parent_component appears in the bigM argument dict
       3. if any block which is an ancestor to the constraint appears in
          the bigM argument dict
       4. if 'None' is in the bigM argument dict
       5. if the constraint or the constraint parent_component appear in
          a BigM Suffix attached to any parent_block() beginning with the
          constraint's parent_block and moving up to the root model.
       6. if None appears in a BigM Suffix attached to any
          parent_block() between the constraint and the root model.
       7. if the constraint is linear, estimate M using the variable bounds

    M values may be a single value or a 2-tuple specifying the M for the
    lower bound and the upper bound of the constraint body.

    Specifying "bigM=N" is automatically mapped to "bigM={None: N}".

    The transformation will create a new Block with a unique
    name beginning "_pyomo_gdp_bigm_reformulation".  That Block will
    contain an indexed Block named "relaxedDisjuncts", which will hold
    the relaxed disjuncts.  This block is indexed by an integer
    indicating the order in which the disjuncts were relaxed. All
    transformed Disjuncts will have a pointer to the block their transformed
    constraints are on, and all transformed Disjunctions will have a
    pointer to the corresponding 'Or' or 'ExactlyOne' constraint.

    """

    CONFIG = ConfigDict("gdp.bigm")
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
        'bigM',
        ConfigValue(
            default=None,
            domain=_to_dict,
            description="Big-M value used for constraint relaxation",
            doc="""

        A user-specified value, dict, or ComponentMap of M values that override
        M-values found through model Suffixes or that would otherwise be
        calculated using variable domains.""",
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
        This is only relevant when the transformation will be estimating values
        for M. If True, the transformation will calculate M values assuming that
        fixed variables will always be fixed to their current values. This means
        that if a fixed variable is unfixed after transformation, the
        transformed model is potentially no longer valid. By default, the
        transformation will assume fixed variables could be unfixed in the
        future and will use their bounds to calculate the M value rather than
        their value. Note that this could make for a weaker LP relaxation
        while the variables remain fixed.
        """,
        ),
    )
    transformation_name = 'bigm'

    def __init__(self):
        super().__init__(logger)
        self._set_up_expr_bound_visitor()

    def _apply_to(self, instance, **kwds):
        self.used_args = ComponentMap()  # If everything was sure to go well,
        # this could be a dictionary. But if
        # someone messes up and gives us a Var
        # as a key in bigMargs, I need the error
        # not to be when I try to put it into
        # this map!
        with PauseGC():
            try:
                self._apply_to_impl(instance, **kwds)
            finally:
                self._restore_state()
                self.used_args.clear()
                self._expr_bound_visitor.leaf_bounds.clear()
                self._expr_bound_visitor.use_fixed_var_values_as_bounds = False

    def _apply_to_impl(self, instance, **kwds):
        self._process_arguments(instance, **kwds)
        if self._config.assume_fixed_vars_permanent:
            self._expr_bound_visitor.use_fixed_var_values_as_bounds = True

        # filter out inactive targets and handle case where targets aren't
        # specified.
        targets = self._filter_targets(instance)
        # transform logical constraints based on targets
        self._transform_logical_constraints(instance, targets)
        # we need to preprocess targets to make sure that if there are any
        # disjunctions in targets that their disjuncts appear before them in
        # the list.
        gdp_tree = self._get_gdp_tree_from_targets(instance, targets)
        preprocessed_targets = gdp_tree.reverse_topological_sort()

        bigM = self._config.bigM
        for t in preprocessed_targets:
            if t.ctype is Disjunction:
                self._transform_disjunctionData(t, t.index(), bigM, gdp_tree)

        # issue warnings about anything that was in the bigM args dict that we
        # didn't use
        _warn_for_unused_bigM_args(bigM, self.used_args, logger)

    def _transform_disjunctionData(self, obj, index, bigM, gdp_tree):
        parent_disjunct = gdp_tree.parent(obj)
        root_disjunct = gdp_tree.root_disjunct(obj)
        (transBlock, xorConstraint) = self._setup_transform_disjunctionData(
            obj, root_disjunct
        )

        # add or (or xor) constraint
        or_expr = 0
        for disjunct in obj.disjuncts:
            or_expr += disjunct.binary_indicator_var
            self._transform_disjunct(disjunct, bigM, transBlock, gdp_tree)

        if obj.xor:
            xorConstraint[index] = or_expr == 1
        else:
            xorConstraint[index] = or_expr >= 1
        # Mark the DisjunctionData as transformed by mapping it to its XOR
        # constraint.
        obj._algebraic_constraint = weakref_ref(xorConstraint[index])

        # and deactivate for the writers
        obj.deactivate()

    def _transform_disjunct(self, obj, bigM, transBlock, gdp_tree):
        # We're not using the preprocessed list here, so this could be
        # inactive. We've already done the error checking in preprocessing, so
        # we just skip it here.
        if not obj.active:
            return

        suffix_list = _get_bigM_suffix_list(obj)
        arg_list = self._get_bigM_arg_list(bigM, obj)

        relaxationBlock = self._get_disjunct_transformation_block(obj, transBlock)

        indicator_expression = 0
        node = obj
        while node is not None:
            indicator_expression += 1 - node.binary_indicator_var
            node = gdp_tree.parent_disjunct(node)

        # This is crazy, but if the disjunction has been previously
        # relaxed, the disjunct *could* be deactivated.  This is a big
        # deal for Hull, as it uses the component_objects /
        # component_data_objects generators.  For BigM, that is OK,
        # because we never use those generators with active=True.  I am
        # only noting it here for the future when someone (me?) is
        # comparing the two relaxations.
        #
        # Transform each component within this disjunct
        self._transform_block_components(
            obj, obj, bigM, arg_list, suffix_list, indicator_expression
        )

        # deactivate disjunct to keep the writers happy
        obj._deactivate_without_fixing_indicator()

    def _transform_constraint(
        self,
        obj,
        disjunct,
        bigMargs,
        arg_list,
        disjunct_suffix_list,
        indicator_expression,
    ):
        # add constraint to the transformation block, we'll transform it there.
        transBlock = disjunct._transformation_block()
        bigm_src = transBlock.private_data().bigm_src
        constraint_map = transBlock.private_data('pyomo.gdp')

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
            lower, upper = self._get_M_from_args(c, bigMargs, arg_list, lower, upper)
            M = (lower[0], upper[0])

            if self._generate_debug_messages:
                logger.debug(
                    "GDP(BigM): The value for M for constraint '%s' "
                    "from the BigM argument is %s." % (c.name, str(M))
                )

            # if we didn't get something we need from args, try suffixes:
            if (M[0] is None and c.lower is not None) or (
                M[1] is None and c.upper is not None
            ):
                # first get anything parent to c but below disjunct
                suffix_list = _get_bigM_suffix_list(
                    c.parent_block(), stopping_block=disjunct
                )
                # prepend that to what we already collected for the disjunct.
                suffix_list.extend(disjunct_suffix_list)
                lower, upper = self._update_M_from_suffixes(
                    c, suffix_list, lower, upper
                )
                M = (lower[0], upper[0])

            if self._generate_debug_messages:
                logger.debug(
                    "GDP(BigM): The value for M for constraint '%s' "
                    "after checking suffixes is %s." % (c.name, str(M))
                )

            if c.lower is not None and M[0] is None:
                M = (self._estimate_M(c.body, c)[0] - c.lower, M[1])
                lower = (M[0], None, None)
            if c.upper is not None and M[1] is None:
                M = (M[0], self._estimate_M(c.body, c)[1] - c.upper)
                upper = (M[1], None, None)

            if self._generate_debug_messages:
                logger.debug(
                    "GDP(BigM): The value for M for constraint '%s' "
                    "after estimating (if needed) is %s." % (c.name, str(M))
                )

            # save the source information
            bigm_src[c] = (lower, upper)

            self._add_constraint_expressions(
                c,
                i,
                M,
                disjunct.binary_indicator_var,
                newConstraint,
                constraint_map,
                indicator_expression=indicator_expression,
            )

            # deactivate because we relaxed
            c.deactivate()

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
                (lower, upper, need_lower, need_upper) = self._process_M_value(
                    M,
                    lower,
                    upper,
                    need_lower,
                    need_upper,
                    bigm,
                    constraint,
                    constraint,
                )
                if not need_lower and not need_upper:
                    return lower, upper

            # if c is indexed, check for the parent component
            if constraint.parent_component() in bigm:
                parent = constraint.parent_component()
                M = bigm[parent]
                (lower, upper, need_lower, need_upper) = self._process_M_value(
                    M, lower, upper, need_lower, need_upper, bigm, parent, constraint
                )
                if not need_lower and not need_upper:
                    return lower, upper

        # if we didn't get an M that way, traverse upwards through the blocks
        # and see if None has a value on any of them.
        if M is None:
            for bigm in suffix_list:
                if None in bigm:
                    M = bigm[None]
                    (lower, upper, need_lower, need_upper) = self._process_M_value(
                        M, lower, upper, need_lower, need_upper, bigm, None, constraint
                    )
                if not need_lower and not need_upper:
                    return lower, upper
        return lower, upper

    @deprecated(
        "The get_m_value_src function is deprecated. Use "
        "the get_M_value_src function if you need source "
        "information or the get_M_value function if you "
        "only need values.",
        version='5.7.1',
    )
    def get_m_value_src(self, constraint):
        transBlock = _get_constraint_transBlock(constraint)
        ((lower_val, lower_source, lower_key), (upper_val, upper_source, upper_key)) = (
            transBlock.private_data().bigm_src[constraint]
        )

        if (
            constraint.lower is not None
            and constraint.upper is not None
            and (not lower_source is upper_source or not lower_key is upper_key)
        ):
            raise GDP_Error(
                "This is why this method is deprecated: The lower "
                "and upper M values for constraint %s came from "
                "different sources, please use the get_M_value_src "
                "method." % constraint.name
            )
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
        return transBlock.private_data().bigm_src[constraint]

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
        lower, upper = transBlock.private_data().bigm_src[constraint]
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
            Disjunct, active=None, descend_into=(Block, Disjunct)
        ):
            transBlock = disj.transformation_block
            # First check if it was transformed at all.
            if transBlock is not None:
                # If it was transformed with BigM, we get the M values.
                for cons in transBlock.private_data().bigm_src:
                    m_values[cons] = self.get_M_value(cons)
        return m_values

    def get_largest_M_value(self, model):
        """Returns the largest M value for any constraint on the model.

        Parameters
        ----------
        model: A GDP model that has been transformed with BigM
        """
        return max(
            max(abs(m) for m in m_values if m is not None)
            for m_values in self.get_all_M_values_by_constraint(model).values()
        )
