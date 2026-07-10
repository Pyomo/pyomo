# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import enum
import logging

from pyomo.common.autoslots import AutoSlots
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.config import ConfigDict, ConfigValue, InEnum
from pyomo.common.modeling import unique_component_name
from pyomo.core import Any, Block, Constraint, SortComponents
from pyomo.core.base import Transformation, TransformationFactory
from pyomo.core.base.component import ComponentBase
from pyomo.core.base.constraint import ConstraintData
from pyomo.core.util import target_list
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.util import _parent_disjunct, get_gdp_tree

logger = logging.getLogger(__name__)


class _SimpleDisjunctionTransformationData(AutoSlots.Mixin):
    """Private data stored on each transformation Block.

    Holds the mapping from a generated (simple) Disjunction back to the
    original Disjunction it was built from. The map holds the components
    directly (a hard reference), so callers can always recover the source
    Disjunction from a simple one via ``get_src_disjunction``.
    """

    __slots__ = ('src_disjunction',)

    def __init__(self):
        self.src_disjunction = ComponentMap()


Block.register_private_data_initializer(_SimpleDisjunctionTransformationData)


class ConstraintSelectionMethod(str, enum.Enum):
    """Strategy used to derive the single Constraint kept for each Disjunct.

    user_specified:
        Use the Constraint that the user assigned to each Disjunct through the
        ``selected_constraints`` mapping.
    first:
        Use the first active Constraint encountered on each Disjunct (in
        deterministic component order).

    Each method reduces the (active) source Constraints of a Disjunct to the
    single expression placed in the corresponding simple Disjunct. The two
    methods above each keep one Constraint as-is; future methods may instead
    aggregate several Constraints into one (see ``_combine_sources``).
    """

    user_specified = 'user_specified'
    first = 'first'


def _as_constraint_list(value):
    """Normalize a ``selected_constraints`` value into a list of Constraints.

    Accepts a single ConstraintData, an indexed Constraint container (expanded
    into its members), or any iterable of Constraints. Anything else is wrapped
    in a single-element list so that validation can reject it with a clear
    message. Returning a list (rather than a single Constraint) keeps the data
    structure ready for selection methods that aggregate several Constraints
    into one.
    """
    if isinstance(value, ConstraintData):
        return [value]
    if isinstance(value, ComponentBase) and value.ctype is Constraint:
        return list(value.values())
    if isinstance(value, (list, tuple, set, ComponentSet)):
        return list(value)
    return [value]


def _selected_constraints_map(arg):
    """ConfigValue domain coercing the user's selection into a ComponentMap.

    Accepts a dict or ComponentMap mapping Disjuncts (the keys) to the
    Constraint(s) (the values) that should be retained for each of them. Each
    value is normalized to a list of Constraints.
    """
    if arg is None:
        return ComponentMap()
    if isinstance(arg, (ComponentMap, dict)):
        items = arg.items()
    else:
        try:
            items = dict(arg).items()
        except (TypeError, ValueError):
            raise ValueError(
                "Expected a dict or ComponentMap mapping Disjuncts to "
                "Constraints for 'selected_constraints', but received an object "
                "of type %s" % (type(arg).__name__,)
            )
    result = ComponentMap()
    for disjunct, value in items:
        result[disjunct] = _as_constraint_list(value)
    return result


@TransformationFactory.register(
    'gdp.simple_disjunction',
    doc="Reformulate selected Disjunctions by building, for each one, a "
    "'simple' Disjunction whose Disjuncts each retain a single Constraint "
    "derived from the corresponding original Disjunct.",
)
class SimpleDisjunctionTransformation(Transformation):
    """Reformulate one or more Disjunctions as *simple* Disjunctions.

    A *simple* Disjunction is one in which every Disjunct holds exactly one
    Constraint. For each Disjunction that is transformed, this transformation
    derives a single Constraint for each of its Disjuncts and assembles those
    Constraints into a brand new Disjunction. Each new Disjunct keeps a single
    Constraint derived from the Disjunct it was generated from, and reuses the
    original model (problem) variables.

    The original Disjunction is never modified: the generated simple Disjunction
    (together with its Disjuncts) is placed on a transformation Block added to
    the parent Block of the Disjunction it was generated from. A single such
    transformation Block is shared by every simple Disjunction built on the same
    parent Block. The new Disjuncts get their own indicator variables, so the
    simple Disjunction is an independent component that the caller may transform
    or otherwise use however they see fit.

    There is more than one reasonable way to reduce a Disjunct to a single
    Constraint, so the strategy is selectable through the
    ``constraint_selection_method`` option (see
    :class:`ConstraintSelectionMethod`):

      * ``'first'`` (the default) keeps the first active Constraint encountered
        on each Disjunct.
      * ``'user_specified'`` keeps the Constraint that the user assigned to each
        Disjunct through the ``selected_constraints`` mapping.

    Both currently-implemented methods keep a single existing Constraint. The
    selection machinery is, however, organized around lists of source
    Constraints and a separate combination step (``_combine_sources``) so that
    future methods which aggregate several Constraints into a single one can be
    added without restructuring.

    Only active Constraints are ever considered. If a Disjunct has no active
    Constraints, it is skipped (no corresponding Disjunct is created in the
    simple Disjunction). If *every* Disjunct of a Disjunction is skipped, so
    that the simple Disjunction would be empty, a GDP_Error is raised and no
    simple Disjunction is created for that Disjunction.

    Nested Disjunctions are not supported: a Disjunction is not transformed if
    it is itself nested inside a Disjunct or if any of its Disjuncts contains a
    Disjunction of its own. When no targets are given, such Disjunctions are
    skipped automatically; when a nested Disjunction is supplied as an explicit
    target, a GDP_Error is raised. Likewise, deactivated Disjunctions are
    skipped automatically but raise a GDP_Error when supplied as an explicit
    target.

    After the reformulation, ``get_simple_disjunction`` and
    ``get_src_disjunction`` map between an original Disjunction and the simple
    Disjunction generated from it.
    """

    CONFIG = ConfigDict('gdp.simple_disjunction')
    CONFIG.declare(
        'targets',
        ConfigValue(
            default=None,
            domain=target_list,
            description="target or list of targets (Disjunctions or Blocks) to "
            "reformulate",
            doc="""
            This specifies the list of Disjunctions to reformulate, or the
            Blocks whose (active, non-nested) Disjunctions should be
            reformulated. If None (default), every active, non-nested Disjunction
            on the instance is reformulated. Note that if the transformation is
            done out of place, the
            list of targets should be attached to the model before it is cloned,
            and the list will specify the targets on the cloned instance.
            """,
        ),
    )
    CONFIG.declare(
        'constraint_selection_method',
        ConfigValue(
            default=ConstraintSelectionMethod.first,
            domain=InEnum(ConstraintSelectionMethod),
            description="Strategy used to derive the single Constraint kept for "
            "each Disjunct",
            doc="""
            How to reduce each Disjunct to a single Constraint. Options are the
            elements of the enum ConstraintSelectionMethod, or equivalently the
            strings 'first' or 'user_specified'.

            'first' keeps the first active Constraint encountered on each
            Disjunct (in deterministic component order). 'user_specified' keeps
            the Constraint that the user assigned to each Disjunct through the
            'selected_constraints' option, which is required in that case.
            """,
        ),
    )
    CONFIG.declare(
        'selected_constraints',
        ConfigValue(
            default=None,
            domain=_selected_constraints_map,
            description="Mapping from Disjuncts to the Constraint(s) to keep for "
            "each of them",
            doc="""
            A dict or ComponentMap whose keys are Disjuncts and whose values are
            the (active) Constraint, or list of Constraints, to retain for each
            of those Disjuncts. This is only used (and is required) when
            'constraint_selection_method' is 'user_specified'. A Disjunct that
            owns active Constraints but is absent from this mapping is an error;
            a Disjunct with no active Constraints may be omitted and is skipped.

            The currently-implemented selection methods keep a single Constraint
            per Disjunct, so a single Constraint is expected for each entry.
            Values are nevertheless stored as lists so that selection methods
            which aggregate several Constraints into one can reuse this option.
            """,
        ),
    )
    transformation_name = 'simple_disjunction'

    #: Dispatch from selection method to the builder that turns a Disjunct's
    #: selected source Constraints into the single expression placed in the
    #: simple Disjunct. New strategies only need a new entry and builder.
    _EXPRESSION_BUILDERS = {
        ConstraintSelectionMethod.first: '_first_constraint_expression',
        ConstraintSelectionMethod.user_specified: '_user_specified_expression',
    }

    def __init__(self):
        super().__init__()
        self.logger = logger

    def _apply_to(self, instance, **kwds):
        if instance.ctype is not Block:
            raise GDP_Error(
                "Transformation called on %s of type %s. 'instance' must be a "
                "ConcreteModel or Block." % (instance.name, instance.ctype)
            )

        config = self.CONFIG(kwds.pop('options', {}))
        config.set_value(kwds)

        method = config.constraint_selection_method
        selected_constraints = config.selected_constraints
        if method is ConstraintSelectionMethod.user_specified:
            if not selected_constraints:
                raise GDP_Error(
                    "The 'user_specified' constraint selection method was "
                    "requested, but no 'selected_constraints' mapping was "
                    "provided."
                )
        elif selected_constraints:
            logger.warning(
                "A 'selected_constraints' mapping was provided, but the "
                "constraint selection method is '%s', so the mapping will be "
                "ignored. Set constraint_selection_method='user_specified' to "
                "use it." % (method.value,)
            )

        build_expression = getattr(self, self._EXPRESSION_BUILDERS[method])

        targets = config.targets
        # Track the Disjunctions the user named explicitly (as opposed to ones
        # we merely came across while descending into a Block or the instance).
        # An explicit, ineligible target is an error; one we only discovered is
        # skipped quietly.
        explicit = ComponentSet()
        if targets is None:
            targets = (instance,)
        else:
            for t in targets:
                if t.ctype is Disjunction:
                    explicit.update(t.values() if t.is_indexed() else (t,))

        # We reuse a single transformation Block per parent Block, so all the
        # simple Disjunctions built on a given parent Block share one Block.
        transformation_blocks = {}

        # The GDP tree captures the whole nesting structure for us, so we can
        # ask it directly which Disjunctions are nested (or contain nesting)
        # rather than walking the model ourselves. (get_gdp_tree also validates
        # that every target lives on the instance.)
        gdp_tree = get_gdp_tree(targets, instance)
        for disjunction in gdp_tree.topological_sort():
            if disjunction.ctype is not Disjunction:
                continue
            if not disjunction.active:
                # Discovery only turns up active Disjunctions, so an inactive one
                # here was named explicitly.
                if disjunction in explicit:
                    raise GDP_Error(
                        "Disjunction '%s' is deactivated, so a simple "
                        "disjunction cannot be created from it. (Deactivated "
                        "Disjunctions are skipped automatically when no targets "
                        "are specified.)" % disjunction.name
                    )
                continue
            if _parent_disjunct(disjunction) is not None:
                # This Disjunction is itself nested inside a Disjunct.
                if disjunction in explicit:
                    raise GDP_Error(
                        "Disjunction '%s' is nested in another Disjunct. This "
                        "transformation does not create simple disjunctions from "
                        "nested Disjunctions." % disjunction.name
                    )
                logger.debug(
                    "Skipping Disjunction '%s' because it is nested in another "
                    "Disjunct." % disjunction.name
                )
                continue
            nested_owner = self._nested_disjunction_owner(disjunction, gdp_tree)
            if nested_owner is not None:
                # One of this Disjunction's Disjuncts owns a Disjunction.
                if disjunction in explicit:
                    raise GDP_Error(
                        "Disjunction '%s' contains a nested Disjunction (on "
                        "Disjunct '%s'). This transformation does not create "
                        "simple disjunctions from nested Disjunctions."
                        % (disjunction.name, nested_owner.name)
                    )
                logger.debug(
                    "Skipping Disjunction '%s' because it contains a nested "
                    "Disjunction." % disjunction.name
                )
                continue
            self._transform_disjunction(
                disjunction,
                build_expression,
                selected_constraints,
                transformation_blocks,
            )

    @staticmethod
    def _nested_disjunction_owner(disjunction, gdp_tree):
        # Return the first Disjunct of `disjunction` that owns a nested
        # Disjunction, or None. In the GDP tree, a Disjunct with a nested
        # Disjunction has that Disjunction as a child, so it is not a leaf.
        for disjunct in disjunction.disjuncts:
            if disjunct in gdp_tree.vertices and not gdp_tree.is_leaf(disjunct):
                return disjunct
        return None

    def _transform_disjunction(
        self, disjunction, build_expression, selected_constraints, transformation_blocks
    ):
        # Build the single Constraint expression for each (active) Disjunct,
        # skipping Disjuncts that have nothing to contribute.
        chosen = []
        for disjunct in disjunction.disjuncts:
            if not disjunct.active:
                continue
            expression = build_expression(disjunction, disjunct, selected_constraints)
            if expression is None:
                logger.debug(
                    "Disjunct '%s' has no active constraints to select, so it "
                    "is skipped in the simple disjunction generated from "
                    "Disjunction '%s'." % (disjunct.name, disjunction.name)
                )
                continue
            chosen.append((disjunct, expression))

        if not chosen:
            raise GDP_Error(
                "Cannot create a simple disjunction from Disjunction '%s': none "
                "of its active Disjuncts produced a constraint for the simple "
                "disjunction." % disjunction.name
            )

        trans_block = self._add_transformation_block(
            disjunction.parent_block(), transformation_blocks
        )

        # The simple Disjuncts for every Disjunction on this parent Block live in
        # one shared indexed container, so index the new ones off its current
        # length rather than restarting at 0.
        new_disjuncts = []
        for orig_disjunct, expression in chosen:
            idx = len(trans_block.simple_disjuncts)
            new_disjunct = trans_block.simple_disjuncts[idx]
            new_disjunct.constraint = Constraint(expr=expression)
            new_disjuncts.append(new_disjunct)

        simple_disjunction = Disjunction(expr=new_disjuncts)
        trans_block.add_component(
            unique_component_name(trans_block, 'simple_disjunction'), simple_disjunction
        )

        # Record the mapping in both directions (with hard references) so callers
        # can move between an original Disjunction and the simple Disjunction
        # generated from it. The forward map lives on the original Disjunction;
        # the reverse map lives in the transformation Block's private data.
        disjunction._transformation_map[self.transformation_name] = simple_disjunction
        trans_block.private_data().src_disjunction[simple_disjunction] = disjunction

        return simple_disjunction

    def _add_transformation_block(self, to_block, transformation_blocks):
        """Return the transformation Block on ``to_block``, creating it once.

        There is a single transformation Block per parent Block; all the simple
        Disjunctions built on that parent Block are placed on it.
        """
        if to_block in transformation_blocks:
            return transformation_blocks[to_block]

        trans_block_name = unique_component_name(
            to_block, '_pyomo_gdp_%s_reformulation' % self.transformation_name
        )
        transformation_blocks[to_block] = trans_block = Block()
        to_block.add_component(trans_block_name, trans_block)
        trans_block.simple_disjuncts = Disjunct(Any)

        return trans_block

    # ---------------------------------------------------------------------- #
    # Constraint selection methods                                           #
    #                                                                        #
    # Each builder gathers the source Constraints relevant to a Disjunct and #
    # returns the single relational expression to place in the corresponding #
    # simple Disjunct, or None to skip the Disjunct. The actual reduction of #
    # source Constraints to one expression is factored into                  #
    # _combine_sources so that aggregating methods can be added later.       #
    # ---------------------------------------------------------------------- #
    def _first_constraint_expression(self, disjunction, disjunct, selected_constraints):
        sources = self._own_active_constraints(disjunct)
        if not sources:
            return None
        return self._combine_sources(disjunction, disjunct, sources[:1])

    def _user_specified_expression(self, disjunction, disjunct, selected_constraints):
        if disjunct not in selected_constraints:
            # No selection for this Disjunct: skip it if it has nothing to keep,
            # otherwise the user left out a Disjunct we cannot reduce on our own.
            if not self._own_active_constraints(disjunct):
                return None
            raise GDP_Error(
                "Disjunct '%s' (in Disjunction '%s') has active constraints but "
                "was not assigned one in 'selected_constraints'. Assign a "
                "constraint for it, deactivate it, or use the 'first' "
                "constraint selection method." % (disjunct.name, disjunction.name)
            )
        sources = selected_constraints[disjunct]
        self._validate_selected_sources(disjunction, disjunct, sources)
        return self._combine_sources(disjunction, disjunct, sources)

    def _combine_sources(self, disjunction, disjunct, sources):
        """Reduce a Disjunct's selected source Constraints to one expression.

        The currently-implemented selection methods keep a single Constraint, so
        this references that Constraint's expression directly. Selection methods
        that aggregate several Constraints into one will perform that reduction
        here (for example, by combining the Constraint bodies into a new
        expression).
        """
        if len(sources) != 1:
            raise GDP_Error(
                "Disjunct '%s' (in Disjunction '%s') was assigned %d constraints, "
                "but the current selection methods keep exactly one constraint "
                "per Disjunct. (Aggregating several constraints into one is not "
                "yet implemented.)" % (disjunct.name, disjunction.name, len(sources))
            )
        # Reference the original Constraint's relational expression. This reuses
        # the original model Vars, so the new Constraint constrains exactly what
        # the original did, without touching the original.
        return sources[0].expr

    def _validate_selected_sources(self, disjunction, disjunct, sources):
        own = ComponentSet(self._own_active_constraints(disjunct))
        for constraint in sources:
            # In the spirit of duck-typing, we assume each selected object is a
            # Constraint and let it fail naturally below if it is not one.
            if not constraint.active:
                raise GDP_Error(
                    "The constraint '%s' selected for Disjunct '%s' is not "
                    "active. Only active constraints may be selected."
                    % (constraint.name, disjunct.name)
                )
            if constraint not in own:
                raise GDP_Error(
                    "The constraint '%s' selected for Disjunct '%s' is not one "
                    "of that Disjunct's own active constraints. (Constraints "
                    "inside a nested Disjunct cannot be selected.)"
                    % (constraint.name, disjunct.name)
                )

    @staticmethod
    def _own_active_constraints(disjunct):
        # Deterministic order, and does not descend into nested Disjuncts.
        return list(
            disjunct.component_data_objects(
                Constraint,
                active=True,
                descend_into=Block,
                sort=SortComponents.deterministic,
            )
        )

    # ---------------------------------------------------------------------- #
    # Mapping between original and simple Disjunctions                       #
    # ---------------------------------------------------------------------- #
    def get_simple_disjunction(self, src_disjunction):
        """Return the simple Disjunction generated from ``src_disjunction``.

        Raises a GDP_Error if ``src_disjunction`` was not transformed by this
        transformation.
        """
        simple = src_disjunction._transformation_map.get(self.transformation_name)
        if simple is None:
            raise GDP_Error(
                "Disjunction '%s' has not been transformed with the 'gdp.%s' "
                "transformation, so it has no simple disjunction."
                % (src_disjunction.name, self.transformation_name)
            )
        return simple

    def get_src_disjunction(self, simple_disjunction):
        """Return the original Disjunction that ``simple_disjunction`` was built from.

        Parameters
        ----------
        simple_disjunction: Disjunction generated by this transformation (i.e.,
            a ``simple_disjunction`` component on one of the reformulation
            Blocks created by this transformation).
        """
        trans_block = simple_disjunction.parent_block()
        src_map = trans_block.private_data().src_disjunction
        if simple_disjunction not in src_map:
            raise GDP_Error(
                "It does not appear that '%s' is a simple disjunction generated "
                "by the 'gdp.%s' transformation."
                % (simple_disjunction.name, self.transformation_name)
            )
        return src_map[simple_disjunction]
