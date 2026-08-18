# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________
#
#  Additional contributions Copyright (c) 2026 OLI Systems, Inc.
#  ___________________________________________________________________________________
"""
This module implements the ExternalGreyBoxConstraint class, which is used to
represent implicit constraints defined by external grey-box models within Pyomo.
"""

from __future__ import annotations
from operator import index
import sys
import logging
from collections.abc import Mapping
from typing import Union, Type
from weakref import ref as weakref_ref

from pyomo.common.pyomo_typing import overload
from pyomo.common.formatting import tabular_writer
from pyomo.common.modeling import NOTSET

from pyomo.core.expr.numvalue import value
from pyomo.core.base.component import ComponentData, ModelComponentFactory
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.indexed_component import IndexedComponent, UnindexedComponent_set
from pyomo.core.base.disable_methods import disable_methods

logger = logging.getLogger('pyomo.contrib.pynumero')


JAC_ZERO_TOLERANCE = 1e-8


class EGBConstraintBody:
    """
    This class creates a representation of the "body" of an implicit constraint in
    an ExternalGreyBox model.

    Currently, this supports:
    * evaluation of the residual of the implicit constraint
    * identification of incident variables in the implicit constraint
    """

    def __init__(self, grey_box, constraint_id):
        # Store a weakref to the parent ExternalGreyBoxBlock to avoid a
        # circular reference: EGBData -> EGBConstraint -> EGBConstraintBody
        # -> EGBData.  EGBData always outlives EGBConstraintBody (it owns the
        # chain), so a weakref from child back to parent is safe.
        self._egb = weakref_ref(grey_box)
        self._constraint_id = constraint_id

        self._output_idx = None
        self._eq_cons_idx = None

        ext_model = self._egb().get_external_model()

        if self._constraint_id in ext_model.equality_constraint_names():
            self._eq_cons_idx = ext_model.equality_constraint_names().index(
                self._constraint_id
            )
        elif self._constraint_id in ext_model.output_names():
            self._output_idx = ext_model.output_names().index(self._constraint_id)
        else:
            raise ValueError(
                f"Implicit_constraint_id '{self._constraint_id}' is not a valid identifier in "
                f"the external model."
            )

    def _dereference_egb(self):
        """Dereference the weakref to the parent ExternalGreyBoxBlock.

        Raises
        ------
        ReferenceError
            If the ExternalGreyBoxBlock has been garbage-collected while this
            EGBConstraintBody is still alive.  This indicates that the caller
            is holding a constraint body beyond the lifetime of the owning
            block.
        """
        egb = self._egb()
        if egb is None:
            raise ReferenceError(
                f"The ExternalGreyBoxBlock that owns implicit constraint "
                f"'{self._constraint_id}' has been garbage-collected while "
                "this EGBConstraintBody is still alive. Ensure the block "
                "remains in scope as long as any constraint bodies derived "
                "from it are used."
            )
        return egb

    def get_output_var(self):
        """
        If this EGBConstraintBody corresponds to an output variable, return the corresponding Pyomo VarData object.
        Otherwise, return None.
        """
        if self._output_idx is not None:
            out_var = list(self._dereference_egb().outputs.values())[self._output_idx]
            return out_var
        return None

    @property
    def is_numeric_type(self):
        """
        Returns True if the body of this constraint is a numeric type (i.e., it can be evaluated to a number).
        """
        return True

    def __call__(self, exception=NOTSET):
        """Compute the value of the body of this constraint."""
        if self._eq_cons_idx is not None:
            # For an implicit constraint, return the residual
            try:
                return (
                    self._dereference_egb()
                    .get_external_model()
                    .evaluate_equality_constraints()[self._eq_cons_idx]
                )
            except Exception as e:
                raise RuntimeError(
                    f"Error evaluating implicit equality constraint '{self._constraint_id}' "
                    "in external model. Have the external model inputs been set?"
                ) from e
        # For an output, the ExternalGreyBox will always return the value
        # of the output as a function of the inputs.
        evaluated_value = (
            self._dereference_egb()
            .get_external_model()
            .evaluate_outputs()[self._output_idx]
        )
        var_value = value(self.get_output_var(), exception=exception)
        return var_value - evaluated_value

    def identify_variables(self):
        """
        Get the variables that are incident on this implicit constraint.

        This method evaluates the Jacobian of the external model to determine
        which input variables have a structural non-zero entry for this
        constraint.  As a result, **the external model's input values must be
        initialised** (via :meth:`ExternalGreyBoxModel.set_input_values`) before
        calling this method; if they have not been set, the behaviour depends on
        the :class:`ExternalGreyBoxModel` implementation — typically the
        Jacobian evaluation will raise or return incorrect sparsity.

        Returns
        -------
        list of VarData
            List of Pyomo variable data objects that participate in this
            implicit constraint.  For output-variable constraints the output
            variable itself is always included, followed by any input variables
            with a structural non-zero Jacobian entry.

        Raises
        ------
        ValueError
            If any of the incident variables are fixed.  EGB variables cannot
            be fixed; fixing one is a modelling error that would otherwise
            surface as a cryptic failure at solve time.
        RuntimeError
            If the external model raises when its Jacobian is evaluated.

        Notes
        -----
        There are (at least) three ways incident variables could be defined for
        an implicit constraint:

        1. All inputs to the external model are considered incident.
        2. The sparsity pattern of the Jacobian returned by the
           ExternalGreyBoxModel defines which inputs are potentially incident
           (entries may be structurally non-zero but numerically zero at the
           current point).
        3. Only inputs with a numerically non-zero Jacobian entry are
           considered incident.

        Option 2 is used here as it is the most general.
        """
        egb = self._dereference_egb()
        ext_model = egb.get_external_model()
        incident_variables = []

        # First, if this implicit constraint corresponds to an output variable, we need to include that output
        # variable as an incident variable since it is part of the expression for the implicit constraint.
        out_var = self.get_output_var()
        if out_var is not None:
            incident_variables.append(out_var)

        # Next, get the Jacobian for the external model
        try:
            if self._eq_cons_idx is not None:
                jac = ext_model.evaluate_jacobian_equality_constraints().tocsr()
                con_idx = self._eq_cons_idx
            else:
                jac = ext_model.evaluate_jacobian_outputs().tocsr()
                con_idx = self._output_idx
        except Exception as e:
            raise RuntimeError(
                f"Error evaluating Jacobian for external model when getting incident variables for implicit constraint "
                f"'{self._constraint_id}'. Original error message: {str(e)}"
            ) from e

        # Get all variables with entries for this constraint
        # We do not check value, as we assume entries indicate potential incident variables,
        # however they may currently have zero derivative values.
        var_indices = jac.getrow(con_idx).indices
        incident_variables.extend(list(egb.inputs.values())[j] for j in var_indices)

        # EGB variables cannot be fixed — the solver interface will raise a
        # cryptic error at solve time if any are.  Check here to surface the
        # problem early with an actionable message.
        fixed_vars = [v.name for v in incident_variables if v.fixed]
        if fixed_vars:
            raise ValueError(
                f"ExternalGreyBoxBlock variables cannot be fixed, but the "
                f"following variables incident to implicit constraint "
                f"'{self._constraint_id}' are fixed: {fixed_vars}. Use an "
                "equality Constraint instead of fixing an EGB input or output "
                "variable."
            )

        return incident_variables


class ExternalGreyBoxConstraintData(ComponentData):
    """Data object for a single implicit ExternalGreyBoxConstraint.

    Parameters
    ----------
    implicit_constraint_id : str, optional
        The identifier for this implicit constraint or output variable in
        the external model.
    component : ExternalGreyBoxConstraint, optional
        The ExternalGreyBoxConstraint component that owns this data object.
        ``component`` is the last positional argument, following the
        convention of other Pyomo ComponentData subclasses.  The ``_index``
        attribute is **not** accepted here; it is set by the owning
        component (via ``construct`` or ``__setitem__``) after the object
        is created, matching the standard Pyomo pattern.

    """

    # _index is intentionally omitted: ComponentData already defines it in
    # its own __slots__.  Redefining it here would create a second, shadowing
    # slot descriptor and waste memory.
    __slots__ = ('_implicit_constraint_id', '_body')

    def __init__(self, implicit_constraint_id=None, component=None):
        #
        # These lines represent in-lining of the
        # following constructors:
        #   - ExternalGreyBoxConstraintData
        #   - ComponentData
        # Note: _index is intentionally left as NOTSET here; callers are
        # responsible for setting obj._index after construction, following
        # the standard Pyomo ComponentData convention.
        self._component = weakref_ref(component) if (component is not None) else None
        self._index = NOTSET
        self._implicit_constraint_id = implicit_constraint_id

        # Placeholder for body
        self._body = None

    def _validate_implicit_constraint_id(self):
        """Validate ``implicit_constraint_id`` against the attached external model.

        Checks that the identifier is a string and that it matches either an
        equality-constraint name or an output name in the external model.
        Must be called after the owning component has been added to an
        ExternalGreyBoxBlock (i.e. from within ``construct()``), so that
        ``self.parent_block()`` is available.

        Raises
        ------
        TypeError
            If ``implicit_constraint_id`` is not a string.
        ValueError
            If the identifier does not exist in the external model.
        """
        implicit_constraint_id = self._implicit_constraint_id
        if not isinstance(implicit_constraint_id, str):
            raise TypeError(
                "ExternalGreyBoxConstraint implicit_constraint_id values must be "
                f"strings. Invalid value: {implicit_constraint_id!r}"
            )
        external_model = self.parent_block().get_external_model()
        if not (
            implicit_constraint_id in external_model.equality_constraint_names()
            or implicit_constraint_id in external_model.output_names()
        ):
            raise ValueError(
                f"implicit_constraint_id '{implicit_constraint_id}' does not exist "
                f"in the external model associated with ExternalGreyBoxBlock "
                f"'{self.parent_block().name}'."
            )

    def __call__(self, exception=NOTSET):
        """Compute the value of the body of this constraint."""
        body = value(self.body, exception=exception)
        return body

    def to_bounded_expression(self, *args, **kwargs):
        """Duck-type method from ConstraintData.

        Raises
        ------

        TypeError
            Always. ExternalGreyBoxConstraints do not have an explicit expression.

        """
        raise TypeError(
            "ExternalGreyBoxConstraints do not have an explicit expression."
        )

    @property
    def body(self):
        """Value (residual) of the implicit ExternalGreyBoxConstraint."""
        if self._body is None:
            # Create the EGBConstraintBody object. EGBConstraintBody holds a
            # weakref back to the parent block (see EGBConstraintBody.__init__),
            # so storing a strong reference here does not create a cycle.
            self._body = EGBConstraintBody(
                grey_box=self.parent_block(), constraint_id=self._implicit_constraint_id
            )
        return self._body

    @property
    def lower(self):
        """The lower bound of a ExternalGreyBoxConstraint.

        Implicit constraints always have a lower bound of 0.

        """
        return 0.0

    @property
    def upper(self):
        """Access the upper bound of a ExternalGreyBoxConstraint.

        Implicit constraints always have an upper bound of 0.

        """
        return 0.0

    @property
    def lb(self):
        """float : the value of the lower bound of a ExternalGreyBoxConstraint expression.

        Implicit constraints always have a lower bound of 0.
        """
        return 0.0

    @property
    def ub(self):
        """float : the value of the upper bound of a ExternalGreyBoxConstraint expression.

        Implicit constraints always have an upper bound of 0.
        """
        return 0.0

    @property
    def equality(self):
        """bool : True. ExternalGreyBoxConstraints are always equalities."""
        return True

    @property
    def strict_lower(self):
        """bool : True if this ExternalGreyBoxConstraint has a strict lower bound."""
        return False

    @property
    def strict_upper(self):
        """bool : True if this ExternalGreyBoxConstraint has a strict upper bound."""
        return False

    def has_lb(self):
        """Returns :const:`True`. Implicit constraints always have a lower bound."""
        return True

    def has_ub(self):
        """Returns :const:`True`. Implicit constraints always have an upper bound."""
        return True

    @property
    def expr(self):
        """Return the expression associated with this ExternalGreyBoxConstraint.

        Raises:
            TypeError
                Always. ExternalGreyBoxConstraints do not have an explicit expression.
        """
        raise TypeError(
            "ExternalGreyBoxConstraints do not have an explicit expression."
        )

    def get_value(self):
        """Get the expression on this ExternalGreyBoxConstraint.

        Raises:
            TypeError
                Always. ExternalGreyBoxConstraints do not have an explicit expression.
        """
        return self.expr

    def set_value(self, expr):
        """Set the expression on this ExternalGreyBoxConstraint.

        Raises:
            TypeError
                Always. ExternalGreyBoxConstraints do not have an explicit expression.
        """
        raise TypeError(
            "ExternalGreyBoxConstraints do not have an explicit expression."
        )

    def lslack(self):
        """
        Returns the value of f(x)-L for ExternalGreyBoxConstraints of the form:
            L <= f(x) (<= U)
            (U >=) f(x) >= L
        """
        return value(self.body)

    def uslack(self):
        """
        Returns the value of U-f(x) for ExternalGreyBoxConstraints of the form:
            (L <=) f(x) <= U
            U >= f(x) (>= L)
        """
        return -value(self.body)

    def slack(self):
        """
        Returns the smaller of lslack and uslack values
        """
        return -abs(value(self.body))

    # Duck-typing a few common Constraint methods and properties
    @property
    def active(self):
        """bool : True if this ExternalGreyBoxConstraint is active."""
        return self.parent_block().active

    def activate(self):
        """Raise a TypeError, as ExternalGreyBoxConstraints cannot be activated or deactivated."""
        raise TypeError(
            "ExternalGreyBoxConstraints cannot be activated or deactivated individually. "
            "Activate or deactivate the parent ExternalGreyBoxBlock instead."
        )

    def deactivate(self):
        """Raise a TypeError, as ExternalGreyBoxConstraints cannot be activated or deactivated."""
        raise TypeError(
            "ExternalGreyBoxConstraints cannot be activated or deactivated individually. "
            "Activate or deactivate the parent ExternalGreyBoxBlock instead."
        )


@ModelComponentFactory.register("General ExternalGreyBoxConstraint expressions.")
class ExternalGreyBoxConstraint(IndexedComponent):
    """An implicit constraint (or output-variable residual) from an ExternalGreyBoxBlock.

    Each instance represents one or more implicit equalities of the form
    ``f(inputs, outputs) == 0``, where ``f`` is evaluated by the attached
    :class:`ExternalGreyBoxModel`.

    Parameters
    ----------
    *indexes :
        One or more index sets, as for any Pyomo :class:`IndexedComponent`.
        For the common case where each index value is already the string
        identifier of the constraint in the external model, no
        ``implicit_constraint_ids`` argument is needed — the index is used
        directly as the ID (see below).
    implicit_constraint_ids : str or Mapping, optional
        Controls how each data object's constraint identifier is resolved.
        The plural name reflects that this argument acts as the
        *construction rule* for the whole component — analogous to
        ``rule``/``expr`` on :class:`Constraint` — supplying one identifier
        per datum rather than being a per-datum property itself.

        *Why this lives on the container and not only on the Data class:*
        :class:`ExternalGreyBoxConstraintData` objects are not created until
        :meth:`construct` is called (after the component is attached to a
        block).  ``implicit_constraint_ids`` is therefore a staging store
        that :meth:`construct` reads to initialise each data object's
        ``implicit_constraint_id``.  For the scalar case,
        :class:`ScalarExternalGreyBoxConstraint` is itself both the
        container *and* the single data object; construct copies the value
        across so that the per-datum ``_implicit_constraint_id`` attribute
        is also set.

        The three supported forms are:

        ``None`` *(default for indexed)*
            Each index value is used as the constraint identifier.  This is
            the expected usage when the index set is already the collection
            of constraint-name strings returned by
            :meth:`ExternalGreyBoxModel.equality_constraint_names` or
            :meth:`ExternalGreyBoxModel.output_names`, e.g.::

                self.eq_constraints = ExternalGreyBoxConstraint(
                    self._equality_constraint_set
                )

        ``str`` *(required for scalar)*
            The single string identifier of the constraint or output
            variable in the external model, e.g.::

                m.egb.c = ExternalGreyBoxConstraint(implicit_constraint_ids='pdrop')

            Omitting this argument for a scalar constraint raises
            :class:`ValueError` at construct-time.

        ``Mapping[index, str]``
            An explicit index-to-name mapping.  Use this when the index set
            does not coincide with the constraint-name strings, e.g.::

                m.egb.c = ExternalGreyBoxConstraint(
                    [1, 2],
                    implicit_constraint_ids={1: 'pdrop', 2: 'Pout'},
                )

            The mapping must cover every index in the component's index set
            exactly; missing or extra keys raise :class:`ValueError` at
            construct-time.
    name : str, optional
        Name for this component.
    doc : str, optional
        A text string describing this component.
    """

    _ComponentDataClass = ExternalGreyBoxConstraintData

    @overload
    def __new__(
        cls: Type[ScalarExternalGreyBoxConstraint], *args, **kwds
    ) -> ScalarExternalGreyBoxConstraint: ...

    @overload
    def __new__(
        cls: Type[IndexedExternalGreyBoxConstraint], *args, **kwds
    ) -> IndexedExternalGreyBoxConstraint: ...

    @overload
    def __new__(
        cls: Type[ExternalGreyBoxConstraint], *args, **kwds
    ) -> Union[ScalarExternalGreyBoxConstraint, IndexedExternalGreyBoxConstraint]: ...

    def __new__(cls, *args, **kwds):
        if cls != ExternalGreyBoxConstraint:
            return super().__new__(cls)
        if not args or (args[0] is UnindexedComponent_set and len(args) == 1):
            return super().__new__(AbstractScalarExternalGreyBoxConstraint)
        return super().__new__(IndexedExternalGreyBoxConstraint)

    @overload
    def __init__(
        self,
        *indexes,
        implicit_constraint_ids: str | Mapping | None = None,
        name=None,
        doc=None,
    ): ...

    def __init__(self, *args, **kwargs):
        # Store the construction rule for constraint identifiers.  This is a
        # staging store (analogous to Constraint._rule) that construct() reads
        # to set each data object's per-datum _implicit_constraint_id.
        self._implicit_constraint_ids = kwargs.pop('implicit_constraint_ids', None)

        kwargs.setdefault('ctype', ExternalGreyBoxConstraint)
        IndexedComponent.__init__(self, *args, **kwargs)
        # Validation of implicit_constraint_ids is deferred to construct(), where
        # the parent block and its external model are guaranteed to be available.

    def construct(self, data=None):
        """
        Construct the ExternalGreyBoxConstraint.
        """
        if self._constructed:
            return

        # First, check that the parent_block is an ExternalGreyBoxBlock
        if self.parent_block() is None or not hasattr(
            self.parent_block(), "get_external_model"
        ):
            raise ValueError(
                "ExternalGreyBoxConstraint components must be "
                "added to an ExternalGreyBoxBlock."
            )

        super().construct(data=data)

        self._constructed = True

    def _pprint(self):
        """
        Return data that will be printed for this component.
        """
        return (
            [
                ("Size", len(self)),
                ("Index", self._index_set if self.is_indexed() else None),
                ("Active", self.active),
            ],
            self.items,
            ("Lower", "Body", "Upper", "Active"),
            lambda k, v: [
                "-Inf" if v.lower is None else v.lower,
                v.body,
                "+Inf" if v.upper is None else v.upper,
                v.active,
            ],
        )

    @property
    def implicit_constraint_ids(self):
        """The construction-rule argument passed at component creation time.

        This is the ``str | Mapping | None`` value supplied as
        ``implicit_constraint_ids`` to the constructor.  It is the
        container-level staging store used by :meth:`construct` to
        initialise each data object's per-datum ``_implicit_constraint_id``.
        See the class docstring for the three supported forms.
        """
        return self._implicit_constraint_ids

    def display(self, prefix="", ostream=None):
        """
        Print component state information

        This duplicates logic in Component.pprint()
        """
        if not self.active:
            return
        if ostream is None:
            ostream = sys.stdout
        tab = "    "
        ostream.write(prefix + self.local_name + " : ")
        ostream.write("Size=" + str(len(self)))

        ostream.write("\n")
        tabular_writer(
            ostream,
            prefix + tab,
            ((k, v) for k, v in self._data.items() if v.active),
            ("Lower", "Body", "Upper"),
            lambda k, v: [
                value(v.lower, exception=False),
                value(v.body, exception=False),
                value(v.upper, exception=False),
            ],
        )


class ScalarExternalGreyBoxConstraint(
    ExternalGreyBoxConstraintData, ExternalGreyBoxConstraint
):
    """
    ScalarExternalGreyBoxConstraint is the implementation representing a single,
    non-indexed ExternalGreyBoxConstraint.
    """

    def __init__(self, *args, **kwds):
        ExternalGreyBoxConstraintData.__init__(self, component=self)
        ExternalGreyBoxConstraint.__init__(self, *args, **kwds)
        self._index = UnindexedComponent_index

        # Set _data here, as it isn't getting set elsewhere
        self._data[None] = self

    def construct(self, data=None):
        if self._constructed:
            return

        ExternalGreyBoxConstraint.construct(self, data=data)

        # For scalar constraints the ID string must be given explicitly;
        # there is no index to fall back on.
        if self._implicit_constraint_ids is None:
            raise ValueError(
                "The 'implicit_constraint_ids' argument must be provided for "
                "non-indexed ExternalGreyBoxConstraints."
            )
        # Copy the resolved ID from the container staging attribute to the
        # per-datum data slot.  For the scalar case self is simultaneously
        # the container and the single data object, so the two attributes
        # are distinct: _implicit_constraint_ids (container, __dict__) and
        # _implicit_constraint_id (data, __slots__ from ExternalGreyBoxConstraintData).
        self._implicit_constraint_id = self._implicit_constraint_ids
        self._validate_implicit_constraint_id()

    #
    # Singleton ExternalGreyBoxConstraints are strange in that we want them to be
    # both be constructed but have len() == 0 when not initialized with
    # anything (at least according to the unit tests that are
    # currently in place). So during initialization only, we will
    # treat them as "indexed" objects where things like
    # Constraint.Skip are managed. But after that they will behave
    # like ExternalGreyBoxConstraintData objects where set_value does not handle
    # Constraint.Skip but expects a valid expression or None.
    #
    @property
    def body(self):
        """The body (variable portion) of a ExternalGreyBoxConstraint expression."""
        if not self._data:
            raise ValueError(
                f"Accessing the body of ScalarExternalGreyBoxConstraint "
                f"'{self.name}' before the ExternalGreyBoxConstraint has been assigned. "
                "There is currently nothing to access."
            )
        return ExternalGreyBoxConstraintData.body.fget(self)

    @property
    def lower(self):
        """The lower bound of a ExternalGreyBoxConstraint expression.

        This is the fixed lower bound of a ExternalGreyBoxConstraint as a Pyomo
        expression.  This may contain potentially variable terms
        that are currently fixed.  If there is no lower bound, this will
        return `None`.

        """
        if not self._data:
            raise ValueError(
                f"Accessing the lower bound of ScalarExternalGreyBoxConstraint "
                f"'{self.name}' before the ExternalGreyBoxConstraint has been assigned. "
                "There is currently nothing to access."
            )
        return ExternalGreyBoxConstraintData.lower.fget(self)

    @property
    def upper(self):
        """Access the upper bound of a ExternalGreyBoxConstraint expression.

        This is the fixed upper bound of a ExternalGreyBoxConstraint as a Pyomo
        expression.  This may contain potentially variable terms
        that are currently fixed.  If there is no upper bound, this will
        return `None`.

        """
        if not self._data:
            raise ValueError(
                f"Accessing the upper bound of ScalarExternalGreyBoxConstraint "
                f"'{self.name}' before the ExternalGreyBoxConstraint has been assigned. "
                "There is currently nothing to access."
            )
        return ExternalGreyBoxConstraintData.upper.fget(self)

    @property
    def equality(self):
        """bool : True if this is an equality ExternalGreyBoxConstraint."""
        if not self._data:
            raise ValueError(
                f"Accessing the equality flag of ScalarExternalGreyBoxConstraint "
                f"'{self.name}' before the ExternalGreyBoxConstraint has been assigned. "
                "There is currently nothing to access."
            )
        return ExternalGreyBoxConstraintData.equality.fget(self)

    @property
    def strict_lower(self):
        """bool : True if this ExternalGreyBoxConstraint has a strict lower bound."""
        if not self._data:
            raise ValueError(
                f"Accessing the strict_lower flag of ScalarExternalGreyBoxConstraint "
                f"'{self.name}' before the ExternalGreyBoxConstraint has been assigned. "
                "There is currently nothing to access."
            )
        return ExternalGreyBoxConstraintData.strict_lower.fget(self)

    @property
    def strict_upper(self):
        """bool : True if this ExternalGreyBoxConstraint has a strict upper bound."""
        if not self._data:
            raise ValueError(
                f"Accessing the strict_upper flag of ScalarExternalGreyBoxConstraint "
                f"'{self.name}' before the ExternalGreyBoxConstraint has been assigned. "
                "There is currently nothing to access."
            )
        return ExternalGreyBoxConstraintData.strict_upper.fget(self)

    def clear(self):
        self._data = {}

    def set_value(self, expr):
        """Set the expression on this ExternalGreyBoxConstraint."""
        if not self._data:
            self._data[None] = self
        return super().set_value(expr)

    #
    # Leaving this method for backward compatibility reasons.
    # (probably should be removed)
    #
    def add(self, index, expr):
        """Add a ExternalGreyBoxConstraint with a given index."""
        if index is not None:
            raise ValueError(
                f"ScalarExternalGreyBoxConstraint object '{self.name}' does not accept "
                f"index values other than None. Invalid value: {index}"
            )
        self.set_value(expr)
        return self


@disable_methods(
    {
        '__call__',
        'add',
        'set_value',
        'to_bounded_expression',
        'expr',
        'body',
        'lower',
        'upper',
        'equality',
        'strict_lower',
        'strict_upper',
    }
)
class AbstractScalarExternalGreyBoxConstraint(ScalarExternalGreyBoxConstraint):
    """
    Implementation of abstract ExternalGreyBoxConstraints.
    """


class IndexedExternalGreyBoxConstraint(ExternalGreyBoxConstraint):
    """
    Implementation of indexed ExternalGreyBoxConstraints.
    """

    #
    # Leaving this method for backward compatibility reasons
    #
    # Note: Beginning after Pyomo 5.2 this method will now validate that
    # the index is in the underlying index set (through 5.2 the index
    # was not checked).
    #
    def add(self, index, expr):
        """Add a ExternalGreyBoxConstraint with a given index."""
        return self.__setitem__(index, expr)

    def construct(self, data=None):
        super().construct(data=data)

        if not self._data:
            # Validate the implicit_constraint_ids mapping before creating data
            # objects.  This mirrors the shape-checking that other indexed
            # components perform in construct() rather than __init__().
            if isinstance(self._implicit_constraint_ids, Mapping):
                valid_indices = set(self.index_set())
                provided_indices = set(self._implicit_constraint_ids.keys())
                missing = valid_indices - provided_indices
                extra = provided_indices - valid_indices
                if missing or extra:
                    msg = (
                        "For indexed ExternalGreyBoxConstraints, the "
                        "'implicit_constraint_ids' mapping keys must exactly "
                        "match the component index set."
                    )
                    if missing:
                        msg += f" Missing keys: {sorted(missing, key=str)}."
                    if extra:
                        msg += f" Invalid keys: {sorted(extra, key=str)}."
                    raise ValueError(msg)
            elif self._implicit_constraint_ids is not None:
                raise TypeError(
                    "For indexed ExternalGreyBoxConstraints, the "
                    "'implicit_constraint_ids' argument must be a mapping from "
                    "index to identifier or None."
                )

            for idx in self.index_set():
                # Resolve the per-datum identifier: use the mapping when
                # provided, otherwise fall back to the index value itself.
                if self._implicit_constraint_ids is not None:
                    implicit_constraint_id = self._implicit_constraint_ids[idx]
                else:
                    implicit_constraint_id = idx

                obj = self._ComponentDataClass(
                    implicit_constraint_id=implicit_constraint_id, component=self
                )
                obj._index = idx
                # Validate against the external model on the data object itself,
                # keeping the check close to the data and avoiding a separate
                # module-level helper function.
                obj._validate_implicit_constraint_id()
                self._data[idx] = obj

    @overload
    def __getitem__(self, index) -> ExternalGreyBoxConstraintData: ...

    __getitem__ = IndexedComponent.__getitem__  # type: ignore
