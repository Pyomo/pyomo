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
"""
This module implements the ExternalGreyBoxConstraint class, which is used to
represent implicit constraints defined by external grey-box models within Pyomo.
"""

from __future__ import annotations
import sys
import logging
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

    def __init__(self, parent_model, implicit_constraint_id):
        self._parent_model = parent_model
        self._implicit_constraint_id = implicit_constraint_id

        self._ext_output_idx = None
        self._ext_eq_cons_idx = None

        ext_model = parent_model.get_external_model()

        if self._implicit_constraint_id in ext_model.equality_constraint_names():
            self._ext_eq_cons_idx = ext_model.equality_constraint_names().index(
                self._implicit_constraint_id
            )
        elif self._implicit_constraint_id in ext_model.output_names():
            self._ext_output_idx = ext_model.output_names().index(
                self._implicit_constraint_id
            )
        else:
            raise ValueError(
                f"Implicit_constraint_id '{self._implicit_constraint_id}' is not a valid identifier in "
                f"the external model."
            )

    @property
    def is_numeric_type(self):
        """
        Returns True if the body of this constraint is a numeric type (i.e., it can be evaluated to a number).
        """
        return True

    def __call__(self, exception=NOTSET):
        """Compute the value of the body of this constraint."""
        if self._ext_eq_cons_idx is not None:
            # For an implicit constraint, return the residual
            try:
                return self._parent_model.get_external_model().evaluate_equality_constraints()[
                    self._ext_eq_cons_idx
                ]
            except Exception as e:
                raise RuntimeError(
                    f"Error evaluating implicit equality constraint '{self._implicit_constraint_id}' "
                    "in external model. Have the external model inputs been set?"
                ) from e
        # For an output, the ExternalGreyBox will always return the value
        # of the output as a function of the inputs.
        # In this case, the "residual" of the implicit constraint is 0.0
        return 0.0

    def get_incident_variables(
        self, use_jacobian=False, jac_tolerance=JAC_ZERO_TOLERANCE
    ):
        """
        Get the variables that are incident on this implicit constraint.

        Parameters
        ----------
        use_jacobian : bool, optional
            If True, only include variables with non-zero Jacobian entries.
        jac_tolerance : float, optional
            The tolerance below which Jacobian entries are considered zero.

        Returns
        -------
        list of VarData
            List containing the variables that participate in the expression

        """
        # There are two ways incident variables could be defined for an implicit constraint:
        # 1) We consider all inputs to the external model to be incident on the implicit constraint
        # 2) We consider only the inputs that have a non-zero Jacobian entry for the implicit constraint
        # to be incident on the implicit constraint
        # Both have their uses, so we will support both.
        ext_model = self._parent_model.get_external_model()
        incident_variables = []

        if self._ext_output_idx is not None:
            # If this constraint is linked to an output variable, then that variable is also incident on the constraint
            incident_variables.append(
                self._parent_model.outputs[self._implicit_constraint_id]
            )

        if not use_jacobian:
            # If we are not using the Jacobian to determine incidence, then all inputs are incident
            for input_name in ext_model.input_names():
                incident_variables.append(self._parent_model.inputs[input_name])
        else:
            # If we are using the Jacobian to determine incidence, then only include variables with non-zero Jacobian entries
            # AL: To be even more robust, we could look at the Hessian too to catch cases where the Jacobian is just
            # passing through zero.
            if self._ext_eq_cons_idx is not None:
                jac = ext_model.evaluate_jacobian_equality_constraints().tocsr()
                con_idx = self._ext_eq_cons_idx
            else:
                jac = ext_model.evaluate_jacobian_outputs().tocsr()
                con_idx = self._ext_output_idx

            for input_name in ext_model.input_names():
                var_idx = ext_model.input_names().index(input_name)

                jacobian_entry = jac[con_idx, var_idx]

                if abs(jacobian_entry) >= jac_tolerance:
                    incident_variables.append(self._parent_model.inputs[input_name])

        return incident_variables


class ExternalGreyBoxConstraintData(ComponentData):
    """This class defines the data for a single algebraic constraint.

    Parameters
    ----------
    expr : ExpressionBase
        The Pyomo expression stored in this constraint.

    component : ExternalGreyBoxConstraint
        The ExternalGreyBoxConstraint object that owns this data.

    """

    __slots__ = ('_implicit_constraint_id', '_body')

    def __init__(self, implicit_constraint_id=None, component=None):
        #
        # These lines represent in-lining of the
        # following constructors:
        #   - ExternalGreyBoxConstraintData
        #   - ComponentData
        self._component = weakref_ref(component) if (component is not None) else None

        self._implicit_constraint_id = implicit_constraint_id

        # Placeholder for body
        self._body = None

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
            # Create the EGBConstraintBody object
            self._body = EGBConstraintBody(
                parent_model=self.parent_block(),
                implicit_constraint_id=self._implicit_constraint_id,
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
        # Refer back to the activate method to ensure message consistency.
        self.activate()


@ModelComponentFactory.register("General ExternalGreyBoxConstraint expressions.")
class ExternalGreyBoxConstraint(IndexedComponent):
    """
    This modeling component defines a ExternalGreyBoxConstraint for either an
    implicit equality constraint or output variable in an ExternalGreyBox model.

    Constructor arguments:
        implicit_constraint_id
            The identifier for this implicit constraint or output variable
        name
            A name for this component
        doc
            A text string describing this component

    Public class attributes:
        doc
            A text string describing this component
        name
            A name for this component
        active
            A boolean that is true if this component will be used to
            construct a model instance
        implicit_constraint_id
            The identifier for this implicit constraint or output variable

    Private class attributes:
        _constructed
            A boolean that is true if this component has been constructed
        _data
            A dictionary from the index set to component data objects
        _index
            The set of valid indices
        _model
            A weakref to the model that owns this component
        _parent
            A weakref to the parent block that owns this component
        _type
            The class type for the derived subclass
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
        expr=None,
        rule=None,
        implicit_constraint_id=None,
        name=None,
        doc=None,
    ): ...

    def __init__(self, *args, **kwargs):
        # Get id of the implicit constraint (either the equality_constraint_name or output_name)
        implicit_constraint_id = kwargs.pop('implicit_constraint_id', None)
        if implicit_constraint_id is not None:
            self._implicit_constraint_id = implicit_constraint_id
        else:
            raise ValueError(
                "ExternalGreyBoxConstraints must be provided with a 'implicit_constraint_id' argument "
            )

        # Check for normal Constraint arguments, and raise a TypeError if found
        rule = kwargs.pop('rule', None)
        expr = kwargs.pop('expr', None)

        if rule is not None:
            raise TypeError(
                "The 'rule' argument is not supported by ExternalGreyBoxConstraint. "
                "Use the 'implicit_constraint_id' argument instead."
            )
        if expr is not None:
            raise TypeError(
                "The 'expr' argument is not supported by ExternalGreyBoxConstraint. "
                "ExternalGreyBoxConstraints do not have explicit expressions."
            )

        kwargs.setdefault('ctype', ExternalGreyBoxConstraint)
        IndexedComponent.__init__(self, *args, **kwargs)

    def construct(self, data=None):
        """
        Construct the ExternalGreyBoxConstraint.
        """
        # First, check that the parent_block is an ExternalGreyBoxBlock
        if self.parent_block() is None or not hasattr(
            self.parent_block(), "get_external_model"
        ):
            raise ValueError(
                "ExternalGreyBoxConstraint components must be "
                "added to an ExternalGreyBoxBlock."
            )

        # Next, check that the implicit_constraint_id exists in the
        # external model
        external_model = self.parent_block().get_external_model()
        if not (
            self._implicit_constraint_id in external_model.equality_constraint_names()
            or self._implicit_constraint_id in external_model.output_names()
        ):
            raise ValueError(
                f"implicit_constraint_id '{self._implicit_constraint_id}' does not exist in the "
                f"external model associated with ExternalGreyBoxBlock '{self.parent_block().name}'."
            )

        super().construct(data=data)

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
    def implicit_constraint_id(self):
        """
        Identifier for this implicit constraint or output variable in the external model.
        """
        return self._implicit_constraint_id

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

    @overload
    def __getitem__(self, index) -> ExternalGreyBoxConstraintData: ...

    __getitem__ = IndexedComponent.__getitem__  # type: ignore
