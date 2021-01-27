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
import weakref

from pyomo.core.base.set_types import Any
from pyomo.core.expr.numvalue import value
from pyomo.core.expr.numeric_expr import LinearExpression
from pyomo.core.base.plugin import ModelComponentFactory
from pyomo.repn.standard_repn import StandardRepn
from pyomo.core.base.constraint import (IndexedConstraint,
                                        _ConstraintData)
from pyomo.common.gc_manager import PauseGC
from pyomo.common.log import is_debug_set

import six
from six.moves import xrange

if six.PY3:
    from collections.abc import Mapping as collections_Mapping
else:
    from collections import Mapping as collections_Mapping


logger = logging.getLogger('pyomo.core')

class _MatrixConstraintData(_ConstraintData):
    """
    This class defines the data for a single linear constraint
        derived from a canonical form Ax=b constraint.

    Constructor arguments:
        index           The index of this component within the container.
        component       The Constraint object that owns this data.

    Public class attributes:
        active          A boolean that is true if this constraint is
                            active in the model.
        body            The Pyomo expression for this constraint
        lower           The Pyomo expression for the lower bound
        upper           The Pyomo expression for the upper bound
        equality        A boolean that indicates whether this is an
                            equality constraint
        strict_lower    A boolean that indicates whether this
                            constraint uses a strict lower bound
        strict_upper    A boolean that indicates whether this
                            constraint uses a strict upper bound

    Private class attributes:
        _component      The objective component.
        _active         A boolean that indicates whether this data is active
        _index          The row index into the main coefficient matrix
    """

    __slots__ = ('_index',)

    # the super secret flag that makes the writers
    # handle _MatrixConstraintData objects more efficiently
    _linear_canonical_form = True

    #
    # Define methods that writers expect when the
    # _linear_canonical_form flag is True
    #

    def canonical_form(self, compute_values=True):
        """Build a canonical representation of the body of
        this constraints"""
        comp = self.parent_component()
        index = self._index
        data = comp._A_data
        indices = comp._A_indices
        indptr = comp._A_indptr
        x = comp._x

        variables = []
        coefficients = []
        constant = 0
        for p in xrange(indptr[index],
                        indptr[index+1]):
            v = x[indices[p]]
            c = data[p]
            if not v.fixed:
                variables.append(v)
                if compute_values:
                    coefficients.append(value(c))
                else:
                    coefficients.append(c)
            else:
                if compute_values:
                    constant += value(c) * v.value
                else:
                    constant += c * v
        repn = StandardRepn()
        repn.linear_vars = tuple(variables)
        repn.linear_coefs = tuple(coefficients)
        repn.constant = constant
        return repn

    def __init__(self, index, component_ref):
        #
        # These lines represent in-lining of the
        # following constructors:
        #   - _ConstraintData,
        #   - ActiveComponentData
        #   - ComponentData
        self._component = component_ref
        self._active = True

        # row index into the sparse matrix stored on the parent
        assert index >= 0
        self._index = index

    def __getstate__(self):
        """
        This method must be defined because this class uses slots.
        """
        result = super(_MatrixConstraintData, self).__getstate__()
        result['_index'] = self._index
        return result

    # Since this class requires no special processing of the state
    # dictionary, it does not need to implement __setstate__()

    #
    # Override the default interface methods to
    # avoid generating the body expression where
    # possible
    #

    def __call__(self, exception=True):
        """Compute the value of the body of this constraint."""
        comp = self.parent_component()
        index = self._index
        data = comp._A_data
        indices = comp._A_indices
        indptr = comp._A_indptr
        x = comp._x
        ptrs = xrange(indptr[index],
                      indptr[index+1])
        try:
            return sum(x[indices[p]].value * data[p]
                       for p in ptrs)
        except (ValueError, TypeError):
            if exception:
                raise
            return None

    def has_lb(self):
        """Returns :const:`False` when the lower bound is
        :const:`None` or negative infinity"""
        lb = self.lower
        return (lb is not None) and \
            (lb != float('-inf'))

    def has_ub(self):
        """Returns :const:`False` when the upper bound is
        :const:`None` or positive infinity"""
        ub = self.upper
        return (ub is not None) and \
            (ub != float('inf'))

    def lslack(self):
        """Lower slack (body - lb). Returns :const:`None` if
        a value for the body can not be computed."""
        # this method is written so that constraint
        # types that build the body expression on the
        # fly do not have to here
        body = self(exception=False)
        if body is None:
            return None
        lb = self.lower
        if lb is None:
            lb = -float('inf')
        else:
            lb = value(lb)
        return body - lb

    def uslack(self):
        """Upper slack (ub - body). Returns :const:`None` if
        a value for the body can not be computed."""
        # this method is written so that constraint
        # types that build the body expression on the
        # fly do not have to here
        body = self(exception=False)
        if body is None:
            return None
        ub = self.upper
        if ub is None:
            ub = float('inf')
        else:
            ub = value(ub)
        return ub - body

    def slack(self):
        """min(lslack, uslack). Returns :const:`None` if a
        value for the body can not be computed."""
        # this method is written so that constraint
        # types that build the body expression on the
        # fly do not have to here
        body = self(exception=False)
        if body is None:
            return None
        return min(self.lslack(), self.uslack())

    #
    # Override some default implementations on ComponentData
    #

    def index(self):
        return self._index

    #
    # Abstract Interface (_ConstraintData)
    #

    @property
    def body(self):
        """Access the body of a constraint expression."""
        comp = self.parent_component()
        index = self._index
        data = comp._A_data
        indices = comp._A_indices
        indptr = comp._A_indptr
        x = comp._x
        ptrs = xrange(indptr[index],
                      indptr[index+1])
        return LinearExpression(
            linear_vars=[x[indices[p]] for p in ptrs],
            linear_coefs=[data[p] for p in ptrs],
            constant=0
        )

    @property
    def lower(self):
        """Access the lower bound of a constraint
        expression."""
        comp = self.parent_component()
        index = self._index
        return comp._lower[index]

    @property
    def upper(self):
        """Access the upper bound of a constraint
        expression."""
        comp = self.parent_component()
        index = self._index
        return comp._upper[index]

    @property
    def equality(self):
        """A boolean indicating whether this is an equality
        constraint."""
        comp = self.parent_component()
        index = self._index
        if (comp._lower[index] is None) or \
           (comp._upper[index] is None):
            return False
        return comp._lower[index] == comp._upper[index]

    @property
    def strict_lower(self):
        """A boolean indicating whether this constraint has
        a strict lower bound."""
        return False

    @property
    def strict_upper(self):
        """A boolean indicating whether this constraint has
        a strict upper bound."""
        return False

    def set_value(self, expr):
        """Set the expression on this constraint."""
        raise NotImplementedError(
            "MatrixConstraint row elements can not be updated"
        )


@ModelComponentFactory.register(
                   "A set of constraint expressions in Ax=b form.")
class MatrixConstraint(collections_Mapping,
                       IndexedConstraint):
    """
    Defines a set of linear constraints of the form:

       lb <= Ax <= ub

    where A is specified in the standard compressed sparse
    row (CSR) format. Variables must be provided as a list,
    whose ordering maps the variables to their column index
    in the associated coefficient matrix. This modeling
    component allows for fast construction of large linear
    constraint sets as it bypasses Pyomo's expression
    system.

    Parameters
    ----------
    A_data : list
        The values of the CSR format sparse matrix
    A_indices : list
        The column indices of the CSR format sparse matrix
    A_indptr : list
        The row start-stop pointers of the CSR format sparse matrix
    lb : list
        The list of constraint lower bounds
    ub : list
        The list of constraint upper bounds
    x : list
        The list of pyomo variables mapped to their appropriate column

    Example
    -------
    >>> from pyomo.environ import *
    >>> from pyomo.core.base.matrix_constraint import MatrixConstraint
    >>> model = ConcreteModel()
    >>>
    >>> # x_{i} <= x_{i+1}   (for i in {1,2})
    >>> model.v = Var(RangeSet(0,2))
    >>> data    = [1.0, -1.0, 1.0, -1.0]
    >>> indices = [  0,    1,   1,    2]
    >>> indptr  = [0, 2, 4]
    >>> lb      = [None, None]
    >>> ub      = [ 0.0,  0.0]
    >>> x       = [model.v[0], model.v[1], model.v[2]]
    >>> model.c = MatrixConstraint(data, indices, indptr, lb, ub, x)
    """

    def __init__(self, A_data, A_indices, A_indptr, lb, ub, x):

        m = len(lb)
        n = len(x)
        nnz = len(A_data)
        assert len(A_indices) == nnz
        assert len(A_indptr) == m + 1
        assert len(ub) == m
        IndexedConstraint.__init__(self, Any)

        self._A_data = A_data
        self._A_indices = A_indices
        self._A_indptr = A_indptr
        self._lower = lb
        self._upper = ub
        self._x = tuple(x)

    def construct(self, data=None):
        """Construct the expression(s) for this constraint."""
        if is_debug_set(logger):
            logger.debug("Constructing constraint %s"
                         % (self.name))
        if self._constructed:
            return
        self._constructed = True

        ref = weakref.ref(self)
        with PauseGC():
            self._data = tuple(_MatrixConstraintData(i, ref)
                               for i in xrange(len(self._lower)))

    #
    # Override some IndexedComponent methods
    #

    def __getitem__(self, key):
        return self._data[key]

    def __len__(self):
        return self._data.__len__()

    def __iter__(self):
        return iter(i for i in xrange(len(self)))

    #
    # Remove methods that allow modifying this constraint
    #

    def add(self, index, expr):  # pragma:nocover
        raise NotImplementedError

    def __delitem__(self):  # pragma:nocover
        raise NotImplementedError

    def __setitem__(self, key, value):  # pragma:nocover
        raise NotImplementedError
