#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = ("_LinearConstraintData", "MatrixConstraint",
           "compile_block_linear_constraints",)

import time
import logging
import array
from weakref import ref as weakref_ref

from pyomo.common.log import is_debug_set
from pyomo.core.base.set_types import Any
from pyomo.core.base import (SortComponents,
                             Var)
from pyomo.core.base.numvalue import (is_fixed,
                                      value,
                                      ZeroConstant)
from pyomo.core.base.plugin import ModelComponentFactory
from pyomo.core.base.constraint import (Constraint,
                                        IndexedConstraint,
                                        SimpleConstraint,
                                        _ConstraintData)
from pyomo.core.expr.numvalue import native_numeric_types
from pyomo.repn import generate_standard_repn

from six import iteritems, PY3
from six.moves import xrange

if PY3:
    from collections.abc import Mapping as collections_Mapping
else:
    from collections import Mapping as collections_Mapping


logger = logging.getLogger('pyomo.core')

def _label_bytes(x):
    if x < 1e3:
        return str(x)+" B"
    if x < 1e6:
        return str(x / 1.0e3)+" KB"
    if x < 1e9:
        return str(x / 1.0e6)+" MB"
    return str(x / 1.0e9)+" GB"

#
# Compile a Pyomo constructed model in-place, storing the compiled
# sparse constraint object on the model under constraint_name.
#
def compile_block_linear_constraints(parent_block,
                                     constraint_name,
                                     skip_trivial_constraints=False,
                                     single_precision_storage=False,
                                     verbose=False,
                                     descend_into=True):

    if verbose:
        print("")
        print("Compiling linear constraints on block with name: %s"
              % (parent_block.name))

    if not parent_block.is_constructed():
        raise RuntimeError(
            "Attempting to compile block '%s' with unconstructed "
            "component(s)" % (parent_block.name))

    #
    # Linear MatrixConstraint in CSR format
    #
    SparseMat_pRows = []
    SparseMat_jCols = []
    SparseMat_Vals = []
    Ranges = []
    RangeTypes = []

    def _get_bound(exp):
        if exp is None:
            return None
        if is_fixed(exp):
            return value(exp)
        raise ValueError("non-fixed bound: " + str(exp))

    start_time = time.time()
    if verbose:
        print("Sorting active blocks...")

    sortOrder = SortComponents.indices | SortComponents.alphabetical
    all_blocks = [_b for _b in parent_block.block_data_objects(
        active=True,
        sort=sortOrder,
        descend_into=descend_into)]

    stop_time = time.time()
    if verbose:
        print("Time to sort active blocks: %.2f seconds"
              % (stop_time-start_time))

    start_time = time.time()
    if verbose:
        print("Collecting variables on active blocks...")

    #
    # First Pass: assign each variable a deterministic id
    #             (an index in a list)
    #
    VarSymbolToVarObject = []
    for block in all_blocks:
        VarSymbolToVarObject.extend(
            block.component_data_objects(Var,
                                         sort=sortOrder,
                                         descend_into=False))
    VarIDToVarSymbol = \
        dict((id(vardata), index)
             for index, vardata in enumerate(VarSymbolToVarObject))

    stop_time = time.time()
    if verbose:
        print("Time to collect variables on active blocks: %.2f seconds"
              % (stop_time-start_time))

    start_time = time.time()
    if verbose:
        print("Compiling active linear constraints...")

    #
    # Second Pass: collect and remove active linear constraints
    #
    constraint_data_to_remove = []
    empty_constraint_containers_to_remove = []
    constraint_containers_to_remove = []
    constraint_containers_to_check = set()
    referenced_variable_symbols = set()
    nnz = 0
    nrows = 0
    SparseMat_pRows = [0]
    for block in all_blocks:

        if hasattr(block, '_repn'):
            del block._repn

        for constraint in block.component_objects(Constraint,
                                                  active=True,
                                                  sort=sortOrder,
                                                  descend_into=False):

            assert not isinstance(constraint, MatrixConstraint)

            if len(constraint) == 0:

                empty_constraint_containers_to_remove.append((block, constraint))

            else:

                singleton = isinstance(constraint, SimpleConstraint)

                # Note that as we may be removing items from the _data
                # dictionary, we need to make a copy of the items list
                # before iterating:
                for index, constraint_data in list(iteritems(constraint)):

                    if constraint_data.body.__class__ in native_numeric_types or constraint_data.body.polynomial_degree() <= 1:

                        # collect for removal
                        if singleton:
                            constraint_containers_to_remove.append((block, constraint))
                        else:
                            constraint_data_to_remove.append((constraint, index))
                            constraint_containers_to_check.add((block, constraint))

                        repn = generate_standard_repn(constraint_data.body)

                        assert repn.nonlinear_expr is None

                        row_variable_symbols = []
                        row_coefficients = []
                        if len(repn.linear_vars) == 0:
                            if skip_trivial_constraints:
                                continue
                        else:
                            row_variable_symbols = \
                                [VarIDToVarSymbol[id(vardata)]
                                 for vardata in repn.linear_vars]
                            referenced_variable_symbols.update(
                                row_variable_symbols)
                            assert repn.linear_coefs is not None
                            row_coefficients = repn.linear_coefs

                        SparseMat_pRows.append(SparseMat_pRows[-1] + \
                                               len(row_variable_symbols))
                        SparseMat_jCols.extend(row_variable_symbols)
                        SparseMat_Vals.extend(row_coefficients)

                        nnz += len(row_variable_symbols)
                        nrows += 1

                        L = _get_bound(constraint_data.lower)
                        U = _get_bound(constraint_data.upper)
                        constant = value(repn.constant)

                        Ranges.append(L - constant if (L is not None) else 0)
                        Ranges.append(U - constant if (U is not None) else 0)
                        if (L is not None) and \
                           (U is not None) and \
                           (not constraint_data.equality):
                            RangeTypes.append(MatrixConstraint.LowerBound |
                                              MatrixConstraint.UpperBound)
                        elif constraint_data.equality:
                            RangeTypes.append(MatrixConstraint.Equality)
                        elif L is not None:
                            assert U is None
                            RangeTypes.append(MatrixConstraint.LowerBound)
                        else:
                            assert U is not None
                            RangeTypes.append(MatrixConstraint.UpperBound)

                        # Start freeing up memory
                        constraint[index] = Constraint.Skip

    ncols = len(referenced_variable_symbols)

    stop_time = time.time()
    if verbose:
        print("Time to compile active linear constraints: %.2f seconds"
              % (stop_time-start_time))

    start_time = time.time()
    if verbose:
        print("Removing compiled constraint objects...")

    #
    # Remove compiled constraints
    #
    constraints_removed = 0
    constraint_containers_removed = 0
    for block, constraint in empty_constraint_containers_to_remove:
        block.del_component(constraint)
        constraint_containers_removed += 1
    for constraint, index in constraint_data_to_remove:
        # Note that this del is not needed: assigning Constraint.Skip
        # above removes the _ConstraintData from the _data dict.
        #del constraint[index]
        constraints_removed += 1
    for block, constraint in constraint_containers_to_remove:
        block.del_component(constraint)
        constraints_removed += 1
        constraint_containers_removed += 1
    for block, constraint in constraint_containers_to_check:
        if len(constraint) == 0:
            block.del_component(constraint)
            constraint_containers_removed += 1

    stop_time = time.time()
    if verbose:
        print("Eliminated %s constraints and %s Constraint container objects"
              % (constraints_removed, constraint_containers_removed))
        print("Time to remove compiled constraint objects: %.2f seconds"
              % (stop_time-start_time))

    start_time = time.time()
    if verbose:
        print("Assigning variable column indices...")

    #
    # Assign a column index to the set of referenced variables
    #
    ColumnIndexToVarSymbol = sorted(referenced_variable_symbols)
    VarSymbolToColumnIndex = dict((symbol, column)
                                  for column, symbol in enumerate(ColumnIndexToVarSymbol))
    SparseMat_jCols = [VarSymbolToColumnIndex[symbol] for symbol in SparseMat_jCols]
    del VarSymbolToColumnIndex
    ColumnIndexToVarObject = [VarSymbolToVarObject[var_symbol]
                              for var_symbol in ColumnIndexToVarSymbol]

    stop_time = time.time()
    if verbose:
        print("Time to assign variable column indices: %.2f seconds"
              % (stop_time-start_time))

    start_time = time.time()
    if verbose:
        print("Converting compiled constraint data to array storage...")
        print("  - Using %s precision for numeric values"
              % ('single' if single_precision_storage else 'double'))

    #
    # Convert to array storage
    #

    number_storage = 'f' if single_precision_storage else 'd'
    SparseMat_pRows = array.array('L', SparseMat_pRows)
    SparseMat_jCols = array.array('L', SparseMat_jCols)
    SparseMat_Vals = array.array(number_storage, SparseMat_Vals)
    Ranges = array.array(number_storage, Ranges)
    RangeTypes = array.array('B', RangeTypes)

    stop_time = time.time()
    if verbose:
        storage_bytes = \
            SparseMat_pRows.buffer_info()[1] * SparseMat_pRows.itemsize + \
            SparseMat_jCols.buffer_info()[1] * SparseMat_jCols.itemsize + \
            SparseMat_Vals.buffer_info()[1] * SparseMat_Vals.itemsize + \
            Ranges.buffer_info()[1] * Ranges.itemsize + \
            RangeTypes.buffer_info()[1] * RangeTypes.itemsize
        print("Sparse Matrix Dimension:")
        print("  - Rows: "+str(nrows))
        print("  - Cols: "+str(ncols))
        print("  - Nonzeros: "+str(nnz))
        print("Compiled Data Storage: "+str(_label_bytes(storage_bytes)))
        print("Time to convert compiled constraint data to "
              "array storage: %.2f seconds" % (stop_time-start_time))

    parent_block.add_component(constraint_name,
                               MatrixConstraint(nrows, ncols, nnz,
                                                SparseMat_pRows,
                                                SparseMat_jCols,
                                                SparseMat_Vals,
                                                Ranges,
                                                RangeTypes,
                                                ColumnIndexToVarObject))

#class _LinearConstraintData(_ConstraintData,LinearCanonicalRepn):
#
# This change breaks this class, but it's unclear whether this
# is being used...
#
class _LinearConstraintData(_ConstraintData):
    """
    This class defines the data for a single linear constraint
        in canonical form.

    Constructor arguments:
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
        variables       A tuple of variables comprising the body
                            of this constraint
        coefficients    A tuple of coefficients matching the order
                            of variables that comprise the body of
                            this constraint
        constant        A number representing the aggregation of any
                            constant/fixed items found in the body of
                            this constraint

    Private class attributes:
        _component      The objective component.
        _active         A boolean that indicates whether this data is active
    """

    __slots__ = ()

    def __init__(self, index, component=None):
        #
        # These lines represent in-lining of the
        # following constructors:
        #   - _ConstraintData,
        #   - ActiveComponentData
        #   - ComponentData
        self._component = weakref_ref(component) if (component is not None) \
                          else None
        self._active = True

class _LinearMatrixConstraintData(_LinearConstraintData):
    """
    This class defines the data for a single linear constraint
        derived from a canonical form Ax=b constraint.

    Constructor arguments:
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
        variables       A tuple of variables comprising the body
                            of this constraint
        coefficients    A tuple of coefficients matching the order
                            of variables that comprise the body of
                            this constraint
        constant        A number representing the aggregate of any
                            constants found in the body of this
                            constraint

    Private class attributes:
        _component      The objective component.
        _active         A boolean that indicates whether this data is active
    """

    __slots__ = ('_index')

    def __init__(self, index, component=None):
        #
        # These lines represent in-lining of the
        # following constructors:
        #   - _LinearConstraintData
        #   - _ConstraintData,
        #   - ActiveComponentData
        #   - ComponentData
        self._component = weakref_ref(component) if (component is not None) \
                          else None
        self._active = True

        # row index into the sparse matrix stored on the parent
        assert index >= 0
        self._index = index

    def __getstate__(self):
        """
        This method must be defined because this class uses slots.
        """
        result = super(_LinearMatrixConstraintData, self).__getstate__()
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
        """
        Compute the value of the body of this constraint.
        """
        comp = self.parent_component()
        index = self.index()
        prows = comp._prows
        jcols = comp._jcols
        varmap = comp._varmap
        vals = comp._vals
        try:
            return sum(varmap[jcols[p]]() * vals[p]
                       for p in xrange(prows[index],
                                       prows[index+1]))
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
        """
        Returns the value of L-f(x) for constraints of the form:
            L <= f(x) (<= U)
            (U >=) f(x) >= L
        """
        raise self.lower - self()

    def uslack(self):
        """
        Returns the value of U-f(x) for constraints of the form:
            (L <=) f(x) <= U
            U >= f(x) (>= L)
        """
        self.upper - self()

    #
    # Override some default implementations on ComponentData
    #

    def index(self):
        return self._index

    #
    # Abstract Interface (LinearCanonicalRepn)
    #

    @property
    def variables(self):
        """A tuple of variables comprising the constraint body."""
        comp = self.parent_component()
        prows = comp._prows
        jcols = comp._jcols
        varmap = comp._varmap
        if prows[self._index] == prows[self._index+1]:
            return()
        variables = tuple(varmap[jcols[p]]
                          for p in xrange(prows[self._index],
                                          prows[self._index+1])
                          if not varmap[jcols[p]].fixed)

        return variables

    @property
    def coefficients(self):
        """A tuple of coefficients associated with the variables."""
        comp = self.parent_component()
        prows = comp._prows
        jcols = comp._jcols
        vals = comp._vals
        varmap = comp._varmap
        if prows[self._index] == prows[self._index+1]:
            return ()
        coefs = tuple(vals[p] for p in xrange(prows[self._index],
                                              prows[self._index+1])
                      if not varmap[jcols[p]].fixed)

        return coefs

    # for backwards compatibility
    linear=coefficients

    @property
    def constant(self):
        """The constant value associated with the constraint body."""
        comp = self.parent_component()
        prows = comp._prows
        jcols = comp._jcols
        vals = comp._vals
        varmap = comp._varmap
        if prows[self._index] == prows[self._index+1]:
            return 0
        terms = tuple(vals[p] * varmap[jcols[p]]()
                      for p in xrange(prows[self._index],
                                      prows[self._index+1])
                      if varmap[jcols[p]].fixed)

        return sum(terms)

    #
    # Abstract Interface (_ConstraintData)
    #

    @property
    def body(self):
        """Access the body of a constraint expression."""
        comp = self.parent_component()
        index = self.index()
        prows = comp._prows
        jcols = comp._jcols
        varmap = comp._varmap
        vals = comp._vals
        if prows[self._index] == prows[self._index+1]:
            return ZeroConstant
        return sum(varmap[jcols[p]] * vals[p]
                   for p in xrange(prows[index],
                                   prows[index+1]))

    @property
    def lower(self):
        """Access the lower bound of a constraint expression."""
        comp = self.parent_component()
        index = self.index()
        if (comp._range_types[index] & MatrixConstraint.LowerBound):
            return comp._ranges[2 * index]
        return None

    @property
    def upper(self):
        """Access the upper bound of a constraint expression."""
        comp = self.parent_component()
        index = self.index()
        if (comp._range_types[index] & MatrixConstraint.UpperBound):
            return comp._ranges[(2 * index) + 1]
        return None

    @property
    def equality(self):
        """A boolean indicating whether this is an equality constraint."""
        return (self.parent_component()._range_types[self.index()] & \
                MatrixConstraint.Equality) == MatrixConstraint.Equality

    @property
    def strict_lower(self):
        """A boolean indicating whether this constraint has a strict lower bound."""
        return (self.parent_component()._range_types[self.index()] & \
                MatrixConstraint.StrictLowerBound) == \
                MatrixConstraint.StrictLowerBound

    @property
    def strict_upper(self):
        """A boolean indicating whether this constraint has a strict upper bound."""
        return (self.parent_component()._range_types[self.index()] & \
                MatrixConstraint.StrictUpperBound) == \
                MatrixConstraint.StrictUpperBound

    def set_value(self, expr):
        """Set the expression on this constraint."""
        raise NotImplementedError("MatrixConstraint row elements can not "
                                  "be updated")

@ModelComponentFactory.register(
                   "A set of constraint expressions in Ax=b form.")
class MatrixConstraint(collections_Mapping,
                       IndexedConstraint):

    #
    # Bound types
    # (make sure the maximum value here
    #  will fit in an unsigned char)
    #
    StrictUpperBound = 0b00011
    UpperBound =       0b00010
    Equality =         0b01110
    LowerBound =       0b01000
    StrictLowerBound = 0b11000
    NoBound =          0b00000

    def __init__(self,
                 nrows,
                 ncols,
                 nnz,
                 prows,
                 jcols,
                 vals,
                 ranges,
                 range_types,
                 varmap):

        assert len(prows) == nrows + 1
        assert len(jcols) == nnz
        assert len(vals) == nnz
        assert len(ranges) == 2 * nrows
        assert len(range_types) == nrows
        assert len(varmap) == ncols

        IndexedConstraint.__init__(self,
                                   Any)

        self._prows = prows
        self._jcols = jcols
        self._vals = vals
        self._ranges = ranges
        self._range_types = range_types
        self._varmap = varmap

    def construct(self, data=None):
        """
        Construct the expression(s) for this constraint.
        """
        if is_debug_set(logger):
            logger.debug("Constructing constraint %s"
                         % (self.name))
        if self._constructed:
            return
        self._constructed=True

        _init = _LinearMatrixConstraintData
        self._data = tuple(_init(i, component=self)
                           for i in xrange(len(self._range_types)))

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

    def add(self, index, expr):
        raise NotImplementedError

    def __delitem__(self):
        raise NotImplementedError

