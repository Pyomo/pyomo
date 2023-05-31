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

import enum
import itertools
import logging
import sys

from pyomo.common.collections import Sequence, ComponentMap
from pyomo.common.deprecation import deprecation_warning
from pyomo.common.errors import DeveloperError, InvalidValueError
from pyomo.core.base import (
    Var,
    Param,
    Expression,
    Objective,
    Block,
    Constraint,
    Suffix,
    SortComponents,
)
from pyomo.core.base.component import ActiveComponent
from pyomo.core.expr.numvalue import native_numeric_types, is_fixed, value
import pyomo.core.kernel as kernel

logger = logging.getLogger('pyomo.core')

valid_expr_ctypes_minlp = {Var, Param, Expression, Objective}
valid_active_ctypes_minlp = {Block, Constraint, Objective, Suffix}

HALT_ON_EVALUATION_ERROR = False
nan = float('nan')


class ExprType(enum.IntEnum):
    CONSTANT = 0
    MONOMIAL = 10
    LINEAR = 20
    QUADRATIC = 30
    GENERAL = 40


_FileDeterminism_deprecation = {
    1: 20,
    2: 30,
    'DEPRECATED_KEYS': 20,
    'DEPRECATED_KEYS_AND_NAMES': 30,
}


class FileDeterminism(enum.IntEnum):
    NONE = 0
    # DEPRECATED_KEYS = 1
    # DEPRECATED_KEYS_AND_NAMES = 2
    ORDERED = 10
    SORT_INDICES = 20
    SORT_SYMBOLS = 30

    # We will define __str__ and __format__ so that behavior in python
    # 3.11 is consistent with 3.7 - 3.10.

    def __str__(self):
        return enum.Enum.__str__(self)

    def __format__(self, spec):
        # This cannot just call Enum.__format__ because that returns the
        # numeric value in Python 3.7
        return str(self).__format__(spec)

    @classmethod
    def _missing_(cls, value):
        # This is not a perfect deprecation path, as the old attributes
        # are no longer valid.  However, as the previous implementation
        # was a pure int and not an Enum, this is sufficient for our
        # needs.
        if value in _FileDeterminism_deprecation:
            new = FileDeterminism(_FileDeterminism_deprecation[value])
            deprecation_warning(
                f'FileDeterminism({value}) is deprecated.  '
                f'Please use {str(new)} ({int(new)})',
                version='6.5.0',
            )
            return new
        return super()._missing_(value)


class InvalidNumber(object):
    def __init__(self, value):
        self.value = value

    def duplicate(self, new_value):
        return InvalidNumber(new_value)

    def merge(self, other, new_value):
        return InvalidNumber(new_value)

    def __eq__(self, other):
        if other.__class__ is InvalidNumber:
            return self.value == other.value
        else:
            return self.value == other

    def __lt__(self, other):
        if other.__class__ is InvalidNumber:
            return self.value < other.value
        else:
            return self.value < other

    def __gt__(self, other):
        if other.__class__ is InvalidNumber:
            return self.value > other.value
        else:
            return self.value > other

    def __le__(self, other):
        if other.__class__ is InvalidNumber:
            return self.value <= other.value
        else:
            return self.value <= other

    def __ge__(self, other):
        if other.__class__ is InvalidNumber:
            return self.value >= other.value
        else:
            return self.value >= other

    def __str__(self):
        return f'InvalidNumber({self.value})'

    def __repr__(self):
        # FIXME: We want to move to where converting InvalidNumber to
        # string (with either repr() or f"") should raise a
        # InvalidValueError.  However, at the moment, this breaks some
        # tests in PyROS.
        return repr(self.value)
        # raise InvalidValueError(f'Cannot emit {str(self)} in compiled representation')

    def __format__(self, format_spec):
        # FIXME: We want to move to where converting InvalidNumber to
        # string (with either repr() or f"") should raise a
        # InvalidValueError.  However, at the moment, this breaks some
        # tests in PyROS.
        return self.value.__format__(format_spec)
        # raise InvalidValueError(f'Cannot emit {str(self)} in compiled representation')

    def __neg__(self):
        return self.duplicate(-self.value)

    def __abs__(self):
        return self.duplicate(abs(self.value))

    def __add__(self, other):
        if other.__class__ is InvalidNumber:
            return self.merge(other, self.value + other.value)
        else:
            return self.duplicate(self.value + other)

    def __sub__(self, other):
        if other.__class__ is InvalidNumber:
            return self.merge(other, self.value - other.value)
        else:
            return self.duplicate(self.value - other)

    def __mul__(self, other):
        if other.__class__ is InvalidNumber:
            return self.merge(other, self.value * other.value)
        else:
            return self.duplicate(self.value * other)

    def __truediv__(self, other):
        if other.__class__ is InvalidNumber:
            return self.merge(other, self.value / other.value)
        else:
            return self.duplicate(self.value / other)

    def __pow__(self, other):
        if other.__class__ is InvalidNumber:
            return self.merge(other, self.value**other.value)
        else:
            return self.duplicate(self.value**other)

    def __radd__(self, other):
        return self.duplicate(other + self.value)

    def __rsub__(self, other):
        return self.duplicate(other - self.value)

    def __rmul__(self, other):
        return self.duplicate(other * self.value)

    def __rtruediv__(self, other):
        return self.duplicate(other / self.value)

    def __rpow__(self, other):
        return self.duplicate(other**self.value)


def apply_node_operation(node, args):
    try:
        ans = node._apply_operation(args)
        if ans != ans and ans.__class__ is not InvalidNumber:
            ans = InvalidNumber(ans)
        return ans
    except:
        logger.warning(
            "Exception encountered evaluating expression "
            "'%s(%s)'\n\tmessage: %s\n\texpression: %s"
            % (node.name, ", ".join(map(str, args)), str(sys.exc_info()[1]), node)
        )
        if HALT_ON_EVALUATION_ERROR:
            raise
        return InvalidNumber(nan)


def complex_number_error(value, visitor, expr, node=""):
    msg = f'Pyomo {visitor.__class__.__name__} does not support complex numbers'
    logger.warning(
        ' '.join(filter(None, ("Complex number returned from expression", node)))
        + f"\n\tmessage: {msg}\n\texpression: {expr}"
    )
    if HALT_ON_EVALUATION_ERROR:
        raise InvalidValueError(
            f'Pyomo {visitor.__class__.__name__} does not support complex numbers'
        )
    return InvalidNumber(value)


def categorize_valid_components(
    model, active=True, sort=None, valid=set(), targets=set()
):
    """Walk model and check for valid component types

    This routine will walk the model and check all component types.
    Components types in the `valid` set are ignored, blocks with
    components in the `targets` set are collected, and all other
    component types are added to a dictionary of `unrecognized`
    components.

    A Component type may not appear in both `valid` and `targets` sets.

    Parameters
    ----------
    model: _BlockData
        The model tree to walk

    active: True or None
        If True, only unrecognized active components are returned in the
        `uncategorized` dictionary.  Also, if True, only active Blocks
        are descended into.

    sort: bool or SortComponents
        The sorting flag to pass to the block walkers

    valid: Set[type]
        The set of "valid" component types.  These are ignored by the
        categorizer.

    targets: Set[type]
        The set of component types to "collect".  Blocks with components
        in the `targets` set will be returned in the `component_map`

    Returns
    -------
    component_map: Dict[type, List[_BlockData]]
        A dict mapping component type to a list of block data
        objects that contain declared component of that type.

    unrecognized: Dict[type, List[ComponentData]]
        A dict mapping unrecognized component types to a (non-empty)
        list of component data objects found on the model.

    """
    assert active in (True, None)
    # Note: we assume every target component is valid but that we expect
    # there to be far mode valid components than target components.
    # Generate an error if a target is in the valid set (because the
    # valid set will preclude recording the block in the component_map)
    if any(ctype in valid for ctype in targets):
        ctypes = list(filter(valid.__contains__, targets))
        raise DeveloperError(
            f"categorize_valid_components: Cannot have component type {ctypes} in "
            "both the `valid` and `targets` sets"
        )
    unrecognized = {}
    component_map = {k: [] for k in targets}
    for block in model.block_data_objects(active=active, descend_into=True, sort=sort):
        local_ctypes = block.collect_ctypes(active=None, descend_into=False)
        for ctype in local_ctypes:
            if ctype in kernel.base._kernel_ctype_backmap:
                ctype = kernel.base._kernel_ctype_backmap[ctype]
            if ctype in valid:
                continue
            if ctype in targets:
                component_map[ctype].append(block)
                continue
            # TODO: we should rethink the definition of "active" for
            # Components that are not subclasses of ActiveComponent
            if (
                active
                and not issubclass(ctype, ActiveComponent)
                and not issubclass(ctype, kernel.base.ICategorizedObject)
            ):
                continue
            if ctype not in unrecognized:
                unrecognized[ctype] = []
            unrecognized[ctype].extend(
                block.component_data_objects(
                    ctype=ctype,
                    active=active,
                    descend_into=False,
                    sort=SortComponents.unsorted,
                )
            )
    return component_map, {k: v for k, v in unrecognized.items() if v}


def FileDeterminism_to_SortComponents(file_determinism):
    sorter = SortComponents.unsorted
    if file_determinism >= FileDeterminism.SORT_INDICES:
        sorter = sorter | SortComponents.indices
        if file_determinism >= FileDeterminism.SORT_SYMBOLS:
            sorter = sorter | SortComponents.alphabetical
    return sorter


def initialize_var_map_from_column_order(model, config, var_map):
    column_order = config.column_order
    sorter = FileDeterminism_to_SortComponents(config.file_determinism)

    if column_order is None or column_order.__class__ is bool:
        if not column_order:
            column_order = None
    elif isinstance(column_order, ComponentMap):
        # The column order has historically has supported a ComponentMap of
        # component to position in addition to the simple list of
        # components.  Convert it to the simple list
        column_order = sorted(column_order, key=column_order.__getitem__)

    if column_order == True:
        column_order = model.component_data_objects(Var, descend_into=True, sort=sorter)
    elif config.file_determinism > FileDeterminism.ORDERED:
        # We will pre-gather the variables so that their order
        # matches the file_determinism flag.  This is a little
        # cumbersome, but is implemented this way for consistency
        # with the original NL writer.
        var_objs = model.component_data_objects(Var, descend_into=True, sort=sorter)
        if column_order is None:
            column_order = var_objs
        else:
            column_order = itertools.chain(column_order, var_objs)

    if column_order is not None:
        # Note that Vars that appear twice (e.g., through a
        # Reference) will be sorted with the FIRST occurrence.
        for var in column_order:
            if var.is_indexed():
                for _v in var.values(sorter):
                    if not _v.fixed:
                        var_map[id(_v)] = _v
            elif not var.fixed:
                var_map[id(var)] = var
    return var_map


def ordered_active_constraints(model, config):
    sorter = FileDeterminism_to_SortComponents(config.file_determinism)
    constraints = model.component_data_objects(Constraint, active=True, sort=sorter)

    row_order = config.row_order
    if row_order is None or row_order.__class__ is bool:
        return constraints
    elif isinstance(row_order, ComponentMap):
        # The row order has historically also supported a ComponentMap of
        # component to position in addition to the simple list of
        # components.  Convert it to the simple list
        row_order = sorted(row_order, key=row_order.__getitem__)

    row_map = {}
    for con in row_order:
        if con.is_indexed():
            for c in con.values(sorter):
                row_map[id(c)] = c
        else:
            row_map[id(con)] = con
    if not row_map:
        return constraints
    # map the implicit dict ordering to an explicit 0..n ordering
    row_map = {_id: i for i, _id in enumerate(row_map)}
    # sorted() is stable (per Python docs), so we can let all
    # unspecified rows have a row number one bigger than the
    # number of rows specified by the user ordering.
    _n = len(row_map)
    _row_getter = row_map.get
    return sorted(constraints, key=lambda x: _row_getter(id(x), _n))


# Copied from cpxlp.py:
# Keven Hunter made a nice point about using %.16g in his attachment
# to ticket #4319. I am adjusting this to %.17g as this mocks the
# behavior of using %r (i.e., float('%r'%<number>) == <number>) with
# the added benefit of outputting (+/-). The only case where this
# fails to mock the behavior of %r is for large (long) integers (L),
# which is a rare case to run into and is probably indicative of
# other issues with the model.
# *** NOTE ***: If you use 'r' or 's' here, it will break code that
#               relies on using '%+' before the formatting character
#               and you will need to go add extra logic to output
#               the number's sign.
_ftoa_precision_str = '%.17g'


def ftoa(val, parenthesize_negative_values=False):
    if val is None:
        return val
    #
    # Basic checking, including conversion of *fixed* Pyomo types to floats
    if type(val) in native_numeric_types:
        _val = val
    else:
        if is_fixed(val):
            _val = value(val)
        else:
            raise ValueError(
                "Converting non-fixed bound or value to string: %s" % (val,)
            )
    #
    # Convert to string
    a = _ftoa_precision_str % _val
    #
    # Remove unnecessary least significant digits.  While not strictly
    # necessary, this helps keep the emitted string consistent between
    # python versions by simplifying things like "1.0000000000001" to
    # "1".
    i = len(a) - 1
    if i:
        try:
            while float(a[:i]) == _val:
                i -= 1
        except:
            pass
    i += 1
    #
    # It is important to issue a warning if the conversion loses
    # precision (as the emitted model is not exactly what the user
    # specified)
    if i == len(a) and float(a) != _val:
        logger.warning("Converting %s to string resulted in loss of precision" % val)
    #
    if parenthesize_negative_values and a[0] == '-':
        return '(' + a[:i] + ')'
    else:
        return a[:i]
