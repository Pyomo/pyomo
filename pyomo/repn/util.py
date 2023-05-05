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
import logging
import sys

from pyomo.common.collections import Mapping
from pyomo.common.deprecation import deprecation_warning
from pyomo.common.errors import DeveloperError
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


def apply_node_operation(node, args):
    try:
        tmp = node._apply_operation(args)
        if tmp.__class__ is complex:
            raise ValueError('Pyomo does not support complex numbers')
        return tmp
    except:
        logger.warning(
            "Exception encountered evaluating expression "
            "'%s(%s)'\n\tmessage: %s\n\texpression: %s"
            % (node.name, ", ".join(map(str, args)), str(sys.exc_info()[1]), node)
        )
        if HALT_ON_EVALUATION_ERROR:
            raise
        return nan


def categorize_valid_components(
    model, active=True, sort=None, valid=set(), targets=set()
):
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

    if isinstance(column_order, Mapping):
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
        if column_order is None:
            column_order = []
        column_order.extend(
            model.component_data_objects(Var, descend_into=True, sort=sorter)
        )
    if column_order:
        # Note that Vars that appear twice (e.g., through a
        # Reference) will be sorted with the FIRST occurrence.
        for var in column_order:
            if var.is_indexed():
                for _v in var.values():
                    if not _v.fixed:
                        var_map[id(_v)] = _v
            elif not var.fixed:
                var_map[id(var)] = var
    return var_map


def ordered_active_constraints(model, config):
    sorter = FileDeterminism_to_SortComponents(config.file_determinism)
    constraints = model.component_data_objects(Constraint, active=True, sort=sorter)

    row_order = config.row_order
    if isinstance(row_order, Mapping):
        # The row order has historically also supported a ComponentMap of
        # component to position in addition to the simple list of
        # components.  Convert it to the simple list
        row_order = sorted(row_order, key=row_order.__getitem__)

    if row_order:
        row_map = {}
        for con in row_order:
            if con.is_indexed():
                for c in con.values():
                    row_map[id(c)] = c
            else:
                row_map[id(con)] = con
        # map the implicit dict ordering to an explicit 0..n ordering
        row_map = {_id: i for i, _id in enumerate(row_map)}
        # sorted() is stable (per Python docs), so we can let all
        # unspecified rows have a row number one bigger than the
        # number of rows specified by the user ordering.
        _n = len(row_map)
        _row_getter = row_map.get
        constraints = sorted(constraints, key=lambda x: _row_getter(id(x), _n))
    return constraints


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
    i = len(a)
    try:
        while i > 1:
            if float(a[: i - 1]) == _val:
                i -= 1
            else:
                break
    except:
        pass
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
