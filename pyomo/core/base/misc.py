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

import logging
import sys

from pyomo.common.deprecation import relocated_module_attribute

logger = logging.getLogger('pyomo.core')

relocated_module_attribute(
    'tabular_writer', 'pyomo.common.formatting.tabular_writer', version='6.1'
)
relocated_module_attribute(
    'sorted_robust', 'pyomo.common.sorting.sorted_robust', version='6.1'
)


def display(obj, ostream=None):
    """Display data in a Pyomo object"""
    if ostream is None:
        ostream = sys.stdout
    try:
        display_fcn = obj.display
    except AttributeError:
        raise TypeError(
            "Error trying to display values for object of type %s:\n"
            "\tObject does not support the 'display()' method" % (type(obj),)
        )
    try:
        display_fcn(ostream=ostream)
    except Exception:
        err = sys.exc_info()[1]
        logger.error(
            "Error trying to display values for object of type %s:\n\t%s"
            % (type(obj), err)
        )
        raise


def create_name(name, ndx):
    """Create a canonical name for a component using the given index"""
    if ndx is None:
        return name
    if type(ndx) is tuple:
        tmp = str(ndx).replace(', ', ',')
        return name + "[" + tmp[1:-1] + "]"
    return name + "[" + str(ndx) + "]"


def apply_indexed_rule(obj, rule, model, index, options=None):
    try:
        if options is None:
            if index.__class__ is tuple:
                return rule(model, *index)
            elif index is None and not obj.is_indexed():
                return rule(model)
            else:
                return rule(model, index)
        else:
            if index.__class__ is tuple:
                return rule(model, *index, **options)
            elif index is None and not obj.is_indexed():
                return rule(model, **options)
            else:
                return rule(model, index, **options)
    except TypeError:
        try:
            if options is None:
                return rule(model)
            else:
                return rule(model, **options)
        except:
            # Nothing appears to have matched... re-trigger the original
            # TypeError
            if options is None:
                if index.__class__ is tuple:
                    return rule(model, *index)
                elif index is None and not obj.is_indexed():
                    return rule(model)
                else:
                    return rule(model, index)
            else:
                if index.__class__ is tuple:
                    return rule(model, *index, **options)
                elif index is None and not obj.is_indexed():
                    return rule(model, **options)
                else:
                    return rule(model, index, **options)


def apply_parameterized_indexed_rule(obj, rule, model, param, index):
    if index.__class__ is tuple:
        return rule(model, param, *index)
    if index is None:
        return rule(model, param)
    return rule(model, param, index)
