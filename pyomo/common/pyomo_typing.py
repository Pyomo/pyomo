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

import typing

_overloads = {}


def _get_fullqual_name(func: typing.Callable) -> str:
    return f"{func.__module__}.{func.__qualname__}"


def overload(func: typing.Callable):
    """Wrap typing.overload that remembers the overloaded signatures

    This provides a custom implementation of typing.overload that
    remembers the overloaded signatures so that they are available for
    runtime inspection.

    """
    _overloads.setdefault(_get_fullqual_name(func), []).append(func)
    return typing.overload(func)


def get_overloads_for(func: typing.Callable):
    return _overloads.get(_get_fullqual_name(func), [])
