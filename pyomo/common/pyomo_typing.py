# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import sys
import typing

_overloads = {}


def _get_fullqual_name(func: typing.Callable) -> str:
    return f"{func.__module__}.{func.__qualname__}"


if sys.version_info[:2] <= (3, 10) and not typing.TYPE_CHECKING:

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

else:
    from typing import get_overloads as get_overloads_for
    from typing import overload as overload


_P = typing.ParamSpec('_P')
_R = typing.TypeVar('_R')


def copy_signature(
    source: typing.Callable[_P, _R],
) -> typing.Callable[[typing.Callable[..., typing.Any]], typing.Callable[_P, _R]]:
    """Declare that the decorated function has the signature of `source`

    This decorator returns the decorated function unchanged.  Its only
    effect is on static type checkers, which will check calls to the
    decorated function against the signature of `source`.  This is
    intended for methods that accept ``*args, **kwds`` and forward them
    to `source` (e.g., a scalar component's ``__init__`` forwarding to
    the base component's ``__init__``), so that their arguments are
    checked without repeating the signature.

    """

    def decorator(func: typing.Any) -> typing.Any:
        return func

    return decorator
