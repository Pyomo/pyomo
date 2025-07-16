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

from pyomo.common.collections import ComponentSet
from typing import Sequence


class ComponentDataSet:
    """ComponentDataSet(ctype)
    Domain validation class that accepts singleton or iterable arguments and
    compiles them into a ComponentSet, verifying that they are all ComponentDatas
    of type 'ctype.'

    Parameters
    ----------
        ctype: Either a single component type or an iterable of component types

    Raises
    ------
        ValueError if all of the arguments are not of a type in 'ctype'
    """

    def __init__(self, ctype):
        if isinstance(ctype, Sequence):
            self._ctypes = set(ctype)
        else:
            self._ctypes = set([ctype])

    def __call__(self, x):
        return ComponentSet(self._process(x))

    def _process(self, x):
        if hasattr(x, 'ctype'):
            if x.ctype not in self._ctypes:
                # Ordering for determinism
                _names = ', '.join(sorted([ct.__name__ for ct in self._ctypes]))
                raise ValueError(
                    f"Expected component or iterable of one "
                    f"of the following ctypes: "
                    f"{_names}.\n\tReceived {type(x)}"
                )
            if x.is_indexed():
                yield from x.values()
            else:
                yield x
        elif hasattr(x, '__iter__'):
            for y in x:
                yield from self._process(y)
        else:
            # Ordering for determinism
            _names = ', '.join(sorted([ct.__name__ for ct in self._ctypes]))
            raise ValueError(
                f"Expected component or iterable of one "
                f"of the following ctypes: "
                f"{_names}.\n\tReceived {type(x)}"
            )

    def domain_name(self):
        # Ordering for determinism
        _ctypes = sorted([ct.__name__ for ct in self._ctypes])
        _names = ', '.join(_ctypes)
        if len(self._ctypes) > 1:
            _names = '[' + _names + ']'
        return f"ComponentDataSet({_names})"
