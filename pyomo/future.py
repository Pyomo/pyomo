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

__doc__ = """
Preview capabilities through ``pyomo.__future__``
=================================================

This module provides a uniform interface for gaining access to future
("preview") capabilities that are either slightly incompatible with the
current official offering, or are still under development with the
intent to replace the current offering.

Currently supported ``__future__`` offerings include:

.. autosummary::

   solver_factory

"""

from typing import Any, Literal, overload

import pyomo.contrib.appsi.base as _appsi
import pyomo.contrib.solver.common.factory as _contrib
import pyomo.environ as _environ
import pyomo.opt.base.solvers as _solvers

_SolverFactoryClassV1 = _solvers.SolverFactoryClass
_SolverFactoryClassV2 = _appsi.SolverFactoryClass
_SolverFactoryClassV3 = _contrib.SolverFactoryClass

solver_factory_v1: _SolverFactoryClassV1 = _solvers.LegacySolverFactory
solver_factory_v2: _SolverFactoryClassV2 = _appsi.SolverFactory
solver_factory_v3: _SolverFactoryClassV3 = _contrib.SolverFactory

_versions = {1: solver_factory_v1, 2: solver_factory_v2, 3: solver_factory_v3}


def _get_environ_version() -> int:
    # Go look and see what it was initialized to in pyomo.environ
    for ver, cls in _versions.items():
        if cls._cls is _environ.SolverFactory._cls:
            return ver
    # If initialized correctly, never reached
    raise NotImplementedError


_active_version = _get_environ_version()


@overload
def solver_factory(version: None = None) -> int: ...
@overload
def solver_factory(version: Literal[1]) -> _SolverFactoryClassV1: ...
@overload
def solver_factory(version: Literal[2]) -> _SolverFactoryClassV2: ...
@overload
def solver_factory(version: Literal[3]) -> _SolverFactoryClassV3: ...
@overload
def solver_factory(version: int) -> Any: ...


def solver_factory(version: int | None = None):
    """Get (or set) the active implementation of the SolverFactory

    This allows users to query / set the current implementation of the
    SolverFactory that should be used throughout Pyomo.  Valid options are:

    - ``1``: the original Pyomo SolverFactory
    - ``2``: the SolverFactory from APPSI
    - ``3``: the SolverFactory from pyomo.contrib.solver

    The current active version can be obtained by calling the method
    with no arguments

    .. doctest::

        >>> from pyomo.__future__ import solver_factory
        >>> solver_factory()
        1

    The active factory can be set either by passing the appropriate
    version to this function:

    .. doctest::

        >>> solver_factory(3)
        <pyomo.contrib.solver.common.factory.SolverFactoryClass object ...>

    or by importing the "special" name:

    .. doctest::

        >>> from pyomo.__future__ import solver_factory_v3

    .. doctest::
       :hide:

       >>> from pyomo.__future__ import solver_factory_v1

    """
    global _active_version

    if version is None:
        return _active_version

    # Update the current SolverFactory to be a shim around (shallow copy
    # of) the new active factory
    selected_factory = _versions.get(version, None)
    if selected_factory is not None:
        _active_version = version
        for attr in ("_description", "_cls", "_doc"):
            setattr(_environ.SolverFactory, attr, getattr(selected_factory, attr))
    else:
        raise ValueError(
            "Invalid value for target solver factory version; expected {1, 2, 3}, "
            f"received {version}"
        )
    return selected_factory
