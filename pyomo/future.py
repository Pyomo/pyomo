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

from typing import TYPE_CHECKING, Any, Literal, overload

import pyomo.environ as _environ

if TYPE_CHECKING:
    import pyomo.contrib.appsi.base as _appsi
    import pyomo.contrib.solver.common.factory as _contrib
    import pyomo.opt.base.solvers as _solvers

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

solver_factory_v1: "_solvers.SolverFactoryClass"
solver_factory_v2: "_appsi.SolverFactoryClass"
solver_factory_v3: "_contrib.SolverFactoryClass"


def __getattr__(name):
    if name in ("solver_factory_v1", "solver_factory_v2", "solver_factory_v3"):
        return solver_factory(int(name[-1]))
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


@overload
def solver_factory(version: None = None) -> int: ...
@overload
def solver_factory(version: Literal[1]) -> "_solvers.SolverFactoryClass": ...
@overload
def solver_factory(version: Literal[2]) -> "_appsi.SolverFactoryClass": ...
@overload
def solver_factory(version: Literal[3]) -> "_contrib.SolverFactoryClass": ...
@overload
def solver_factory(version: int) -> Any: ...


def solver_factory(version: int | None = None) -> int | Any:
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
    global _active_solver_factory_version
    import pyomo.opt.base.solvers as _solvers
    import pyomo.contrib.solver.common.factory as _contrib
    import pyomo.contrib.appsi.base as _appsi

    versions = {
        1: _solvers.LegacySolverFactory,
        2: _appsi.SolverFactory,
        3: _contrib.SolverFactory,
    }
    # The user is just asking what the current SolverFactory is; tell them.
    if version is None:
        if "_active_solver_factory_version" not in globals():
            for ver, cls in versions.items():
                if cls._cls is _environ.SolverFactory._cls:
                    _active_solver_factory_version = ver
                    break
        return _active_solver_factory_version
    #
    # Update the current SolverFactory to be a shim around (shallow copy
    # of) the new active factory
    src = versions.get(version, None)
    if src is not None:
        _active_solver_factory_version = version
        for attr in ('_description', '_cls', '_doc'):
            setattr(_environ.SolverFactory, attr, getattr(src, attr))
    else:
        raise ValueError(
            "Invalid value for target solver factory version; expected {1, 2, 3}, "
            f"received {version}"
        )
    return src


_active_solver_factory_version = solver_factory()
