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

import pyomo.environ as _environ

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


def __getattr__(name):
    if name in ('solver_factory_v1', 'solver_factory_v2', 'solver_factory_v3'):
        return solver_factory(int(name[-1]))
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def solver_factory(version=None):
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
    import pyomo.opt.base.solvers as _solvers
    import pyomo.contrib.solver.common.factory as _contrib
    import pyomo.contrib.appsi.base as _appsi

    versions = {
        1: _solvers.LegacySolverFactory,
        2: _appsi.SolverFactory,
        3: _contrib.SolverFactory,
    }

    current = getattr(solver_factory, '_active_version', None)
    # First time through, _active_version is not defined.  Go look and
    # see what it was initialized to in pyomo.environ
    if current is None:
        for ver, cls in versions.items():
            if cls._cls is _environ.SolverFactory._cls:
                solver_factory._active_version = ver
                break
        return solver_factory._active_version
    #
    # The user is just asking what the current SolverFactory is; tell them.
    if version is None:
        return solver_factory._active_version
    #
    # Update the current SolverFactory to be a shim around (shallow copy
    # of) the new active factory
    src = versions.get(version, None)
    if version is not None:
        solver_factory._active_version = version
        for attr in ('_description', '_cls', '_doc'):
            setattr(_environ.SolverFactory, attr, getattr(src, attr))
    else:
        raise ValueError(
            "Invalid value for target solver factory version; expected {1, 2, 3}, "
            f"received {version}"
        )
    return src


solver_factory._active_version = solver_factory()
