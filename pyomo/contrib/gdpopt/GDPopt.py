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

"""Main driver module for GDPopt solver.

22.5.13 changes:
- rewrite of all algorithms
- deprecate 'strategy' in favor of 'algorithm'
- deprecate 'init_strategy' in favor of 'init_algorithm'
20.2.28 changes:
- bugfixes on tests
20.1.22 changes:
- improved subsolver time limit support for GAMS interface
- add maxTimeLimit exit condition for GDPopt-LBB
- add token Big M for reactivated constraints in GDPopt-LBB
- activate fbbt for branch-and-bound nodes
20.1.15 changes:
- internal cleanup of codebase
- merge GDPbb capabilities (logic-based branch and bound)
- refactoring of GDPbb code
- update logging information to include subsolver options
- improve SuppressInfeasibleWarning
- simplify mip preprocessing
- remove not-fully-implemented 'backtracking' from LOA
19.10.11 changes:
- bugfix on SolverStatus error message
19.5.13 changes:
- add handling to integer cuts for disjunct pruning during FBBT
19.4.23 changes:
- add support for linear subproblems
- use automatic differentiation for large constraints
- bugfixes on time limit support
- treat fixed variables as constants in GLOA cut generation
19.3.25 changes:
- add rudimentary time limit support
- start keeping basic changelog

"""
from pyomo.common.config import document_kwargs_from_configdict, ConfigDict
from pyomo.contrib.gdpopt import __version__
from pyomo.contrib.gdpopt.config_options import (
    _add_common_configs,
    _supported_algorithms,
    _get_algorithm_config,
)
from pyomo.opt.base import SolverFactory


def _handle_strategy_deprecation(config):
    # This method won't be needed when the strategy arg is removed, but for now,
    # we need to copy it over as algorithm. The config system already gave the
    # deprecation warning.
    if config.algorithm is None and config.strategy is not None:
        config.algorithm = config.strategy


@SolverFactory.register(
    'gdpopt',
    doc='The GDPopt decomposition-based '
    'Generalized Disjunctive Programming (GDP) solver',
)
class GDPoptSolver(object):
    """Decomposition solver for Generalized Disjunctive Programming (GDP)
    problems.

    The GDPopt (Generalized Disjunctive Programming optimizer) solver applies a
    variety of decomposition-based approaches to solve Generalized Disjunctive
    Programming (GDP) problems. GDP models can include nonlinear, continuous
    variables and constraints, as well as logical conditions.

    These approaches include:

    - Logic-based outer approximation (LOA)
    - Logic-based branch-and-bound (LBB)
    - Partial surrogate cuts [pending]
    - Generalized Bender decomposition [pending]

    This solver implementation was developed by Carnegie Mellon University in
    the research group of Ignacio Grossmann.

    For nonconvex problems, LOA may not report rigorous lower/upper bounds.

    Questions: Please make a post at StackOverflow and/or contact Qi Chen
    <https://github.com/qtothec> or David Bernal <https://github.com/bernalde>.

    Several key GDPopt components were prototyped by BS and MS students:

    - Logic-based branch and bound: Sunjeev Kale
    - MC++ interface: Johnny Bates
    - LOA set-covering initialization: Eloy Fernandez
    - Logic-to-linear transformation: Romeo Valentin

    """

    CONFIG = ConfigDict("GDPopt")
    _add_common_configs(CONFIG)

    @document_kwargs_from_configdict(CONFIG)
    def solve(self, model, **kwds):
        """Solve the model.

        Args:
            model (Block): a Pyomo model or block to be solved

        """
        # The algorithm should have been specified as an argument to the solve
        # method. We will instantiate an ephemeral instance of the correct
        # solver and call its solve method.
        options = kwds.pop('options', {})
        config = self.CONFIG(options, preserve_implicit=True)
        # Don't complain about extra things, they aren't for us. We just need to
        # get the algorithm and then our job is done.
        config.set_value(kwds, skip_implicit=True)

        alg_config = _get_algorithm_config()(options, preserve_implicit=True)
        alg_config.set_value(kwds, skip_implicit=True)

        _handle_strategy_deprecation(alg_config)
        algorithm = alg_config.algorithm
        if algorithm is None:
            raise ValueError(
                "No algorithm was specified to the solve method. "
                "Please specify an algorithm or use an "
                "algorithm-specific solver."
            )

        # get rid of 'algorithm' and 'strategy' if they exist so that the solver
        # can validate.
        kwds.pop('algorithm', None)
        kwds.pop('strategy', None)

        # The algorithm has already been validated, so this will work.
        return SolverFactory(_supported_algorithms[algorithm][0]).solve(model, **kwds)

    # Support use as a context manager under current solver API
    def __enter__(self):
        return self

    def __exit__(self, t, v, traceback):
        pass

    def available(self, exception_flag=True):
        """Solver is always available. Though subsolvers may not be, they will
        raise an error when the time comes.
        """
        return True

    def license_is_valid(self):
        return True

    def version(self):
        """Return a 3-tuple describing the solver version."""
        return __version__

    _metasolver = False
