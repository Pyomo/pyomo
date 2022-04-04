# -*- coding: utf-8 -*-
"""Main driver module for GDPopt solver.

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

from pyomo.common.deprecation import deprecation_warning
from pyomo.contrib.gdpopt.algorithm_base_class import (
    _GDPoptAlgorithm, _supported_algorithms)
from pyomo.contrib.gdpopt.branch_and_bound import GDP_LBB_Solver
from pyomo.contrib.gdpopt.config_options import (
    _add_mip_solver_configs, _add_nlp_solver_configs, _add_tolerance_configs,
    _add_OA_configs, _add_BB_configs)
from pyomo.contrib.gdpopt.gloa import GDP_GLOA_Solver
from pyomo.contrib.gdpopt.loa import GDP_LOA_Solver
from pyomo.contrib.gdpopt.ric import GDP_RIC_Solver
from pyomo.opt.base import SolverFactory

@SolverFactory.register(
    'gdpopt',
    doc='The GDPopt decomposition-based '
    'Generalized Disjunctive Programming (GDP) solver')
class GDPoptSolver(_GDPoptAlgorithm):
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
    # Declare configuration options for the GDPopt solver
    def __new__(cls, *args, **kwds):
        config = cls.CONFIG(kwds.pop('options', {}), preserve_implicit=True)
        config.set_value(kwds)
        algorithm = config.algorithm
        if algorithm is None:
            algorithm = config.strategy
        if algorithm is None:
            # raise deprecation warning
            deprecation_warning("Instantiating the gdpopt solver without "
                                "specifying an algorithm is deprecated. "
                                "For example, you should write: "
                                "SolverFactory('gdpopt', algorithm='LOA'), "
                                "replacing 'LOA' with a valid solution "
                                "algorithm.", version='TODO')
            return _HACK_GDPoptSolver(*args, **kwds)
        # We know at this point that algorithm is valid because of the config
        # logic
        solver = _supported_algorithms.get(algorithm)[0]
        return SolverFactory(solver, *args, **kwds)

@SolverFactory.register(
    '_HACK_gdpopt',
    doc='The GDPopt decomposition-based '
    'Generalized Disjunctive Programming (GDP) solver, supporting specifying'
    'solution algorithm in the call to solve.')
class _HACK_GDPoptSolver(_GDPoptAlgorithm):
    # This has to declare all the CONFIG options because they can't be
    # undeclared if they get populated in solve. And of course, by definition,
    # we don't know what this is.
    CONFIG = _GDPoptAlgorithm.CONFIG()
    _add_OA_configs(CONFIG)
    _add_mip_solver_configs(CONFIG)
    _add_nlp_solver_configs(CONFIG)
    _add_tolerance_configs(CONFIG)
    _add_BB_configs(CONFIG)

    def __init__(self, **kwds):
        config = self.CONFIG(kwds.pop('options', {}), preserve_implicit=True)
        config.set_value(kwds)

    def solve(self, model, **kwds):
        """Solve the model, depending on the value for config.algorithm. Note
        that this is merely a deprecation path and eventually setting the
        algorithm in the call to solve will not be supported: It will need to
        be set when the solver is instantiated.

        Args:
            model (Block): a Pyomo model or block to be solved

        """
        config = self.CONFIG(kwds.pop('options', {}), preserve_implicit=True)
        config.set_value(kwds)

        algorithm = config.algorithm
        if algorithm is None:
            algorithm = config.strategy
            kwds['algorithm'] = config.strategy
        # We know at this point that algorithm is valid because of the config
        # logic
        return SolverFactory(_supported_algorithms[algorithm][0]).solve(model,
                                                                        **kwds)
