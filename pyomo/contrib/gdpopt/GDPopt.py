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


from pyomo.common.config import (
    add_docstring_list
)
from pyomo.common.deprecation import deprecation_warning
from pyomo.opt.base import SolverFactory
#from pyomo.contrib.gdpopt.branch_and_bound import _perform_branch_and_bound
from pyomo.contrib.gdpopt.algorithm_base_class import _GDPoptAlgorithm
from pyomo.contrib.gdpopt.loa import GDP_LOA_Solver
#from pyomo.contrib.gdptopt.gloa import GDP_GLOA_Solver
# etc...

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
    <https://github.com/qtothec>.

    Several key GDPopt components were prototyped by BS and MS students:

    - Logic-based branch and bound: Sunjeev Kale
    - MC++ interface: Johnny Bates
    - LOA set-covering initialization: Eloy Fernandez

    """
    # Declare configuration options for the GDPopt solver
    def __new__(cls, *args, **kwds):
        config = cls.CONFIG(kwds.pop('options', {}), preserve_implicit=True)
        config.set_value(kwds)
        strategy = config.strategy
        if strategy is None:
            # raise deprecation warning
            deprecation_warning("Instantiating the gdpopt solver without "
                                "specifying an algorithm is deprecated. "
                                "For example, you should write: "
                                "SolverFactory('gdpopt', algorithm='LOA'), "
                                "replacing 'LOA' with a valid solution "
                                "strategy.", version='TODO')
            return _HACK_GDPoptSolver(*args, **kwds)
        solver = self.handlers.get(strategy)
        if solver is None:
            # TODO: make this more general...
            msg = 'Please specify a valid solution strategy. Options are: \n'
            msg += '    LOA:  Logic-based Outer Approximation\n'
            msg += '    GLOA: Global Logic-based Outer Approximation\n'
            msg += '    LBB:  Logic-based Branch and Bound\n'
            msg += '    RIC:  Relaxation with Integer Cuts'
            raise ValueError(msg)
        return SolverFactory(solver, *args, **kwds)

@SolverFactory.register(
    '_HACK_gdpopt',
    doc='The GDPopt decomposition-based '
    'Generalized Disjunctive Programming (GDP) solver, supporting specifying'
    'solution strategy in the call to solve.')
class _HACK_GDPoptSolver(_GDPoptAlgorithm):
    # def __init__(self, **kwds):
    #     config = self.CONFIG(kwds.pop('options', {}), preserve_implicit=True)
    #     config.set_value(kwds)

    def solve(self, model, **kwds):
        """Solve the model, depending on the value for config.strategy. Note
        that this is merely a deprecation path and eventually setting the 
        algorithm in the call to solve will not be supported: It will need to
        be set when the solver is instantiated.

        Args:
            model (Block): a Pyomo model or block to be solved

        """
        config = self.CONFIG(kwds.pop('options', {}), preserve_implicit=True)
        config.set_value(kwds)

        strategy = config.strategy
        if strategy == 'LOA':
            return SolverFactory('_logic_based_oa').solve(model, **kwds)
        elif strategy == 'GLOA':
            return SolverFactory('_global_logic_based_oa').solve(model, **kwds)
        elif strategy == 'RIC':
            return SolverFactory('_relaxation_with_integer_cuts').solve(model,
                                                                        **kwds)
        elif strategy == 'LBB':
            return SolverFactory('_logic_based_branch_and_bound').solve(model,
                                                                        **kwds)
        else:
            msg = 'Please specify a valid solution strategy. Options are: \n'
            msg += '    LOA:  Logic-based Outer Approximation\n'
            msg += '    GLOA: Global Logic-based Outer Approximation\n'
            msg += '    LBB:  Logic-based Branch and Bound\n'
            msg += '    RIC:  Relaxation with Integer Cuts'
            raise ValueError(msg)

# Add the CONFIG arguments to the solve method docstring
# GDPoptSolver.solve.__doc__ = add_docstring_list(
#     GDPoptSolver.solve.__doc__, GDPoptSolver.CONFIG, indent_by=8)
