# -*- coding: utf-8 -*-

# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Implementation of the MindtPy solver.

22.2.10 changes:
- Add support for partitioning nonlinear-sum objective.

22.1.12 changes:
- Improve the log.

21.12.15 changes:
- Improve the online doc.

21.11.10 changes:
- Add support for solution pool of MIP solvers.

21.8.21 changes:
- Add support for gurobi_persistent solver in (Regularized) LP/NLP-based B&B algorithm.

21.5.19 changes:
- Add Feasibility Pump strategy.
- Add Regularized Outer Approximation method.
- Restructure and simplify the MindyPy code.

20.10.15 changes:
- Add Extended Cutting Plane and Global Outer Approximation strategy.
- Update online doc.

20.6.30 changes:
- Add support for different norms (L1, L2, L-infinity) of the objective function in the feasibility subproblem.
- Add support for different differentiate_mode to calculate Jacobian.

20.6.9 changes:
- Add cycling check in Outer Approximation method.
- Add support for GAMS solvers interface.
- Fix warmstart for both OA and LP/NLP method.

20.5.9 changes:
- Add single-tree implementation.
- Add support for cplex_persistent solver.
- Fix bug in OA cut expression in cut_generation.py.

24.1.11 changes:
- fix gurobi single tree termination check bug
- fix Gurobi single tree cycle handling
- fix bug in feasibility pump method
- add special handling for infeasible relaxed NLP
- update the log format of infeasible fixed NLP subproblems
- create a new copy_var_list_values function
"""

from pyomo.contrib.mindtpy import __version__
from pyomo.opt import SolverFactory
from pyomo.contrib.mindtpy.config_options import _get_MindtPy_config
from pyomo.common.config import document_kwargs_from_configdict
from pyomo.contrib.mindtpy.config_options import _supported_algorithms


@SolverFactory.register(
    'mindtpy', doc='MindtPy: Mixed-Integer Nonlinear Decomposition Toolbox in Pyomo'
)
class MindtPySolver:
    """
    Decomposition solver for Mixed-Integer Nonlinear Programming (MINLP) problems.

    The MindtPy (Mixed-Integer Nonlinear Decomposition Toolbox in Pyomo) solver
    applies a variety of decomposition-based approaches to solve Mixed-Integer
    Nonlinear Programming (MINLP) problems.
    These approaches include:

    - Outer approximation (OA)
    - Global outer approximation (GOA)
    - Regularized outer approximation (ROA)
    - LP/NLP based branch-and-bound (LP/NLP)
    - Global LP/NLP based branch-and-bound (GLP/NLP)
    - Regularized LP/NLP based branch-and-bound (RLP/NLP)
    - Feasibility pump (FP)
    """

    CONFIG = _get_MindtPy_config()

    def available(self, exception_flag=True):
        """Check whether the solver interface is available.

        Parameters
        ----------
        exception_flag : bool, optional
            Included for API compatibility and ignored by this implementation.

        Returns
        -------
        bool
            Always ``True`` for the MindtPy interface.
        """
        return True

    def license_is_valid(self):
        """Report whether the MindtPy solver interface is licensed."""
        return True

    def version(self):
        """Return a 3-tuple describing the solver version."""
        return __version__

    @document_kwargs_from_configdict(CONFIG)
    def solve(self, model, **kwds):
        """Solve a model with the configured MindtPy strategy.

        Parameters
        ----------
        model : Block
            Pyomo model or block to solve.
        **kwds
            Additional keyword arguments forwarded to the selected algorithm.

        Returns
        -------
        SolverResults
            Results object returned by the selected MindtPy algorithm.
        """
        # The algorithm should have been specified as an argument to the solve
        # method. We will instantiate an ephemeral instance of the correct
        # solver and call its solve method.
        options = kwds.pop('options', {})
        config = self.CONFIG(options, preserve_implicit=True)
        # Don't complain about extra things, they aren't for us. We just need to
        # get the algorithm and then our job is done.
        config.set_value(kwds, skip_implicit=True)

        return SolverFactory(_supported_algorithms[config.strategy][0]).solve(
            model, **kwds
        )

    #
    # Support 'with' statements.
    #
    def __enter__(self):
        """Return this solver instance for context-manager support."""
        return self

    def __exit__(self, t, v, traceback):
        """Exit the context manager without additional teardown actions.

        Parameters
        ----------
        t : type or None
            Exception type if an exception was raised in the context block.
        v : BaseException or None
            Exception value if an exception was raised in the context block.
        traceback : traceback or None
            Traceback object if an exception was raised in the context block.
        """
        pass
