# -*- coding: utf-8 -*-

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
"""

from pyomo.contrib.mindtpy import __version__
from pyomo.opt import SolverFactory
from pyomo.contrib.mindtpy.config_options import _get_MindtPy_config
from pyomo.common.config import document_kwargs_from_configdict
from pyomo.contrib.mindtpy.config_options import _supported_algorithms


@SolverFactory.register(
    'mindtpy', doc='MindtPy: Mixed-Integer Nonlinear Decomposition Toolbox in Pyomo'
)
class MindtPySolver(object):
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
        """Check if solver is available."""
        return True

    def license_is_valid(self):
        return True

    def version(self):
        """Return a 3-tuple describing the solver version."""
        return __version__

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

        return SolverFactory(_supported_algorithms[config.strategy][0]).solve(
            model, **kwds
        )

    #
    # Support 'with' statements.
    #
    def __enter__(self):
        return self

    def __exit__(self, t, v, traceback):
        pass
