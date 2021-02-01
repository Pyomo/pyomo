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
from __future__ import division

import six
from six import StringIO

from pyomo.common.config import (
    add_docstring_list
)
from pyomo.contrib.gdpopt.branch_and_bound import _perform_branch_and_bound
from pyomo.contrib.gdpopt.config_options import _get_GDPopt_config
from pyomo.contrib.gdpopt.iterate import GDPopt_iteration_loop
from pyomo.contrib.gdpopt.master_initialize import (
    GDPopt_initialize_master
)
from pyomo.contrib.gdpopt.util import (
    presolve_lp_nlp, process_objective,
    time_code, indent,
    setup_solver_environment)
from pyomo.opt.base import SolverFactory

__version__ = (20, 2, 28)  # Note: date-based version number


@SolverFactory.register(
    'gdpopt',
    doc='The GDPopt decomposition-based '
    'Generalized Disjunctive Programming (GDP) solver')
class GDPoptSolver(object):
    """Decomposition solver for Generalized Disjunctive Programming (GDP) problems.

    The GDPopt (Generalized Disjunctive Programming optimizer) solver applies a
    variety of decomposition-based approaches to solve Generalized Disjunctive
    Programming (GDP) problems. GDP models can include nonlinear, continuous
    variables and constraints, as well as logical conditions.

    These approaches include:

    - Logic-based outer approximation (LOA)
    - Logic-based branch-and-bound (LBB)
    - Partial surrogate cuts [pending]
    - Generalized Bender decomposition [pending]

    This solver implementation was developed by Carnegie Mellon University in the
    research group of Ignacio Grossmann.

    For nonconvex problems, LOA may not report rigorous lower/upper bounds.

    Questions: Please make a post at StackOverflow and/or contact Qi Chen
    <https://github.com/qtothec>.

    Several key GDPopt components were prototyped by BS and MS students:

    - Logic-based branch and bound: Sunjeev Kale
    - MC++ interface: Johnny Bates
    - LOA set-covering initialization: Eloy Fernandez

    """

    # Declare configuration options for the GDPopt solver
    CONFIG = _get_GDPopt_config()

    def solve(self, model, **kwds):
        """Solve the model.

        Warning: this solver is still in beta. Keyword arguments subject to
        change. Undocumented keyword arguments definitely subject to change.

        This function performs all of the GDPopt solver setup and problem
        validation. It then calls upon helper functions to construct the
        initial master approximation and iteration loop.

        Args:
            model (Block): a Pyomo model or block to be solved

        """
        config = self.CONFIG(kwds.pop('options', {}), preserve_implicit=True)
        config.set_value(kwds)

        with setup_solver_environment(model, config) as solve_data:
            self._log_solver_intro_message(config)
            solve_data.results.solver.name = 'GDPopt %s - %s' % (
                str(self.version()), config.strategy)

            # Verify that objective has correct form
            process_objective(solve_data, config)

            # Presolve LP or NLP problems using subsolvers
            presolved, presolve_results = presolve_lp_nlp(solve_data, config)
            if presolved:
                # TODO merge the solver results
                return presolve_results  # problem presolved

            if solve_data.active_strategy in {'LOA', 'GLOA'}:
                # Initialize the master problem
                with time_code(solve_data.timing, 'initialization'):
                    GDPopt_initialize_master(solve_data, config)

                # Algorithm main loop
                with time_code(solve_data.timing, 'main loop'):
                    GDPopt_iteration_loop(solve_data, config)
            elif solve_data.active_strategy == 'LBB':
                _perform_branch_and_bound(solve_data)

        return solve_data.results

    """Support use as a context manager under current solver API"""
    def __enter__(self):
        return self

    def __exit__(self, t, v, traceback):
        pass

    def available(self, exception_flag=True):
        """Check if solver is available.

        TODO: For now, it is always available. However, sub-solvers may not
        always be available, and so this should reflect that possibility.

        """
        return True

    def license_is_valid(self):
        return True

    def version(self):
        """Return a 3-tuple describing the solver version."""
        return __version__

    def _log_solver_intro_message(self, config):
        config.logger.info(
            "Starting GDPopt version %s using %s algorithm"
            % (".".join(map(str, self.version())), config.strategy)
        )
        mip_args_output = StringIO()
        nlp_args_output = StringIO()
        minlp_args_output = StringIO()
        lminlp_args_output = StringIO()
        config.mip_solver_args.display(ostream=mip_args_output)
        config.nlp_solver_args.display(ostream=nlp_args_output)
        config.minlp_solver_args.display(ostream=minlp_args_output)
        config.local_minlp_solver_args.display(ostream=lminlp_args_output)
        mip_args_text = indent(mip_args_output.getvalue().rstrip(), prefix=" " * 2 + " - ")
        nlp_args_text = indent(nlp_args_output.getvalue().rstrip(), prefix=" " * 2 + " - ")
        minlp_args_text = indent(minlp_args_output.getvalue().rstrip(), prefix=" " * 2 + " - ")
        lminlp_args_text = indent(lminlp_args_output.getvalue().rstrip(), prefix=" " * 2 + " - ")
        mip_args_text = "" if len(mip_args_text.strip()) == 0 else "\n" + mip_args_text
        nlp_args_text = "" if len(nlp_args_text.strip()) == 0 else "\n" + nlp_args_text
        minlp_args_text = "" if len(minlp_args_text.strip()) == 0 else "\n" + minlp_args_text
        lminlp_args_text = "" if len(lminlp_args_text.strip()) == 0 else "\n" + lminlp_args_text
        config.logger.info(
            """
Subsolvers:
- MILP: {milp}{milp_args}
- NLP: {nlp}{nlp_args}
- MINLP: {minlp}{minlp_args}
- local MINLP: {lminlp}{lminlp_args}
            """.format(
                milp=config.mip_solver,
                milp_args=mip_args_text,
                nlp=config.nlp_solver,
                nlp_args=nlp_args_text,
                minlp=config.minlp_solver,
                minlp_args=minlp_args_text,
                lminlp=config.local_minlp_solver,
                lminlp_args=lminlp_args_text,
            ).strip()
        )
        to_cite_text = """
If you use this software, you may cite the following:
- Implementation:
Chen, Q; Johnson, ES; Siirola, JD; Grossmann, IE.
Pyomo.GDP: Disjunctive Models in Python. 
Proc. of the 13th Intl. Symposium on Process Systems Eng.
San Diego, 2018.
        """.strip()
        if config.strategy == "LOA":
            to_cite_text += "\n"
            to_cite_text += """
- LOA algorithm:
Türkay, M; Grossmann, IE.
Logic-based MINLP algorithms for the optimal synthesis of process networks.
Comp. and Chem. Eng. 1996, 20(8), 959–978.
DOI: 10.1016/0098-1354(95)00219-7.
            """.strip()
        elif config.strategy == "GLOA":
            to_cite_text += "\n"
            to_cite_text += """
- GLOA algorithm:
Lee, S; Grossmann, IE.
A Global Optimization Algorithm for Nonconvex Generalized Disjunctive Programming and Applications to Process Systems.
Comp. and Chem. Eng. 2001, 25, 1675-1697.
DOI: 10.1016/S0098-1354(01)00732-3.
            """.strip()
        elif config.strategy == "LBB":
            to_cite_text += "\n"
            to_cite_text += """
- LBB algorithm:
Lee, S; Grossmann, IE.
New algorithms for nonlinear generalized disjunctive programming.
Comp. and Chem. Eng. 2000, 24, 2125-2141.
DOI: 10.1016/S0098-1354(00)00581-0.
            """.strip()
        config.logger.info(to_cite_text)

    _metasolver = False

    if six.PY2:
        __doc__ = """
    Keyword arguments below are specified for the :code:`solve` function.
        
    """ + add_docstring_list(__doc__, CONFIG)


if six.PY3:
    # Add the CONFIG arguments to the solve method docstring
    GDPoptSolver.solve.__doc__ = add_docstring_list(
        GDPoptSolver.solve.__doc__, GDPoptSolver.CONFIG, indent_by=8)
