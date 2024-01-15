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

from pyomo.common.config import document_kwargs_from_configdict
from pyomo.contrib.gdpopt.algorithm_base_class import _GDPoptAlgorithm
from pyomo.contrib.gdpopt.config_options import (
    _add_mip_solver_configs,
    _add_nlp_solver_configs,
    _add_tolerance_configs,
    _add_oa_configs,
)
from pyomo.contrib.gdpopt.create_oa_subproblems import (
    _get_discrete_problem_and_subproblem,
)
from pyomo.contrib.gdpopt.oa_algorithm_utils import _OAAlgorithmMixIn
from pyomo.contrib.gdpopt.cut_generation import add_no_good_cut
from pyomo.contrib.gdpopt.solve_discrete_problem import solve_MILP_discrete_problem
from pyomo.contrib.gdpopt.util import time_code
from pyomo.core import Objective
from pyomo.opt.base import SolverFactory

# ESJ: In the future, if we have a direct interface to cplex or gurobi, we
# should get the integer solutions several-at-a-time with a solution pool or
# something of the like...


@SolverFactory.register(
    'gdpopt.ric',
    doc="The RIC (relaxation with integer cuts) Generalized Disjunctive "
    "Programming (GDP) solver",
)
class GDP_RIC_Solver(_GDPoptAlgorithm, _OAAlgorithmMixIn):
    """The GDPopt (Generalized Disjunctive Programming optimizer) relaxation
    with integer cuts (RIC) solver.

    Accepts models that can include nonlinear, continuous variables and
    constraints, as well as logical conditions. For non-convex problems, RIC
    will not be exact unless the NLP subproblems are solved globally.
    """

    CONFIG = _GDPoptAlgorithm.CONFIG()
    _add_mip_solver_configs(CONFIG)
    _add_nlp_solver_configs(CONFIG, default_solver='ipopt')
    _add_tolerance_configs(CONFIG)
    _add_oa_configs(CONFIG)

    algorithm = 'RIC'

    # Override solve() to customize the docstring for this solver
    @document_kwargs_from_configdict(CONFIG, doc=_GDPoptAlgorithm.solve.__doc__)
    def solve(self, model, **kwds):
        return super().solve(model, **kwds)

    def _solve_gdp(self, original_model, config):
        logger = config.logger

        (
            discrete_problem_util_block,
            subproblem_util_block,
        ) = _get_discrete_problem_and_subproblem(self, config)
        discrete_problem = discrete_problem_util_block.parent_block()
        subproblem = subproblem_util_block.parent_block()
        discrete_problem_obj = next(
            discrete_problem.component_data_objects(
                Objective, active=True, descend_into=True
            )
        )

        self._log_header(logger)

        # main loop
        while not config.iterlim or self.iteration < config.iterlim:
            self.iteration += 1

            # solve linear discrete problem
            with time_code(self.timing, 'mip'):
                mip_feasible = solve_MILP_discrete_problem(
                    discrete_problem_util_block, self, config
                )
                self._update_bounds_after_discrete_problem_solve(
                    mip_feasible, discrete_problem_obj, logger
                )

            # Check termination conditions
            if self.any_termination_criterion_met(config):
                break

            with time_code(self.timing, 'nlp'):
                self._fix_discrete_soln_solve_subproblem_and_add_cuts(
                    discrete_problem_util_block, subproblem_util_block, config
                )

            # Add integer cut
            with time_code(self.timing, "integer cut generation"):
                add_no_good_cut(discrete_problem_util_block, config)

            # Check termination conditions
            if self.any_termination_criterion_met(config):
                break

    def _add_cuts_to_discrete_problem(
        self,
        subproblem_util_block,
        discrete_problem_util_block,
        objective_sense,
        config,
        timing,
    ):
        # Nothing to do here
        pass
