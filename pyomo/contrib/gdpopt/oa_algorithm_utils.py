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

from pyomo.contrib.gdpopt.nlp_solve import solve_subproblem
from pyomo.contrib.gdpopt.util import fix_master_solution_in_subproblem
from pyomo.core import value
from pyomo.opt import TerminationCondition as tc

def _fix_master_soln_solve_subproblem_and_add_cuts(master_util_block,
                                                   subprob_util_block, config,
                                                   solver):
    with fix_master_solution_in_subproblem(master_util_block,
                                           subprob_util_block, config,
                                           config.force_subproblem_nlp):
        nlp_termination = solve_subproblem(subprob_util_block, config,
                                           solver.timing)
        if nlp_termination in {tc.optimal, tc.feasible}:
            primal_improved = solver._update_bounds_after_solve(
                'subproblem', primal=value(subprob_util_block.obj.expr),
                logger=config.logger)
            if primal_improved:
                solver.update_incumbent(subprob_util_block)
            solver._add_cuts_to_master_problem(subprob_util_block,
                                               master_util_block,
                                               solver.objective_sense, config,
                                               solver.timing)
        elif nlp_termination == tc.unbounded:
            # the whole problem is unbounded, we can stop
            solver._update_primal_bound_to_unbounded()

    return nlp_termination not in {tc.infeasible, tc.unbounded}
