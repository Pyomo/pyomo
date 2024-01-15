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

from math import fabs
from pyomo.contrib.gdpopt.solve_subproblem import solve_subproblem
from pyomo.contrib.gdpopt.util import fix_discrete_problem_solution_in_subproblem
from pyomo.core import value
from pyomo.opt import TerminationCondition as tc


class _OAAlgorithmMixIn(object):
    def _fix_discrete_soln_solve_subproblem_and_add_cuts(
        self, discrete_prob_util_block, subprob_util_block, config
    ):
        with fix_discrete_problem_solution_in_subproblem(
            discrete_prob_util_block, subprob_util_block, self, config
        ):
            nlp_termination = solve_subproblem(subprob_util_block, self, config)
            if nlp_termination in {tc.optimal, tc.feasible}:
                primal_improved = self._update_bounds_after_solve(
                    'subproblem',
                    primal=value(subprob_util_block.obj.expr),
                    logger=config.logger,
                )
                if primal_improved:
                    self.update_incumbent(subprob_util_block)
                self._add_cuts_to_discrete_problem(
                    subprob_util_block,
                    discrete_prob_util_block,
                    self.objective_sense,
                    config,
                    self.timing,
                )
            elif nlp_termination == tc.unbounded:
                # the whole problem is unbounded, we can stop
                self._update_primal_bound_to_unbounded(config)

        return nlp_termination not in {tc.infeasible, tc.unbounded}

    # Utility used in cut_generation: We saved a map of Disjuncts to the active
    # constraints they contain on the discrete problem util_block, and use it
    # here to find the active constraints under the current discrete
    # solution. Note that this preprocesses not just to be efficient, but
    # because everything on the Disjuncts is deactivated at this point, since
    # we've already transformed the discrete problem to a MILP
    def _get_active_untransformed_constraints(self, util_block, config):
        """Yield constraints in disjuncts where the indicator value is set or
        fixed to True."""
        model = util_block.parent_block()
        # Get active global constraints
        for constr in util_block.global_constraint_list:
            yield constr
        # get all the disjuncts in the original model. Check which ones are
        # True.
        for disj, constr_list in util_block.constraints_by_disjunct.items():
            if fabs(disj.binary_indicator_var.value - 1) <= config.integer_tolerance:
                for constr in constr_list:
                    yield constr
