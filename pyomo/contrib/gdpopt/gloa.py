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

from pyomo.common.config import document_kwargs_from_configdict
from pyomo.common.errors import DeveloperError
from pyomo.common.modeling import unique_component_name
from pyomo.contrib.gdp_bounds.info import disjunctive_bounds
from pyomo.contrib.gdpopt.algorithm_base_class import _GDPoptAlgorithm
from pyomo.contrib.gdpopt.config_options import (
    _add_oa_configs,
    _add_mip_solver_configs,
    _add_nlp_solver_configs,
    _add_tolerance_configs,
)
from pyomo.contrib.gdpopt.create_oa_subproblems import (
    _get_discrete_problem_and_subproblem,
    add_constraints_by_disjunct,
    add_global_constraint_list,
)
from pyomo.contrib.gdpopt.cut_generation import add_no_good_cut
from pyomo.contrib.gdpopt.oa_algorithm_utils import _OAAlgorithmMixIn
from pyomo.contrib.gdpopt.solve_discrete_problem import solve_MILP_discrete_problem
from pyomo.contrib.gdpopt.util import (
    _add_bigm_constraint_to_transformed_model,
    time_code,
)
from pyomo.contrib.mcpp.pyomo_mcpp import McCormick as mc, MCPP_Error

from pyomo.core import Constraint, Block, NonNegativeIntegers, Objective, value
from pyomo.core.expr.numvalue import is_potentially_variable
from pyomo.core.expr.visitor import identify_variables
from pyomo.opt.base import SolverFactory


@SolverFactory.register(
    'gdpopt.gloa',
    doc="The GLOA (global logic-based outer approximation) Generalized "
    "Disjunctive Programming (GDP) solver",
)
class GDP_GLOA_Solver(_GDPoptAlgorithm, _OAAlgorithmMixIn):
    """The GDPopt (Generalized Disjunctive Programming optimizer) global
    logic-based outer approximation (GLOA) solver.

    Accepts models that can include nonlinear, continuous variables and
    constraints, as well as logical conditions.
    """

    CONFIG = _GDPoptAlgorithm.CONFIG()
    _add_oa_configs(CONFIG)
    _add_mip_solver_configs(CONFIG)
    _add_nlp_solver_configs(CONFIG, default_solver='couenne')
    _add_tolerance_configs(CONFIG)

    algorithm = 'GLOA'

    # Override solve() to customize the docstring for this solver
    @document_kwargs_from_configdict(CONFIG, doc=_GDPoptAlgorithm.solve.__doc__)
    def solve(self, model, **kwds):
        return super().solve(model, **kwds)

    def _log_citation(self, config):
        config.logger.info(
            "\n"
            + """- GLOA algorithm:
        Lee, S; Grossmann, IE.
        A Global Optimization Algorithm for Nonconvex Generalized
        Disjunctive Programming and Applications to Process Systems.
        Comp. and Chem. Eng. 2001, 25, 1675-1697.
        DOI: 10.1016/S0098-1354(01)00732-3.
        """.strip()
        )

    def _solve_gdp(self, original_model, config):
        logger = config.logger

        # we need to gather a map of Disjuncts to their active Constraints
        # before we call any GDP transformations, as we will need this
        # information for cut generation later
        add_constraints_by_disjunct(self.original_util_block)
        # We also save these in advance because we know only linear logical
        # constraints will be added by the transformation to a MIP, so these are
        # all we'll ever need.
        add_global_constraint_list(self.original_util_block)
        (discrete_problem_util_block, subproblem_util_block) = (
            _get_discrete_problem_and_subproblem(self, config)
        )
        discrete = discrete_problem_util_block.parent_block()
        subproblem = subproblem_util_block.parent_block()
        discrete_obj = next(
            discrete.component_data_objects(Objective, active=True, descend_into=True)
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
                    mip_feasible, discrete_obj, logger
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
        """Add affine cuts"""
        m = discrete_problem_util_block.parent_block()
        if hasattr(discrete_problem_util_block, "aff_utils_blocks"):
            aff_utils_blocks = discrete_problem_util_block.aff_utils_blocks
        else:
            aff_utils_blocks = discrete_problem_util_block.aff_utils_blocks = dict()

        config.logger.debug("Adding affine cuts.")
        counter = 0
        for discrete_var, subprob_var in zip(
            discrete_problem_util_block.algebraic_variable_list,
            subproblem_util_block.algebraic_variable_list,
        ):
            val = subprob_var.value
            if val is not None and not discrete_var.fixed:
                discrete_var.set_value(val, skip_validation=True)

        for constr in self._get_active_untransformed_constraints(
            discrete_problem_util_block, config
        ):
            disjunctive_var_bounds = disjunctive_bounds(constr.parent_block())

            if constr.body.polynomial_degree() in (1, 0):
                continue

            vars_in_constr = list(identify_variables(constr.body))
            if any(var.value is None for var in vars_in_constr):
                continue  # a variable has no values

            # mcpp stuff
            try:
                mc_eqn = mc(constr.body, disjunctive_var_bounds)
            except MCPP_Error as e:
                config.logger.debug(
                    "Skipping constraint %s due to MCPP "
                    "error %s" % (constr.name, str(e))
                )
                continue  # skip to the next constraint
            ccSlope = mc_eqn.subcc()
            cvSlope = mc_eqn.subcv()
            ccStart = mc_eqn.concave()
            cvStart = mc_eqn.convex()
            ub_int = (
                min(value(constr.upper), mc_eqn.upper())
                if constr.has_ub()
                else mc_eqn.upper()
            )
            lb_int = (
                max(value(constr.lower), mc_eqn.lower())
                if constr.has_lb()
                else mc_eqn.lower()
            )

            parent_block = constr.parent_block()
            # Create a block on which to put outer approximation cuts.
            aff_utils = aff_utils_blocks.get(parent_block)
            if aff_utils is None:
                aff_utils = Block(doc="Block holding affine constraints")
                nm = unique_component_name(parent_block, "GDPopt_aff")
                parent_block.add_component(nm, aff_utils)
                aff_utils_blocks[parent_block] = aff_utils
                aff_utils.GDPopt_aff_cons = Constraint(NonNegativeIntegers)
            aff_cuts = aff_utils.GDPopt_aff_cons
            cut_body = sum(
                ccSlope[var] * (var - var.value)
                for var in vars_in_constr
                if not var.fixed
            )
            if not is_potentially_variable(cut_body):
                if (
                    cut_body + ccStart >= lb_int - config.constraint_tolerance
                    and cut_body + cvStart <= ub_int + config.constraint_tolerance
                ):
                    # We won't add them, but nothing is wrong--they hold
                    config.logger.debug("Affine cut is trivially True.")
                else:
                    # something went wrong.
                    raise DeveloperError("One of the affine cuts is trivially False.")
            else:
                concave_cut = cut_body + ccStart >= lb_int
                convex_cut = cut_body + cvStart <= ub_int
                idx = len(aff_cuts)
                aff_cuts[idx] = concave_cut
                aff_cuts[idx + 1] = convex_cut
                _add_bigm_constraint_to_transformed_model(m, aff_cuts[idx], aff_cuts)
                _add_bigm_constraint_to_transformed_model(
                    m, aff_cuts[idx + 1], aff_cuts
                )
                counter += 2

                config.logger.debug("Added %s affine cuts" % counter)
