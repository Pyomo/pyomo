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

from collections import namedtuple
from math import copysign

from pyomo.common.collections import ComponentMap
from pyomo.common.config import document_kwargs_from_configdict
from pyomo.common.modeling import unique_component_name
from pyomo.contrib.gdpopt.algorithm_base_class import _GDPoptAlgorithm
from pyomo.contrib.gdpopt.config_options import (
    _add_oa_configs,
    _add_mip_solver_configs,
    _add_nlp_solver_configs,
    _add_tolerance_configs,
)
from pyomo.contrib.gdpopt.create_oa_subproblems import (
    _get_discrete_problem_and_subproblem,
    add_constraint_list,
)
from pyomo.contrib.gdpopt.cut_generation import add_no_good_cut
from pyomo.contrib.gdpopt.oa_algorithm_utils import _OAAlgorithmMixIn
from pyomo.contrib.gdpopt.solve_discrete_problem import solve_MILP_discrete_problem
from pyomo.contrib.gdpopt.util import (
    time_code,
    _add_bigm_constraint_to_transformed_model,
)

from pyomo.core import (
    Block,
    Constraint,
    minimize,
    NonNegativeIntegers,
    NonNegativeReals,
    Objective,
    value,
    Var,
    VarList,
)
from pyomo.core.expr import differentiate
from pyomo.core.expr.visitor import identify_variables
from pyomo.gdp import Disjunct
from pyomo.opt.base import SolverFactory
from pyomo.repn import generate_standard_repn

MAX_SYMBOLIC_DERIV_SIZE = 1000
JacInfo = namedtuple('JacInfo', ['mode', 'vars', 'jac'])


@SolverFactory.register(
    'gdpopt.loa',
    doc="The LOA (logic-based outer approximation) Generalized Disjunctive "
    "Programming (GDP) solver",
)
class GDP_LOA_Solver(_GDPoptAlgorithm, _OAAlgorithmMixIn):
    """The GDPopt (Generalized Disjunctive Programming optimizer) logic-based
    outer approximation (LOA) solver.

    Accepts models that can include nonlinear, continuous variables and
    constraints, as well as logical conditions. For nonconvex problems, LOA
    may not report rigorous dual bounds.
    """

    CONFIG = _GDPoptAlgorithm.CONFIG()
    _add_oa_configs(CONFIG)
    _add_mip_solver_configs(CONFIG)
    _add_nlp_solver_configs(CONFIG, default_solver='ipopt')
    _add_tolerance_configs(CONFIG)

    algorithm = 'LOA'

    # Override solve() to customize the docstring for this solver
    @document_kwargs_from_configdict(CONFIG, doc=_GDPoptAlgorithm.solve.__doc__)
    def solve(self, model, **kwds):
        return super().solve(model, **kwds)

    def _log_citation(self, config):
        config.logger.info(
            "\n"
            + """- LOA algorithm:
        Türkay, M; Grossmann, IE.
        Logic-based MINLP algorithms for the optimal synthesis of process
        networks. Comp. and Chem. Eng. 1996, 20(8), 959–978.
        DOI: 10.1016/0098-1354(95)00219-7.
        """.strip()
        )

    def _solve_gdp(self, original_model, config):
        logger = config.logger

        # We'll need these to get dual info after solving subproblems
        add_constraint_list(self.original_util_block)

        (
            discrete_problem_util_block,
            subproblem_util_block,
        ) = _get_discrete_problem_and_subproblem(self, config)

        discrete = discrete_problem_util_block.parent_block()
        subproblem = subproblem_util_block.parent_block()

        original_obj = self._setup_augmented_penalty_objective(
            discrete_problem_util_block
        )

        self._log_header(logger)

        # main loop
        while not config.iterlim or self.iteration < config.iterlim:
            self.iteration += 1

            # solve linear discrete problem
            with time_code(self.timing, 'mip'):
                oa_obj = self._update_augmented_penalty_objective(
                    discrete_problem_util_block, original_obj, config.OA_penalty_factor
                )
                mip_feasible = solve_MILP_discrete_problem(
                    discrete_problem_util_block, self, config
                )
                self._update_bounds_after_discrete_problem_solve(
                    mip_feasible, oa_obj, logger
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

    def _setup_augmented_penalty_objective(self, discrete_problem_util_block):
        m = discrete_problem_util_block.parent_block()
        discrete_objective = next(m.component_data_objects(Objective, active=True))

        # Set up augmented penalty objective
        discrete_objective.deactivate()
        # placeholder for OA objective
        discrete_problem_util_block.oa_obj = Objective(sense=minimize)

        return discrete_objective

    def _update_augmented_penalty_objective(
        self, discrete_problem_util_block, discrete_objective, OA_penalty_factor
    ):
        m = discrete_problem_util_block.parent_block()
        sign_adjust = 1 if discrete_objective.sense == minimize else -1
        OA_penalty_expr = (
            sign_adjust
            * OA_penalty_factor
            * sum(
                v
                for v in m.component_data_objects(
                    ctype=Var, descend_into=(Block, Disjunct)
                )
                if v.parent_component().local_name == 'GDPopt_OA_slacks'
            )
        )
        discrete_problem_util_block.oa_obj.expr = (
            discrete_objective.expr + OA_penalty_expr
        )

        return discrete_problem_util_block.oa_obj.expr

    def _add_cuts_to_discrete_problem(
        self,
        subproblem_util_block,
        discrete_problem_util_block,
        objective_sense,
        config,
        timing,
    ):
        """Add outer approximation cuts to the linear GDP model."""
        m = discrete_problem_util_block.parent_block()
        nlp = subproblem_util_block.parent_block()
        sign_adjust = -1 if objective_sense == minimize else 1
        # Dictionary mapping blocks to their child blocks we use to store OA
        # cuts. We do it this way because we don't know for sure we can have any
        # given name since we are sticking these on a clone of a user-generated
        # model. But this keeps track that we find one if we've already created
        # it, so that we can add the cuts as indexed constraints.
        if hasattr(discrete_problem_util_block, 'oa_cut_blocks'):
            oa_cut_blocks = discrete_problem_util_block.oa_cut_blocks
        else:
            oa_cut_blocks = discrete_problem_util_block.oa_cut_blocks = dict()

        for discrete_var, subprob_var in zip(
            discrete_problem_util_block.algebraic_variable_list,
            subproblem_util_block.algebraic_variable_list,
        ):
            val = subprob_var.value
            if val is not None and not discrete_var.fixed:
                discrete_var.set_value(val, skip_validation=True)

        config.logger.debug('Adding OA cuts.')

        counter = 0
        if not hasattr(discrete_problem_util_block, 'jacobians'):
            discrete_problem_util_block.jacobians = ComponentMap()
        for constr, subprob_constr in zip(
            discrete_problem_util_block.constraint_list,
            subproblem_util_block.constraint_list,
        ):
            dual_value = nlp.dual.get(subprob_constr, None)
            if dual_value is None or generate_standard_repn(constr.body).is_linear():
                continue

            # Determine if the user pre-specified that OA cuts should not be
            # generated for the given constraint.
            parent_block = constr.parent_block()
            ignore_set = getattr(parent_block, 'GDPopt_ignore_OA', None)
            config.logger.debug('Ignore_set %s' % ignore_set)
            if ignore_set and (
                constr in ignore_set or constr.parent_component() in ignore_set
            ):
                config.logger.debug(
                    'OA cut addition for %s skipped because it is in '
                    'the ignore set.' % constr.name
                )
                continue

            config.logger.debug(
                "Adding OA cut for %s with dual value %s" % (constr.name, dual_value)
            )

            # Cache jacobian
            jacobian = discrete_problem_util_block.jacobians.get(constr, None)
            if jacobian is None:
                constr_vars = list(identify_variables(constr.body, include_fixed=False))
                if len(constr_vars) >= MAX_SYMBOLIC_DERIV_SIZE:
                    mode = differentiate.Modes.reverse_numeric
                else:
                    mode = differentiate.Modes.sympy

                try:
                    jac_list = differentiate(
                        constr.body, wrt_list=constr_vars, mode=mode
                    )
                    jac_map = ComponentMap(zip(constr_vars, jac_list))
                except:
                    if mode is differentiate.Modes.reverse_numeric:
                        raise
                    mode = differentiate.Modes.reverse_numeric
                    jac_map = ComponentMap()
                jacobian = JacInfo(mode=mode, vars=constr_vars, jac=jac_map)
                discrete_problem_util_block.jacobians[constr] = jacobian
            # Recompute numeric derivatives
            if not jacobian.jac:
                jac_list = differentiate(
                    constr.body, wrt_list=jacobian.vars, mode=jacobian.mode
                )
                jacobian.jac.update(zip(jacobian.vars, jac_list))

            # Create a block on which to put outer approximation cuts, if we
            # don't have one already on this parent block.
            oa_utils = oa_cut_blocks.get(parent_block)
            if oa_utils is None:
                nm = unique_component_name(parent_block, 'GDPopt_OA_cuts')
                oa_utils = Block(
                    doc="Block holding outer approximation cuts and associated data."
                )
                parent_block.add_component(nm, oa_utils)
                oa_cut_blocks[parent_block] = oa_utils
                oa_utils.cuts = Constraint(NonNegativeIntegers)
            discrete_prob_oa_utils = discrete_problem_util_block.component(
                'GDPopt_OA_slacks'
            )
            if discrete_prob_oa_utils is None:
                discrete_prob_oa_utils = (
                    discrete_problem_util_block.GDPopt_OA_slacks
                ) = Block(
                    doc="Block holding outer approximation "
                    "slacks for the whole model (so that the "
                    "writers can find them)."
                )
                discrete_prob_oa_utils.slacks = VarList(
                    bounds=(0, config.max_slack), domain=NonNegativeReals, initialize=0
                )

            oa_cuts = oa_utils.cuts
            slack_var = discrete_prob_oa_utils.slacks.add()
            rhs = value(constr.lower) if constr.has_lb() else value(constr.upper)
            try:
                new_oa_cut = (
                    copysign(1, sign_adjust * dual_value)
                    * (
                        value(constr.body)
                        - rhs
                        + sum(
                            value(jac) * (var - value(var))
                            for var, jac in jacobian.jac.items()
                        )
                    )
                    - slack_var
                    <= 0
                )
                assert new_oa_cut.polynomial_degree() in (1, 0)
                idx = len(oa_cuts)
                oa_cuts[idx] = new_oa_cut
                _add_bigm_constraint_to_transformed_model(m, oa_cuts[idx], oa_cuts)
                config.logger.debug("Cut expression: %s" % new_oa_cut)
                counter += 1
            except ZeroDivisionError:
                config.logger.warning(
                    "Zero division occurred attempting to generate OA cut for "
                    "constraint %s.\n"
                    "Skipping OA cut generation for this constraint." % (constr.name,)
                )
                # Simply continue on to the next constraint.
            # Clear out the numeric Jacobian values
            if jacobian.mode is differentiate.Modes.reverse_numeric:
                jacobian.jac.clear()

        config.logger.debug('Added %s OA cuts' % counter)
