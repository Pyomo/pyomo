#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from collections import namedtuple
from math import copysign

from pyomo.common.collections import ComponentMap
from pyomo.common.config import add_docstring_list
from pyomo.common.modeling import unique_component_name
from pyomo.contrib.gdpopt.algorithm_base_class import _GDPoptAlgorithm
from pyomo.contrib.gdpopt.config_options import (
    _add_OA_configs, _add_mip_solver_configs, _add_nlp_solver_configs,
    _add_tolerance_configs)
from pyomo.contrib.gdpopt.create_oa_subproblems import (
    _get_master_and_subproblem, add_constraint_list)
from pyomo.contrib.gdpopt.cut_generation import add_no_good_cut
from pyomo.contrib.gdpopt.mip_solve import solve_MILP_master_problem
from pyomo.contrib.gdpopt.oa_algorithm_utils import (
    _fix_master_soln_solve_subproblem_and_add_cuts)
from pyomo.contrib.gdpopt.util import (
    time_code, lower_logger_level_to, _add_bigm_constraint_to_transformed_model)

from pyomo.core import (
    Block, Constraint, minimize, NonNegativeIntegers, NonNegativeReals,
    Objective, value, Var, VarList)
from pyomo.core.expr import differentiate
from pyomo.core.expr.visitor import identify_variables
from pyomo.gdp import Disjunct
from pyomo.opt.base import SolverFactory
from pyomo.opt import TerminationCondition
from pyomo.repn import generate_standard_repn
import logging

MAX_SYMBOLIC_DERIV_SIZE = 1000
JacInfo = namedtuple('JacInfo', ['mode','vars','jac'])

@SolverFactory.register(
    '_logic_based_oa',
    doc='GDP Logic-Based Outer Approximation (LOA) solver')
class GDP_LOA_Solver(_GDPoptAlgorithm):
    """The GDPopt (Generalized Disjunctive Programming optimizer) logic-based
    outer approximation (LOA) solver.

    Accepts models that can include nonlinear, continuous variables and
    constraints, as well as logical conditions. For nonconvex problems, LOA
    may not report rigorous lower/upper bounds.
    """
    CONFIG = _GDPoptAlgorithm.CONFIG()
    _add_OA_configs(CONFIG)
    _add_mip_solver_configs(CONFIG)
    _add_nlp_solver_configs(CONFIG)
    _add_tolerance_configs(CONFIG)

    def __init__(self, **kwds):
        self.CONFIG = self.CONFIG(kwds)
        super(GDP_LOA_Solver, self).__init__()

    def solve(self, model, **kwds):
        """Solve the model with LOA

        Args:
            model (Block): a Pyomo model or block to be solved.

        """
        config = self.CONFIG(kwds.pop('options', {}), preserve_implicit=True)
        config.set_value(kwds)

        with time_code(self.timing, 'total', is_main_timer=True), \
            lower_logger_level_to(config.logger, config.tee):
            results = super().solve(model, config)
            if results:
                return results
            return self._solve_gdp_with_loa(model, config)

    def _solve_gdp_with_loa(self, original_model, config):
        logger = config.logger

        # We'll need these to get dual info after solving subproblems
        add_constraint_list(self.original_util_block)

        (master_util_block,
         subproblem_util_block) = _get_master_and_subproblem(self, config)

        master = master_util_block.model()
        subproblem = subproblem_util_block.model()

        original_obj = self._setup_augmented_Lagrangian_objective(
            master_util_block)

        self._log_header(logger)

        # main loop
        while self.iteration < config.iterlim:
            self.iteration += 1

            # solve linear master problem
            with time_code(self.timing, 'mip'):
                oa_obj = self._update_augmented_Lagrangian_objective(
                    master_util_block, original_obj, config.OA_penalty_factor)
                mip_feasible = solve_MILP_master_problem(master_util_block,
                                                         config, self.timing)
                self._update_bounds_after_master_problem_solve(mip_feasible,
                                                               oa_obj, logger)

            # Check termination conditions
            if self.any_termination_criterion_met(config):
                break

            with time_code(self.timing, 'nlp'):
                _fix_master_soln_solve_subproblem_and_add_cuts(
                    master_util_block, subproblem_util_block, config, self)

            # Add integer cut
            with time_code(self.timing, "integer cut generation"):
                add_no_good_cut(master_util_block, config)

            # Check termination conditions
            if self.any_termination_criterion_met(config):
                break

        self._get_final_pyomo_results_object()
        self._log_termination_message(logger)
        if self.pyomo_results.solver.termination_condition not in \
           {TerminationCondition.infeasible, TerminationCondition.unbounded}:
            self._transfer_incumbent_to_original_model()
        return self.pyomo_results

    def _setup_augmented_Lagrangian_objective(self, master_util_block):
        m = master_util_block.model()
        main_objective = next(m.component_data_objects(Objective, active=True))

        # Set up augmented Lagrangean penalty objective
        main_objective.deactivate()
        # placeholder for oa objective
        master_util_block.oa_obj = Objective(sense=minimize)

        return main_objective

    def _update_augmented_Lagrangian_objective(self, master_util_block,
                                               main_objective,
                                               OA_penalty_factor):
        m = master_util_block.model()
        sign_adjust = 1 if main_objective.sense == minimize else -1
        OA_penalty_expr = sign_adjust * OA_penalty_factor * \
                          sum(v for v in m.component_data_objects(
                              ctype=Var, descend_into=(Block, Disjunct))
                          if v.parent_component().local_name ==
                              'GDPopt_OA_slacks')
        master_util_block.oa_obj.expr = main_objective.expr + OA_penalty_expr

        return master_util_block.oa_obj.expr

    def _add_cuts_to_master_problem(self, subproblem_util_block,
                                    master_util_block, objective_sense, config,
                                    timing):
        """Add outer approximation cuts to the linear GDP model."""
        m = master_util_block.model()
        nlp = subproblem_util_block.model()
        sign_adjust = -1 if objective_sense == minimize else 1
        # Dictionary mapping blocks to their child blocks we use to store OA
        # cuts. We do it this way because we don't know for sure we can have any
        # given name since we are sticking these on a clone of a user-generated
        # model. But this keeps track that we find one if we've already created
        # it, so that we can add the cuts as indexed constraints.
        if hasattr(master_util_block, 'oa_cut_blocks'):
            oa_cut_blocks = master_util_block.oa_cut_blocks
        else:
            oa_cut_blocks = master_util_block.oa_cut_blocks = dict()

        for master_var, subprob_var in zip(
                master_util_block.algebraic_variable_list,
                subproblem_util_block.algebraic_variable_list):
            val = subprob_var.value
            if val is not None and not master_var.fixed:
                master_var.set_value(val, skip_validation=True)

        config.logger.debug('Adding OA cuts.')

        counter = 0
        if not hasattr(master_util_block, 'jacobians'):
            master_util_block.jacobians = ComponentMap()
        for constr, subprob_constr in zip(
                master_util_block.constraint_list,
                subproblem_util_block.constraint_list):
            dual_value = nlp.dual.get(subprob_constr, None)
            if (dual_value is None or
                generate_standard_repn(constr.body).is_linear()):
                continue

            # Determine if the user pre-specified that OA cuts should not be
            # generated for the given constraint.
            parent_block = constr.parent_block()
            ignore_set = getattr(parent_block, 'GDPopt_ignore_OA', None)
            config.logger.debug('Ignore_set %s' % ignore_set)
            if (ignore_set and (constr in ignore_set or
                                constr.parent_component() in ignore_set)):
                config.logger.debug(
                    'OA cut addition for %s skipped because it is in '
                    'the ignore set.' % constr.name)
                continue

            config.logger.debug( "Adding OA cut for %s with dual value %s" %
                                 (constr.name, dual_value))

            # Cache jacobian
            jacobian = master_util_block.jacobians.get(constr, None)
            if jacobian is None:
                constr_vars = list(identify_variables(constr.body,
                                                      include_fixed=False))
                if len(constr_vars) >= MAX_SYMBOLIC_DERIV_SIZE:
                    mode = differentiate.Modes.reverse_numeric
                else:
                    mode = differentiate.Modes.sympy

                try:
                    jac_list = differentiate( constr.body, wrt_list=constr_vars,
                                              mode=mode)
                    jac_map = ComponentMap(zip(constr_vars, jac_list))
                except:
                    if mode is differentiate.Modes.reverse_numeric:
                        raise
                    mode = differentiate.Modes.reverse_numeric
                    jac_map = ComponentMap()
                jacobian = JacInfo(mode=mode, vars=constr_vars, jac=jac_map)
                master_util_block.jacobians[constr] = jacobian
            # Recompute numeric derivatives
            if not jacobian.jac:
                jac_list = differentiate( constr.body, wrt_list=jacobian.vars,
                                          mode=jacobian.mode)
                jacobian.jac.update(zip(jacobian.vars, jac_list))

            # Create a block on which to put outer approximation cuts, if we
            # don't have one already on this parent block.
            oa_utils = oa_cut_blocks.get(parent_block)
            if oa_utils is None:
                nm = unique_component_name(parent_block, 'GDPopt_OA_cuts')
                oa_utils = Block(doc="Block holding outer approximation cuts "
                                 "and associated data.")
                parent_block.add_component(nm, oa_utils)
                oa_cut_blocks[parent_block] = oa_utils
                oa_utils.cuts = Constraint(NonNegativeIntegers)
            master_oa_utils = master_util_block.component('GDPopt_OA_slacks')
            if master_oa_utils is None:
                master_oa_utils = master_util_block.GDPopt_OA_slacks = Block(
                    doc="Block holding outer approximation slacks for the "
                    "whole model (so that the writers can find them).")
                master_oa_utils.slacks = VarList( bounds=(0, config.max_slack),
                                                  domain=NonNegativeReals,
                                                  initialize=0)

            oa_cuts = oa_utils.cuts
            slack_var = master_oa_utils.slacks.add()
            rhs = value(constr.lower) if constr.has_lb() else value(
                constr.upper)
            try:
                new_oa_cut = (
                    copysign(1, sign_adjust * dual_value) * (
                        value(constr.body) - rhs + sum(
                            value(jac) * (var - value(var))
                            for var, jac in jacobian.jac.items())
                        ) - slack_var <= 0)
                assert new_oa_cut.polynomial_degree() in (1, 0)
                idx = len(oa_cuts)
                oa_cuts[idx] = new_oa_cut
                _add_bigm_constraint_to_transformed_model(m, oa_cuts[idx],
                                                          oa_cuts)
                config.logger.debug("Cut expression: %s" % new_oa_cut)
                counter += 1
            except ZeroDivisionError:
                config.logger.warning(
                    "Zero division occured attempting to generate OA cut for "
                    "constraint %s.\n"
                    "Skipping OA cut generation for this constraint."
                    % (constr.name,)
                )
                # Simply continue on to the next constraint.
            # Clear out the numeric Jacobian values
            if jacobian.mode is differentiate.Modes.reverse_numeric:
                jacobian.jac.clear()

        config.logger.debug('Added %s OA cuts' % counter)

GDP_LOA_Solver.solve.__doc__ = add_docstring_list(
    GDP_LOA_Solver.solve.__doc__, GDP_LOA_Solver.CONFIG, indent_by=8)
