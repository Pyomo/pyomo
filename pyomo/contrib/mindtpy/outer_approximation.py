
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

TODO: mindtpy rewrite
"""
from __future__ import division
import logging
from pyomo.contrib.gdpopt.util import (copy_var_list_values, 
                                       time_code, lower_logger_level_to)
from pyomo.contrib.mindtpy.initialization import MindtPy_initialize_main
from pyomo.contrib.mindtpy.iterate import MindtPy_iteration_loop
from pyomo.contrib.mindtpy.util import model_is_valid, set_up_solve_data, set_up_logger, get_primal_integral, get_dual_integral, setup_results_object, process_objective, create_utility_block
from pyomo.core import (Block, ConstraintList, NonNegativeReals,
                        Var, VarList, TransformationFactory, RangeSet, minimize, Constraint, Objective)
from pyomo.opt import SolverFactory
from pyomo.contrib.mindtpy.config_options import _get_MindtPy_config, check_config
from pyomo.common.config import add_docstring_list
from pyomo.util.vars_from_expressions import get_vars_from_components
from algorithm_base_class import _MindtPyAlgorithm


__version__ = (0, 1, 0)


@SolverFactory.register(
    'mindtpy',
    doc='MindtPy: Mixed-Integer Nonlinear Decomposition Toolbox in Pyomo')
class MindtPySolver(_MindtPyAlgorithm):
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

    This solver implementation has been developed by David Bernal <https://github.com/bernalde>
    and Zedong Peng <https://github.com/ZedongPeng> as part of research efforts at the Grossmann
    Research Group (http://egon.cheme.cmu.edu/) at the Department of Chemical Engineering at 
    Carnegie Mellon University.
    """
    CONFIG = _get_MindtPy_config()

    def available(self, exception_flag=True):
        """Check if solver is available.
        """
        return True

    def license_is_valid(self):
        return True

    def version(self):
        """Return a 3-tuple describing the solver version."""
        return __version__

    def solve(self, model, **kwds):
        """Solve the model.

        Parameters
        ----------
        model : Pyomo model
            The MINLP model to be solved.

        Returns
        -------
        results : SolverResults
            Results from solving the MINLP problem by MindtPy.
        """
        config = self.config = self.CONFIG(kwds.pop('options', {
        }), preserve_implicit=True)  # TODO: do we need to set preserve_implicit=True?
        config.set_value(kwds)
        set_up_logger(config)
        new_logging_level = logging.INFO if config.tee else None
        with lower_logger_level_to(config.logger, new_logging_level):
            check_config(config)

        self.set_up_solve_data(model, config)

        if config.integer_to_binary:
            TransformationFactory('contrib.integer_to_binary'). \
                apply_to(self.working_model)

        self.create_utility_block(self.working_model, 'MindtPy_utils')
        with time_code(self.timing, 'total', is_main_timer=True), \
                lower_logger_level_to(config.logger, new_logging_level):
            config.logger.info(
                '---------------------------------------------------------------------------------------------\n'
                '              Mixed-Integer Nonlinear Decomposition Toolbox in Pyomo (MindtPy)               \n'
                '---------------------------------------------------------------------------------------------\n'
                'For more information, please visit https://pyomo.readthedocs.io/en/stable/contributed_packages/mindtpy.html')

            MindtPy = self.working_model.MindtPy_utils
            setup_results_object(self.results, self.original_model, config)
            # In the process_objective function, as long as the objective function is nonlinear, it will be reformulated and the variable/constraint/objective lists will be updated.
            # For OA/GOA/LP-NLP algorithm, if the objective funtion is linear, it will not be reformulated as epigraph constraint.
            # If the objective function is linear, it will be reformulated as epigraph constraint only if the Feasibility Pump or ROA/RLP-NLP algorithm is activated. (move_objective = True)
            # In some cases, the variable/constraint/objective lists will not be updated even if the objective is epigraph-reformulated.
            # In Feasibility Pump, since the distance calculation only includes discrete variables and the epigraph slack variables are continuous variables, the Feasibility Pump algorithm will not affected even if the variable list are updated.
            # In ROA and RLP/NLP, since the distance calculation does not include these epigraph slack variables, they should not be added to the variable list. (update_var_con_list = False)
            # In the process_objective function, once the objective function has been reformulated as epigraph constraint, the variable/constraint/objective lists will not be updated only if the MINLP has a linear objective function and regularization is activated at the same time.
            # This is because the epigraph constraint is very "flat" for branching rules. The original objective function will be used for the main problem and epigraph reformulation will be used for the projection problem.
            # TODO: The logic here is too complicated, can we simplify it?
            self.process_objective(config,
                                   move_objective=(config.init_strategy == 'FP'
                                                   or config.add_regularization is not None
                                                   or config.move_objective),
                                   use_mcpp=config.use_mcpp,
                                   update_var_con_list=config.add_regularization is None,
                                   partition_nonlinear_terms=config.partition_obj_nonlinear_terms,
                                   obj_handleable_polynomial_degree=self.mip_objective_polynomial_degree,
                                   constr_handleable_polynomial_degree=self.mip_constraint_polynomial_degree
                                   )
            # The epigraph constraint is very "flat" for branching rules.
            # If ROA/RLP-NLP is activated and the original objective function is linear, we will use the original objective for the main mip.
            if MindtPy.objective_list[0].expr.polynomial_degree() in self.mip_objective_polynomial_degree and config.add_regularization is not None:
                MindtPy.objective_list[0].activate()
                MindtPy.objective_constr.deactivate()
                MindtPy.objective.deactivate()

            # Save model initial values.
            self.initial_var_values = list(
                v.value for v in MindtPy.variable_list)

            # Store the initial model state as the best solution found. If we
            # find no better solution, then we will restore from this copy.
            self.best_solution_found = None
            self.best_solution_found_time = None

            # Record solver name
            self.results.solver.name = 'MindtPy' + str(config.strategy)

            # Validate the model to ensure that MindtPy is able to solve it.
            if not self.model_is_valid():
                return

            # Create a model block in which to store the generated feasibility
            # slack constraints. Do not leave the constraints on by default.
            feas = MindtPy.feas_opt = Block()
            feas.deactivate()
            feas.feas_constraints = ConstraintList(
                doc='Feasibility Problem Constraints')

            # Create a model block in which to store the generated linear
            # constraints. Do not leave the constraints on by default.
            lin = MindtPy.cuts = Block()
            lin.deactivate()

            # no-good cuts exclude particular discrete decisions
            lin.no_good_cuts = ConstraintList(doc='no-good cuts')
            # Feasible no-good cuts exclude discrete realizations that have
            # been explored via an NLP subproblem. Depending on model
            # characteristics, the user may wish to revisit NLP subproblems
            # (with a different initialization, for example). Therefore, these
            # cuts are not enabled by default.
            #
            # Note: these cuts will only exclude integer realizations that are
            # not already in the primary no_good_cuts ConstraintList.
            lin.feasible_no_good_cuts = ConstraintList(
                doc='explored no-good cuts')
            lin.feasible_no_good_cuts.deactivate()

            if config.feasibility_norm == 'L1' or config.feasibility_norm == 'L2':
                feas.nl_constraint_set = RangeSet(len(MindtPy.nonlinear_constraint_list),
                                                  doc='Integer index set over the nonlinear constraints.')
                # Create slack variables for feasibility problem
                feas.slack_var = Var(feas.nl_constraint_set,
                                     domain=NonNegativeReals, initialize=1)
            else:
                feas.slack_var = Var(domain=NonNegativeReals, initialize=1)

            # Create slack variables for OA cuts
            if config.add_slack:
                lin.slack_vars = VarList(
                    bounds=(0, config.max_slack), initialize=0, domain=NonNegativeReals)

            # Initialize the main problem
            with time_code(self.timing, 'initialization'):
                self.MindtPy_initialize_main(config)

            # Algorithm main loop
            with time_code(self.timing, 'main loop'):
                self.MindtPy_iteration_loop(config)
            if self.best_solution_found is not None:
                # Update values in original model
                copy_var_list_values(
                    from_list=self.best_solution_found.MindtPy_utils.variable_list,
                    to_list=MindtPy.variable_list,
                    config=config)
                # The original does not have variable list. Use get_vars_from_components() should be used for both working_model and original_model to exclude the unused variables.
                self.working_model.MindtPy_utils.deactivate()
                if self.working_model.find_component("_int_to_binary_reform") is not None:
                    self.working_model._int_to_binary_reform.deactivate()
                copy_var_list_values(list(get_vars_from_components(block=self.working_model, 
                                         ctype=(Constraint, Objective), 
                                         include_fixed=False, 
                                         active=True,
                                         sort=True, 
                                         descend_into=True,
                                         descent_order=None)),
                                    list(get_vars_from_components(block=self.original_model, 
                                         ctype=(Constraint, Objective), 
                                         include_fixed=False, 
                                         active=True,
                                         sort=True, 
                                         descend_into=True,
                                         descent_order=None)),
                                    config=config)
                # exclude fixed variables here. This is consistent with the definition of variable_list in GDPopt.util
            if self.objective_sense == minimize:
                self.results.problem.lower_bound = self.dual_bound
                self.results.problem.upper_bound = self.primal_bound
            else:
                self.results.problem.lower_bound = self.primal_bound
                self.results.problem.upper_bound = self.dual_bound

            self.results.solver.timing = self.timing
            self.results.solver.user_time = self.timing.total
            self.results.solver.wallclock_time = self.timing.total
            self.results.solver.iterations = self.mip_iter
            self.results.solver.num_infeasible_nlp_subproblem = self.nlp_infeasible_counter
            self.results.solver.best_solution_found_time = self.best_solution_found_time
            self.results.solver.primal_integral = self.get_primal_integral()
            self.results.solver.dual_integral = self.get_dual_integral()
            self.results.solver.primal_dual_gap_integral = self.results.solver.primal_integral + \
                self.results.solver.dual_integral
            config.logger.info(' {:<25}:   {:>7.4f} '.format(
                'Primal-dual gap integral', self.results.solver.primal_dual_gap_integral))

            if config.single_tree:
                self.results.solver.num_nodes = self.nlp_iter - \
                    (1 if config.init_strategy == 'rNLP' else 0)

        return self.results

    #
    # Support 'with' statements.
    #
    def __enter__(self):
        return self

    def __exit__(self, t, v, traceback):
        pass
