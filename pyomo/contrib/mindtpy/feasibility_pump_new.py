
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

TODO: test_FP_OA_8PP will fail. Need to fix.
"""
from __future__ import division
import logging
from pyomo.contrib.gdpopt.util import (time_code, lower_logger_level_to)
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
from pyomo.opt import TerminationCondition as tc
from pyomo.contrib.mindtpy.util import set_solver_options, get_integer_solution, update_suboptimal_dual_bound, copy_var_list_values_from_solution_pool, add_feas_slacks, add_var_bound, epigraph_reformulation
from pyomo.solvers.plugins.solvers.gurobi_direct import gurobipy
from operator import itemgetter
from pyomo.core import minimize, maximize, Objective, VarList, Reals, ConstraintList, Constraint, Block, TransformationFactory
from pyomo.contrib.mindtpy.feasibility_pump import generate_norm_constraint, fp_converged, add_orthogonality_cuts
from pyomo.contrib.mindtpy.util import generate_norm1_objective_function, generate_norm2sq_objective_function, generate_norm_inf_objective_function, generate_lag_objective_function, set_solver_options, GurobiPersistent4MindtPy, update_dual_bound, update_suboptimal_dual_bound
from pyomo.opt import SolverFactory, SolverResults, ProblemSense
from pyomo.contrib.gdpopt.util import (SuppressInfeasibleWarning, _DoNothing,
                                       copy_var_list_values, get_main_elapsed_time)
from pyomo.core import (ConstraintList, Objective,
                        TransformationFactory, maximize, minimize,
                        value, Var)
from pyomo.opt import SolutionStatus, SolverStatus

# 
__version__ = (0, 1, 0)
# TODO: 有问题，需要修复 test_FP_OA_8PP will fail,已解决，需要调用outer_approximation.py

@SolverFactory.register(
    'mindtpy.fp',
    doc='MindtPy: Mixed-Integer Nonlinear Decomposition Toolbox in Pyomo')
class MindtPy_FP_Solver(_MindtPyAlgorithm):
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
        config = self.config = self.CONFIG(kwds.pop('options', {}), preserve_implicit=True)  # TODO: do we need to set preserve_implicit=True?
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

            # Validate the model to ensure that MindtPy is able to solve it.
            if not self.model_is_valid():
                return

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
                                   move_objective=config.move_objective,
                                   use_mcpp=config.use_mcpp,
                                   update_var_con_list=True,
                                   partition_nonlinear_terms=config.partition_obj_nonlinear_terms,
                                   obj_handleable_polynomial_degree=self.mip_objective_polynomial_degree,
                                   constr_handleable_polynomial_degree=self.mip_constraint_polynomial_degree)

            # Save model initial values.
            self.initial_var_values = list(
                v.value for v in MindtPy.variable_list)

            # Initialize the main problem
            with time_code(self.timing, 'initialization'):
                self.MindtPy_initialize_main(config)

            # Load solution
            if self.best_solution_found is not None:
                self.load_solution()
            
            # Update result
            self.update_result()

            config.logger.info(' {:<25}:   {:>7.4f} '.format(
                'Primal-dual gap integral', self.results.solver.primal_dual_gap_integral))

        return self.results

    ################################################################################################################################
    # feasibility_pump.py

    def solve_fp_subproblem(self, config):
        """Solves the feasibility pump NLP subproblem.

        This function sets up the 'fp_nlp' by relax integer variables.
        precomputes dual values, deactivates trivial constraints, and then solves NLP model.

        Parameters
        ----------
        config : ConfigBlock
            The specific configurations for MindtPy.

        Returns
        -------
        fp_nlp : Pyomo model
            Fixed-NLP from the model.
        results : SolverResults
            Results from solving the fixed-NLP subproblem.
        """
        fp_nlp = self.working_model.clone()
        MindtPy = fp_nlp.MindtPy_utils

        # Set up NLP
        fp_nlp.MindtPy_utils.objective_list[-1].deactivate()
        if self.objective_sense == minimize:
            fp_nlp.improving_objective_cut = Constraint(
                expr=sum(fp_nlp.MindtPy_utils.objective_value[:]) <= self.primal_bound)
        else:
            fp_nlp.improving_objective_cut = Constraint(
                expr=sum(fp_nlp.MindtPy_utils.objective_value[:]) >= self.primal_bound)

        # Add norm_constraint, which guarantees the monotonicity of the norm objective value sequence of all iterations
        # Ref: Paper 'A storm of feasibility pumps for nonconvex MINLP'   https://doi.org/10.1007/s10107-012-0608-x
        # the norm type is consistant with the norm obj of the FP-main problem.
        if config.fp_norm_constraint:
            generate_norm_constraint(fp_nlp, self.mip, config)

        MindtPy.fp_nlp_obj = generate_norm2sq_objective_function(
            fp_nlp, self.mip, discrete_only=config.fp_discrete_only)

        MindtPy.cuts.deactivate()
        TransformationFactory('core.relax_integer_vars').apply_to(fp_nlp)
        try:
            TransformationFactory('contrib.deactivate_trivial_constraints').apply_to(
                fp_nlp, tmp=True, ignore_infeasible=False, tolerance=config.constraint_tolerance)
        except ValueError:
            config.logger.warning(
                'infeasibility detected in deactivate_trivial_constraints')
            results = SolverResults()
            results.solver.termination_condition = tc.infeasible
            return fp_nlp, results
        # Solve the NLP
        nlpopt = SolverFactory(config.nlp_solver)
        nlp_args = dict(config.nlp_solver_args)
        set_solver_options(nlpopt, self.timing, config, solver_type='nlp')
        with SuppressInfeasibleWarning():
            with time_code(self.timing, 'fp subproblem'):
                results = nlpopt.solve(fp_nlp,
                                    tee=config.nlp_solver_tee,
                                    load_solutions=False,
                                    **nlp_args)
                if len(results.solution) > 0:
                    fp_nlp.solutions.load_from(results)
        return fp_nlp, results


    def handle_fp_subproblem_optimal(self, fp_nlp, config):
        """Copies the solution to the working model, updates bound, adds OA cuts / no-good cuts /
        increasing objective cut, calculates the duals and stores incumbent solution if it has been improved.

        Parameters
        ----------
        fp_nlp : Pyomo model
            The feasibility pump NLP subproblem.
        config : ConfigBlock
            The specific configurations for MindtPy.
        """
        copy_var_list_values(
            fp_nlp.MindtPy_utils.variable_list,
            self.working_model.MindtPy_utils.variable_list,
            config,
            ignore_integrality=True)
        add_orthogonality_cuts(self.working_model, self.mip, config)

        # if OA-like or fp converged, update Upper bound,
        # add no_good cuts and increasing objective cuts (fp)
        if fp_converged(self.working_model, self.mip, config, discrete_only=config.fp_discrete_only):
            copy_var_list_values(self.mip.MindtPy_utils.variable_list,
                                self.working_model.MindtPy_utils.variable_list,
                                config)
            fixed_nlp, fixed_nlp_results = self.solve_subproblem(config)
            if fixed_nlp_results.solver.termination_condition in {tc.optimal, tc.locallyOptimal, tc.feasible}:
                self.handle_subproblem_optimal(fixed_nlp, config, fp=True)
            else:
                config.logger.error('Feasibility pump Fixed-NLP is infeasible, something might be wrong. '
                                    'There might be a problem with the precisions - the feasibility pump seems to have converged')


    def handle_fp_main_tc(self, feas_main_results, config):
        """Handle the termination condition of the feasibility pump main problem.

        Parameters
        ----------
        feas_main_results : SolverResults
            The results from solving the FP main problem.
        solve_data : MindtPySolveData
            Data container that holds solve-instance data.
        config : ConfigBlock
            The specific configurations for MindtPy.

        Returns
        -------
        bool
            True if FP loop should terminate, False otherwise.
        """
        if feas_main_results.solver.termination_condition is tc.optimal:
            config.logger.info(self.log_formatter.format(
                self.fp_iter, 'FP-MIP', value(
                    self.mip.MindtPy_utils.fp_mip_obj),
                self.primal_bound, self.dual_bound, self.rel_gap, get_main_elapsed_time(self.timing)))
            return False
        elif feas_main_results.solver.termination_condition is tc.maxTimeLimit:
            config.logger.warning('FP-MIP reaches max TimeLimit')
            self.results.solver.termination_condition = tc.maxTimeLimit
            return True
        elif feas_main_results.solver.termination_condition is tc.infeasible:
            config.logger.warning('FP-MIP infeasible')
            no_good_cuts = self.mip.MindtPy_utils.cuts.no_good_cuts
            if no_good_cuts.__len__() > 0:
                no_good_cuts[no_good_cuts.__len__()].deactivate()
            return True
        elif feas_main_results.solver.termination_condition is tc.unbounded:
            config.logger.warning('FP-MIP unbounded')
            return True
        elif (feas_main_results.solver.termination_condition is tc.other and
                feas_main_results.solution.status is SolutionStatus.feasible):
            config.logger.warning('MILP solver reported feasible solution of FP-MIP, '
                                'but not guaranteed to be optimal.')
            return False
        else:
            config.logger.warning('Unexpected result of FP-MIP')
            return True


    def fp_loop(self, config):
        """Feasibility pump loop.

        This is the outermost function for the algorithms in this package; this function
        controls the progression of solving the model.

        Parameters
        ----------
        solve_data : MindtPySolveData
            Data container that holds solve-instance data.
        config : ConfigBlock
            The specific configurations for MindtPy.

        Raises
        ------
        ValueError
            MindtPy unable to handle the termination condition of the FP-NLP subproblem.
        """
        while self.fp_iter < config.fp_iteration_limit:

            self.mip_subiter = 0
            # solve MILP main problem
            feas_main, feas_main_results = self.solve_main(config, fp=True)
            fp_should_terminate = self.handle_fp_main_tc(feas_main_results, config)
            if fp_should_terminate:
                break

            # Solve NLP subproblem
            # The constraint linearization happens in the handlers
            fp_nlp, fp_nlp_result = self.solve_fp_subproblem(config)

            if fp_nlp_result.solver.termination_condition in {tc.optimal, tc.locallyOptimal, tc.feasible}:
                config.logger.info(self.log_formatter.format(
                    self.fp_iter, 'FP-NLP', value(
                        fp_nlp.MindtPy_utils.fp_nlp_obj),
                    self.primal_bound, self.dual_bound, self.rel_gap,
                    get_main_elapsed_time(self.timing)))
                self.handle_fp_subproblem_optimal(fp_nlp, config)
            elif fp_nlp_result.solver.termination_condition in {tc.infeasible, tc.noSolution}:
                config.logger.error('Feasibility pump NLP subproblem infeasible')
                self.should_terminate = True
                self.results.solver.status = SolverStatus.error
                return
            elif fp_nlp_result.solver.termination_condition is tc.maxIterations:
                config.logger.error(
                    'Feasibility pump NLP subproblem failed to converge within iteration limit.')
                self.should_terminate = True
                self.results.solver.status = SolverStatus.error
                return
            else:
                raise ValueError(
                    'MindtPy unable to handle NLP subproblem termination '
                    'condition of {}'.format(fp_nlp_result.solver.termination_condition))
            # Call the NLP post-solve callback
            # TODO fix bug
            config.call_after_subproblem_solve(fp_nlp)
            self.fp_iter += 1
        self.mip.MindtPy_utils.del_component('fp_mip_obj')

        if config.fp_main_norm == 'L1':
            self.mip.MindtPy_utils.del_component('L1_obj')
        elif config.fp_main_norm == 'L_infinity':
            self.mip.MindtPy_utils.del_component(
                'L_infinity_obj')

        # deactivate the improving_objective_cut
        self.mip.MindtPy_utils.cuts.del_component(
            'improving_objective_cut')
        if not config.fp_transfercuts:
            for c in self.mip.MindtPy_utils.cuts.oa_cuts:
                c.deactivate()
            for c in self.mip.MindtPy_utils.cuts.no_good_cuts:
                c.deactivate()
        if config.fp_projcuts:
            self.working_model.MindtPy_utils.cuts.del_component(
                'fp_orthogonality_cuts')


    #
    # Support 'with' statements.
    #
    def __enter__(self):
        return self

    def __exit__(self, t, v, traceback):
        pass
