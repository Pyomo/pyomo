# -*- coding: utf-8 -*-
"""Implementation of the MindtPy solver.

The MindtPy (MINLP Decomposition Tookit) solver applies a variety of
decomposition-based approaches to solve nonlinear continuous-discrete problems.
These approaches include:

- Outer approximation
- Benders decomposition
- Partial surrogate cuts
- Extended cutting plane

This solver implementation was developed by Carnegie Mellon University in the
research group of Ignacio Grossmann.
For nonconvex problems, the bounds solve_data.LB and solve_data.UB may not be rigorous.
Questions: Please make a post at StackOverflow and/or David Bernal <https://github.com/bernalde>

"""
from __future__ import division

import logging
from copy import deepcopy
from math import copysign,fabs
from pprint import pprint

from six import iteritems

from pyutilib.misc.config import ConfigBlock, ConfigValue
from pyomo.contrib.mindtpy.util import MindtPySolveData, _DoNothing, a_logger
from pyomo.contrib.mindtpy.util import (
    build_ordered_component_lists, model_is_valid, copy_values)
from pyomo.contrib.mindtpy.initialization import MindtPy_initialize_master
from pyomo.contrib.mindtpy.initialization import init_rNLP, init_max_binaries
import pyomo.common.plugin
from pyomo.core.base import (Block, ComponentUID, Constraint, ConstraintList,
                             Expression, Objective, RangeSet, Set, Suffix, Var,
                             maximize, minimize, value)
from pyomo.contrib.mindtpy.solve_NLP import solve_NLP_subproblem
from pyomo.core.expr.current import clone_expression
from pyomo.core.base.numvalue import NumericConstant
from pyomo.core.kernel import Binary, NonNegativeReals, Reals
from pyomo.opt import TerminationCondition as tc
from pyomo.opt import SolverFactory, SolverStatus
from pyomo.opt.base import IOptSolver
from pyomo.opt.results import ProblemSense, SolverResults

from pyomo.core import TransformationFactory
from pyomo.core.base import ComponentMap

from six.moves import range

logger = logging.getLogger('pyomo.contrib.mindtpy')

__version__ = (0, 0, 1)


def _positiveFloat(val):
    ans = float(val)
    if ans <= 0:
        raise ValueError("Expected positive float (got %s)" % (val,))
    return ans


def _positiveInt(val):
    ans = int(val)
    if ans <= 0:
        raise ValueError("Expected positive int (got %s)" % (val,))
    return ans


class _In(object):

    def __init__(self, allowable):
        self.allowable = allowable

    def __call__(self, value):
        if value in self.allowable:
            return value
        raise ValueError("%s not in %s" % (value, self._allowable))


class MindtPySolver(pyomo.common.plugin.Plugin):
    """A decomposition-based MINLP solver.
    """

    pyomo.common.plugin.implements(IOptSolver)
    pyomo.common.plugin.alias('mindtpy',
                            doc='MindtPy: Mixed-Integer Nonlinear Decomposition Toolbox in Pyomo')

    # _metasolver = False

    CONFIG = ConfigBlock("MindtPy")
    CONFIG.declare("bound_tolerance", ConfigValue(
        default=1E-5,
        domain=_positiveFloat,
        description="Bound tolerance",
        doc="Relative tolerance for bound feasibility checks"
    ))
    CONFIG.declare("iteration_limit", ConfigValue(
        default=30,
        domain=_positiveInt,
        description="Iteration limit",
        doc="Number of maximum iterations in the decomposition methods"
    ))
    CONFIG.declare("strategy", ConfigValue(
        default="OA",
        domain=_In(["OA","GBD","ECP","PSC"]),
        description="Decomposition strategy",
        doc="MINLP Decomposition strategy to be applied to the method. "
            "Currently available Outer Approximation (OA), Extended Cutting "            'Plane (ECP), Partial Surrogate Cuts (PSC), and Generalized '
            "Benders Decomposition (GBD)"
    ))
    CONFIG.declare("init_strategy", ConfigValue(
        default="rNLP",
        domain=_In(["rNLP","initial_binary","max_binary"]),
        description="Initialization strategy",
        doc="Initialization strategy used by any method. Currently the "
            "continuous relaxation of the MINLP (rNLP), solve a maximal "
            "covering problem (max_binary), and fix the initial value for "
            "the integer variables (initial_binary)"
    ))
    CONFIG.declare("integer_cuts", ConfigValue(
        default=True,
        domain=bool,
        description="Integer cuts",
        doc="Add integer cuts after finding a feasible solution to the MINLP"
    ))
    CONFIG.declare("max_slack", ConfigValue(
        default=1000.0,
        domain=_positiveFloat,
        description="Maximum slack variable",
        doc="Maximum slack variable value allowed for the Outer Approximation "
            "cuts"
    ))
    CONFIG.declare("OA_penalty_factor", ConfigValue(
        default=1000.0,
        domain=_positiveFloat,
        description="Outer Approximation slack penalty factor",
        doc="In the objective function of the Outer Approximation method, the "
            "slack variables correcponding to all the constraints get "
            "multiplied by this number and added to the objective"
    ))
    CONFIG.declare("ECP_tolerance", ConfigValue(
        default=1E-4,
        domain=_positiveFloat,
        description="ECP tolerance",
        doc="Feasibility tolerance used to determine the stopping criterion in"
            "the ECP method. As long as nonlinear constraint are violated for "
            "more than this tolerance, the mothod will keep iterating"
    ))
    CONFIG.declare("nlp_solver", ConfigValue(
        default="ipopt",
        domain=_In(["ipopt"]),
        description="NLP subsolver name",
        doc="Which NLP subsolver is going to be used for solving the nonlinear"
            "subproblems"
    ))
    CONFIG.declare("nlp_solver_kwargs", ConfigBlock(
        implicit=True,
        description="NLP subsolver options",
        doc="Which NLP subsolver options to be passed to the solver while "
            "solving the nonlinear subproblems"
    ))
    CONFIG.declare("mip_solver", ConfigValue(
        default="gurobi",
        domain=_In(["gurobi", "cplex", "cbc", "glpk"]),
        description="MIP subsolver name",
        doc="Which MIP subsolver is going to be used for solving the mixed-"
            "integer master problems"
    ))
    CONFIG.declare("mip_solver_kwargs", ConfigBlock(
        implicit=True,
        description="MIP subsolver options",
        doc="Which MIP subsolver options to be passed to the solver while "
            "solving the mixed-integer master problems"
    ))
    CONFIG.declare("modify_in_place", ConfigValue(
        default=True,
        domain=bool,
        description="Solve subproblems directly upon the model",
        doc="If true, MindtPy manipulations are performed directly upon "
            "the model. Otherwise, the model is first copied and solution "
            "values are copied over afterwards."
    ))
    CONFIG.declare("master_postsolve", ConfigValue(
        default=_DoNothing(),
        domain=None,
        description="Function to be executed after every master problem",
        doc="Callback hook after a solution of the master problem."
    ))
    CONFIG.declare("subproblem_postsolve", ConfigValue(
        default=_DoNothing(),
        domain=None,
        description="Function to be executed after every subproblem",
        doc="Callback hook after a solution of the nonlinear subproblem."
    ))
    CONFIG.declare("subproblem_postfeasible", ConfigValue(
        default=_DoNothing(),
        domain=None,
        description="Function to be executed after every feasible subproblem",
        doc="Callback hook after a feasible solution of the nonlinear subproblem."
    ))
    CONFIG.declare("load_solutions", ConfigValue(
        default=True,
        domain=bool,
        description="Solve subproblems directly upon the model",
        doc="if True, load solutions back into the model."
            "This is only relevant if solve_in_place is not True."
    ))
    CONFIG.declare("tee", ConfigValue(
        default=False,
        description="Stream output to terminal.",
        domain=bool
    ))
    CONFIG.declare("logger", ConfigValue(
        default='pyomo.contrib.mindtpy',
        description="The logger object or name to use for reporting.",
        domain=a_logger
    ))
    CONFIG.declare("small_dual_tolerance", ConfigValue(
        default=1E-8,
        description="When generating cuts, small duals multiplied "
        "by expressions can cause problems. Exclude all duals "
        "smaller in absolue value than the following."
    ))
    CONFIG.declare("integer_tolerance", ConfigValue(
        default=1E-5,
        description="Tolerance on integral values."
    ))
    CONFIG.declare("constraint_tolerance", ConfigValue(
        default=1E-6,
        description="Tolerance on constraint satisfaction."
    ))
    CONFIG.declare("variable_tolerance", ConfigValue(
        default=1E-8,
        description="Tolerance on variable bounds."
    ))
    CONFIG.declare("zero_tolerance", ConfigValue(
        default=1E-15,
        description="Tolerance on variable equal to zero."))
    CONFIG.declare("initial_feas", ConfigValue(
        default=True,
        description="Apply an initial feasibility step.",
        domain=bool
    ))

    # From Qi: this causes issues.
    # I'm not sure exactly why, but commenting for now.
    # __doc__ += CONFIG.generate_yaml_template()

    def available(self, exception_flag=True):
        """Check if solver is available.
        TODO: For now, it is always available. However, sub-solvers may not
        always be available, and so this should reflect that possibility.
        """
        return True

    def version(self):
        """Return a 3-tuple describing the solver version."""
        return __version__

    def solve(self, model, **kwds):
        """Solve the model.
        Warning: this solver is still in beta. Keyword arguments subject to
        change. Undocumented keyword arguments definitely subject to change.
        Warning: at this point in time, if you try to use PSC or GBD with
        anything other than IPOPT as the NLP solver, bad things will happen.
        This is because the suffixes are not in place to extract dual values
        from the variable bounds for any other solver.
        TODO: fix needed with the GBD implementation.
        Args:
            model (Block): a Pyomo model or block to be solved
        """
        config = self.CONFIG(kwds.pop('options', {}))
        config.set_value(kwds)
        solve_data = MindtPySolveData()
        created_MindtPy_block = False


        old_logger_level = config.logger.getEffectiveLevel()
        try:
            if config.tee and old_logger_level > logging.INFO:
                # If the logger does not already include INFO, include it.
                config.logger.setLevel(logging.INFO)
            config.logger.info("Starting MindtPy")

            # Create a model block on which to store MindtPy-specific utility
            # modeling objects.
            if hasattr(model, 'MindtPy_utils'):
                raise RuntimeError("MindtPy_utils already exists.")
            else:
                created_MindtPy_block = True
                model.MindtPy_utils = Block()

            solve_data.original_model = model

            build_ordered_component_lists(model)
            solve_data.working_model = model.clone()
            MindtPy = solve_data.working_model.MindtPy_utils

            # Store the initial model state as the best solution found. If we
            # find no better solution, then we will restore from this copy.
            solve_data.best_solution_found = model.clone()

            # Create the solver results object
            res = solve_data.results = SolverResults()
            res.problem.name = model.name
            res.problem.number_of_nonzeros = None  # TODO
            res.solver.name = 'MindtPy' + str(config.strategy)
            # TODO work on termination condition and message
            res.solver.termination_condition = None
            res.solver.message = None
            # TODO add some kind of timing
            res.solver.user_time = None
            res.solver.system_time = None
            res.solver.wallclock_time = None
            res.solver.termination_message = None

            # Validate the model to ensure that MindtPy is able to solve it.
            if not model_is_valid(solve_data, config):
                return

            # Create a model block in which to store the generated feasibility slack
            # constraints. Do not leave the constraints on by default.
            feas = MindtPy.MindtPy_feas = Block()
            feas.deactivate()
            feas.feas_constraints = ConstraintList(
                doc='Feasibility Problem Constraints')

            # Create a model block in which to store the generated linear
            # constraints. Do not leave the constraints on by default.
            lin = MindtPy.MindtPy_linear_cuts = Block()
            lin.deactivate()

            # Integer cuts exclude particular discrete decisions
            lin.integer_cuts = ConstraintList(doc='integer cuts')
            # Feasible integer cuts exclude discrete realizations that have been
            # explored via an NLP subproblem. Depending on model characteristics,
            # the user may wish to revisit NLP subproblems (with a different
            # initialization, for example). Therefore, these cuts are not enabled
            # by default.
            #
            # Note: these cuts will only exclude integer realizations that are not
            # already in the primary integer_cuts ConstraintList.
            lin.feasible_integer_cuts = ConstraintList(doc='explored integer cuts')
            lin.feasible_integer_cuts.deactivate()

            # Set up iteration counters
            solve_data.nlp_iter = 0
            solve_data.mip_iter = 0
            solve_data.mip_subiter = 0

            # set up bounds
            solve_data.LB = float('-inf')
            solve_data.UB = float('inf')
            solve_data.LB_progress = [solve_data.LB]
            solve_data.UB_progress = [solve_data.UB]

            # Set of NLP iterations for which cuts were generated
            lin.nlp_iters = Set(dimen=1)

            # Set of MIP iterations for which cuts were generated in ECP
            lin.mip_iters = Set(dimen=1)

            lin.nl_constraint_set = RangeSet(
                len(MindtPy.nonlinear_constraints),
                doc="Integer index set over the nonlinear constraints")
            feas.constraint_set = RangeSet(
                len(MindtPy.constraints),
                doc="integer index set over the constraints")
            # Mapping Constraint -> integer index
            MindtPy.nl_map = {}
            # Mapping integer index -> Constraint
            MindtPy.nl_inverse_map = {}
            # Generate the two maps. These maps may be helpful for later
            # interpreting indices on the slack variables or generated cuts.
            for c, n in zip(MindtPy.nonlinear_constraints,
                            lin.nl_constraint_set):
                MindtPy.nl_map[c] = n
                MindtPy.nl_inverse_map[n] = c

            # Mapping Constraint -> integer index
            MindtPy.feas_map = {}
            # Mapping integer index -> Constraint
            MindtPy.feas_inverse_map = {}
            # Generate the two maps. These maps may be helpful for later
            # interpreting indices on the slack variables or generated cuts.
            for c, n in zip(MindtPy.constraints, feas.constraint_set):
                MindtPy.feas_map[c] = n
                MindtPy.feas_inverse_map[n] = c

            # Create slack variables for OA cuts
            lin.slack_vars = Var(lin.nlp_iters, lin.nl_constraint_set,
                                 domain=NonNegativeReals,
                                 bounds=(0, config.max_slack), initialize=0)
            # Create slack variables for feasibility problem
            feas.slack_var = Var(feas.constraint_set,
                                 domain=NonNegativeReals, initialize=1)

            # Flag indicating whether the solution improved in the past
            # iteration or not
            solve_data.solution_improved = False

            # Set up solvers
            solve_data.nlp_solver = SolverFactory(config.nlp_solver)
            solve_data.mip_solver = SolverFactory(config.mip_solver)

            # Initialize the master problem
            MindtPy_initialize_master(solve_data, config)

            # Algorithm main loop
            self._MindtPy_iteration_loop(solve_data, config)

            # Update values in original model
            if config.load_solutions:
                # Update values in original model
                copy_values(
                    solve_data.best_solution_found,
                    solve_data.original_model,
                    config)

        finally:
            config.logger.setLevel(old_logger_level)
            if created_MindtPy_block:
                model.del_component('MindtPy_utils')

    def _MindtPy_iteration_loop(self, solve_data, config):
        m = solve_data.working_model
        MindtPy = m.MindtPy_utils
        # Backup counter to prevent infinite loop
        backup_max_iter = max(1000, config.iteration_limit)
        backup_iter = 0
        while backup_iter < backup_max_iter:
            config.logger.info('')  # print blank lines for visual display
            backup_iter += 1
            # Check bound convergence
            if solve_data.LB + config.bound_tolerance >= solve_data.UB:
                print('MindtPy exiting on bound convergence. '
                      'LB: {} + (tol {}) >= UB: {}'.format(
                          solve_data.LB, config.bound_tolerance, solve_data.UB) + '\n')
                # res.solver.termination_condition = tc.optimal
                break
            # Check iteration limit
            if solve_data.mip_iter >= config.iteration_limit:
                print('MindtPy unable to converge bounds '
                      'after {} master iterations.'.format(solve_data.mip_iter))
                print('Final bound values: LB: {}  UB: {}'.
                      format(solve_data.LB, solve_data.UB))
                break
            solve_data.mip_subiter = 0
            # solve MILP master problem
            if config.strategy == 'OA':
                self._solve_OA_master(solve_data, config)
            elif config.strategy == 'PSC':
                self._solve_PSC_master(solve_data, config)
            elif config.strategy == 'GBD':
                self._solve_GBD_master(solve_data, config)
            elif config.strategy == 'ECP':
                self._solve_ECP_master(solve_data, config)
            # Check bound convergence
            if solve_data.LB + config.bound_tolerance >= solve_data.UB:
                print('MindtPy exiting on bound convergence. '
                      'LB: {} + (tol {}) >= UB: {}'.format(
                          solve_data.LB, config.bound_tolerance, solve_data.UB))
                break
            elif config.strategy == 'ECP':
                # Add ECP cut
                self._add_ecp_cut(solve_data, config)
            else:
                # Solve NLP subproblem
                solve_NLP_subproblem(solve_data, config)

            # If the hybrid algorithm is not making progress, switch to OA.
            progress_required = 1E-6
            if MindtPy.objective.sense == minimize:
                log = solve_data.LB_progress
                sign_adjust = 1
            else:
                log = solve_data.UB_progress
                sign_adjust = -1
            # Maximum number of iterations in which the lower (optimistic)
            # bound does not improve before switching to OA
            max_nonimprove_iter = 5
            making_progress = True
            for i in range(1, max_nonimprove_iter + 1):
                try:
                    if (sign_adjust * log[-i]
                            <= (log[-i - 1] + progress_required)
                            * sign_adjust):
                        making_progress = False
                    else:
                        making_progress = True
                        break
                except IndexError:
                    # Not enough history yet, keep going.
                    making_progress = True
                    break
            if not making_progress and (
                    config.strategy == 'hPSC' and
                    config.strategy == 'PSC'):
                print('Not making enough progress for {} iterations. '
                      'Switching to OA.'.format(max_nonimprove_iter))
                config.strategy = 'OA'

    def _solve_OA_master(self, solve_data, config):
        solve_data.mip_iter += 1
        m = solve_data.working_model.clone()
        MindtPy = m.MindtPy_utils
        config.logger.info(
            'MIP %s: Solve master problem.' %
            (solve_data.mip_iter,))
        # Set up MILP
        for c in MindtPy.nonlinear_constraints:
            c.deactivate()

        MindtPy.MindtPy_linear_cuts.activate()
        MindtPy.objective.deactivate()

        sign_adjust = 1 if MindtPy.objective.sense == minimize else -1
        MindtPy.MindtPy_penalty_expr = Expression(
            expr=sign_adjust * config.OA_penalty_factor * sum(
                v for v in MindtPy.MindtPy_linear_cuts.slack_vars[...]))

        MindtPy.MindtPy_oa_obj = Objective(
            expr=MindtPy.objective.expr + MindtPy.MindtPy_penalty_expr,
            sense=MindtPy.objective.sense)

        # Deactivate extraneous IMPORT/EXPORT suffixes
        getattr(m, 'ipopt_zL_out', _DoNothing()).deactivate()
        getattr(m, 'ipopt_zU_out', _DoNothing()).deactivate()

        # m.pprint() #print oa master problem for debugging
        results = solve_data.mip_solver.solve(m, load_solutions=False,
                                        options=config.mip_solver_kwargs)
        master_terminate_cond = results.solver.termination_condition
        if master_terminate_cond is tc.infeasibleOrUnbounded:
            # Linear solvers will sometimes tell me that it's infeasible or
            # unbounded during presolve, but fails to distinguish. We need to
            # resolve with a solver option flag on.
            from copy import deepcopy
            old_options = deepcopy(solve_data.mip_solver.options)
            # This solver option is specific to Gurobi.
            solve_data.mip_solver.options['DualReductions'] = 0
            results = solve_data.mip_solver.solve(m, load_solutions=False,
                                            options=config.mip_solver_kwargs)
            master_terminate_cond = results.solver.termination_condition
            solve_data.mip_solver.options.update(old_options)

        MindtPy.objective.activate()
        for c in MindtPy.nonlinear_constraints:
            c.activate()
        MindtPy.MindtPy_linear_cuts.deactivate()
        MindtPy.MindtPy_oa_obj.deactivate()
        getattr(m, 'ipopt_zL_out', _DoNothing()).activate()
        getattr(m, 'ipopt_zU_out', _DoNothing()).activate()

        # Process master problem result
        if master_terminate_cond is tc.optimal:
            # proceed. Just need integer values
            m.solutions.load_from(results)
            copy_values(m, solve_data.working_model, config)

            if MindtPy.objective.sense == minimize:
                solve_data.LB = max(value(MindtPy.MindtPy_oa_obj.expr), solve_data.LB)
                solve_data.LB_progress.append(solve_data.LB)
            else:
                solve_data.UB = min(value(MindtPy.MindtPy_oa_obj.expr), solve_data.UB)
                solve_data.UB_progress.append(solve_data.UB)
            config.logger.info(
                'MIP %s: OBJ: %s  LB: %s  UB: %s'
                % (solve_data.mip_iter, value(MindtPy.MindtPy_oa_obj.expr),
                   solve_data.LB, solve_data.UB))
        elif master_terminate_cond is tc.infeasible:
            print('MILP master problem is infeasible. '
                  'Problem may have no more feasible binary combinations.')
            if solve_data.mip_iter == 1:
                print('MindtPy initialization may have generated poor '
                      'quality cuts.')
        elif master_terminate_cond is tc.maxTimeLimit:
            # TODO check that status is actually ok and everything is feasible
            config.logger.info(
                'Unable to optimize MILP master problem '
                'within time limit. '
                'Using current solver feasible solution.')
            results.solver.status = SolverStatus.ok
            m.solutions.load_from(results)
            self._copy_values(m, solve_data.working_model)
            if MindtPy.obj.sense == minimize:
                solve_data.LB = max(
                    value(MindtPy.obj.expr), solve_data.LB)
                solve_data.LB_progress.append(solve_data.LB)
            else:
                solve_data.UB = min(
                    value(MindtPy.obj.expr), solve_data.UB)
                solve_data.UB_progress.append(solve_data.UB)
            config.logger.info(
                'MIP %s: OBJ: %s  LB: %s  UB: %s'
                % (solve_data.mip_iter, value(MindtPy.obj.expr),
                   solve_data.LB, solve_data.UB))
        elif (master_terminate_cond is tc.other and
                results.solution.status is SolutionStatus.feasible):
            # load the solution and suppress the warning message by setting
            # solver status to ok.
            config.logger.info(
                'MILP solver reported feasible solution, '
                'but not guaranteed to be optimal.')
            results.solver.status = SolverStatus.ok
            m.solutions.load_from(results)
            self._copy_values(m, solve_data.working_model)
            if MindtPy.obj.sense == minimize:
                solve_data.LB = max(value(MindtPy.MindtPy_oa_obj.expr), solve_data.LB)
                solve_data.LB_progress.append(solve_data.LB)
            else:
                solve_data.UB = min(value(MindtPy.MindtPy_oa_obj.expr), solve_data.UB)
                solve_data.UB_progress.append(solve_data.UB)
            config.logger.info(
                'MIP %s: OBJ: %s  LB: %s  UB: %s'
                % (solve_data.mip_iter, value(MindtPy.MindtPy_oa_obj.expr),
                   solve_data.LB, solve_data.UB))
        elif master_terminate_cond is tc.infeasible:
            config.logger.info(
                'MILP master problem is infeasible. '
                'Problem may have no more feasible '
                'binary configurations.')
            if solve_data.mip_iter == 1:
                config.logger.warn(
                    'MindtPy initialization may have generated poor '
                    'quality cuts.')
            # set optimistic bound to infinity
            if MindtPy.obj.sense == minimize:
                solve_data.LB = float('inf')
                solve_data.LB_progress.append(solve_data.UB)
            else:
                solve_data.UB = float('-inf')
                solve_data.UB_progress.append(solve_data.UB)
        else:
            raise ValueError(
                'MindtPy unable to handle MILP master termination condition '
                'of %s. Solver message: %s' %
                (master_terminate_cond, results.solver.message))

        # Call the MILP post-solve callback
        config.master_postsolve(m, solve_data)

    def _solve_ECP_master(self, solve_data, config):
        solve_data.mip_iter += 1
        m = solve_data.working_model.clone()
        MindtPy = m.MindtPy_utils

        feas_sol = 0
        config.logger.info(
            'MIP %s: Solve master problem.' %
            (solve_data.mip_iter,))
        # Set up MILP
        for c in MindtPy.nonlinear_constraints:
            c.deactivate()
        MindtPy.MindtPy_linear_cuts.activate()

        # Deactivate extraneous IMPORT/EXPORT suffixes
        getattr(m, 'ipopt_zL_out', _DoNothing()).deactivate()
        getattr(m, 'ipopt_zU_out', _DoNothing()).deactivate()

        results = solve_data.mip_solver.solve(m, load_solutions=False,
                                        options=config.mip_solver_kwargs)
        master_terminate_cond = results.solver.termination_condition
        if master_terminate_cond is tc.infeasibleOrUnbounded:
            # Linear solvers will sometimes tell that it's infeasible or
            # unbounded during presolve, but fails to distinguish. We need to
            # resolve with a solver option flag on.
            from copy import deepcopy
            old_options = deepcopy(solve_data.mip_solver.options)
            # This solver option is specific to Gurobi.
            solve_data.mip_solver.options['DualReductions'] = 0
            results = solve_data.mip_solver.solve(m, load_solutions=False,
                                            options=config.mip_solver_kwargs)
            master_terminate_cond = results.solver.termination_condition
            solve_data.mip_solver.options.update(old_options)
        for c in MindtPy.nonlinear_constraints:
            c.activate()
            MindtPy.MindtPy_linear_cuts.deactivate()

        # Process master problem result
        if master_terminate_cond is tc.optimal:
            # proceed. Just need integer values
            m.solutions.load_from(results)
            self._copy_values(m, solve_data.working_model, config)

            if all(
                (0 if c.upper is None
                 else (value(c.body) - c.upper)) +
                (0 if c.lower is None
                 else (c.lower - value(c.body)))
                    < config.ECP_tolerance
                    for c in MindtPy.nonlinear_constraints):
                solve_data.best_solution_found = m.clone()
                feas_sol = 1
                print('ECP has found a feasible solution within a {} tolerance'
                      .format(config.ECP_tolerance))
            if MindtPy.obj.sense == minimize:
                solve_data.LB = max(value(MindtPy.obj.expr), solve_data.LB)
                solve_data.LB_progress.append(solve_data.LB)
                if feas_sol == 1:
                    solve_data.UB = value(MindtPy.obj.expr)
            else:
                solve_data.UB = min(value(MindtPy.obj.expr), solve_data.UB)
                solve_data.UB_progress.append(solve_data.UB)
                if feas_sol == 1:
                    solve_data.LB = value(MindtPy.obj.expr)
            config.logger.info(
                'MIP %s: OBJ: %s  LB: %s  UB: %s'
                % (solve_data.mip_iter, value(MindtPy.obj.expr),
                   solve_data.LB, solve_data.UB))
        elif master_terminate_cond is tc.infeasible:
            print('MILP master problem is infeasible. '
                  'Problem may have no more feasible binary combinations.')
            if solve_data.mip_iter == 1:
                print('MindtPy initialization may have generated poor '
                      'quality cuts.')
            # set optimistic bound to infinity
            if MindtPy.obj.sense == minimize:
                solve_data.LB = float('inf')
            else:
                solve_data.UB = float('-inf')
        else:
            raise ValueError(
                'MindtPy unable to handle MILP master termination condition '
                'of {}. Solver message: {}'.format(
                    master_terminate_cond, results.solver.message))

        # Call the MILP post-solve callback
        config.master_postsolve(m, solve_data)

    def _solve_PSC_master(self, solve_data, config):
        solve_data.mip_iter += 1
        m = solve_data.working_model.clone()
        MindtPy = m.MindtPy_utils

        config.logger.info(
            'MIP %s: Solve master problem.' %
            (solve_data.mip_iter,))
        # Set up MILP
        for c in MindtPy.nonlinear_constraints:
            c.deactivate()
        MindtPy.MindtPy_linear_cuts.activate()
        getattr(m, 'ipopt_zL_out', _DoNothing()).deactivate()
        getattr(m, 'ipopt_zU_out', _DoNothing()).deactivate()
        # m.pprint() #print psc master problem for debugging
        results = solve_data.mip_solver.solve(m, load_solutions=False,
                                        options=config.mip_solver_kwargs)
        for c in MindtPy.nonlinear_constraints:
            c.activate()
        MindtPy.MindtPy_linear_cuts.deactivate()
        getattr(m, 'ipopt_zL_out', _DoNothing()).activate()
        getattr(m, 'ipopt_zU_out', _DoNothing()).activate()

        # Process master problem result
        master_terminate_cond = results.solver.termination_condition
        if master_terminate_cond is tc.optimal:
            # proceed. Just need integer values
            m.solutions.load_from(results)
            self._copy_values(m, solve_data.working_model, config)

            if MindtPy.obj.sense == minimize:
                solve_data.LB = max(value(MindtPy.obj.expr), solve_data.LB)
                solve_data.LB_progress.append(solve_data.LB)
            else:
                solve_data.UB = min(value(MindtPy.obj.expr), solve_data.UB)
                solve_data.UB_progress.append(solve_data.UB)
            config.logger.info(
                'MIP %s: OBJ: %s  LB: %s  UB: %s'
                % (solve_data.mip_iter, value(MindtPy.obj.expr),
                   solve_data.LB, solve_data.UB))
        elif master_terminate_cond is tc.infeasible:
            print('MILP master problem is infeasible. '
                  'Problem may have no more feasible binary combinations.')
            if solve_data.mip_iter == 1:
                print('MindtPy initialization may have generated poor '
                      'quality cuts.')
            # set optimistic bound to infinity
            if MindtPy.obj.sense == minimize:
                solve_data.LB = float('inf')
            else:
                solve_data.UB = float('-inf')
        else:
            m.solutions.load_from(results)
            raise ValueError(
                'MindtPy unable to handle MILP master termination condition '
                'of {}. Solver message: {}'.format(
                    master_terminate_cond, results.solver.message))

        # Call the MILP post-solve callback
        config.master_postsolve(m, solve_data)

    def _solve_GBD_master(self, solve_data, config, leave_linear_active=True):
        solve_data.mip_iter += 1
        m = solve_data.working_model.clone()
        MindtPy = m.MindtPy_utils

        config.logger.info(
            'MIP %s: Solve master problem.' %
            (solve_data.mip_iter,))
        if not leave_linear_active:
            # Deactivate all constraints except those in MindtPy_linear_cuts
            _MindtPy_linear_cuts = set(
                c for c in MindtPy.MindtPy_linear_cuts.component_data_objects(
                    ctype=Constraint, descend_into=True))
            to_deactivate = set(c for c in m.component_data_objects(
                ctype=Constraint, active=True, descend_into=True)
                if c not in _MindtPy_linear_cuts)
            for c in to_deactivate:
                c.deactivate()
        else:
            for c in MindtPy.nonlinear_constraints:
                c.deactivate()
        MindtPy.MindtPy_linear_cuts.activate()
        # m.MindtPy_objective_expr.activate() # This activation will be deleted
        getattr(m, 'ipopt_zL_out', _DoNothing()).deactivate()
        getattr(m, 'ipopt_zU_out', _DoNothing()).deactivate()
        # m.pprint() #print gbd master problem for debugging
        results = solve_data.mip_solver.solve(m, load_solutions=False,
                                        options=config.mip_solver_kwargs)
        master_terminate_cond = results.solver.termination_condition
        if master_terminate_cond is tc.infeasibleOrUnbounded:
            # Linear solvers will sometimes tell me that it is infeasible or
            # unbounded during presolve, but fails to distinguish. We need to
            # resolve with a solver option flag on.
            old_options = deepcopy(solve_data.mip_solver.options)
            # This solver option is specific to Gurobi.
            solve_data.mip_solver.options['DualReductions'] = 0
            results = solve_data.mip_solver.solve(m, load_solutions=False,
                                            options=config.mip_solver_kwargs)
            master_terminate_cond = results.solver.termination_condition
            solve_data.mip_solver.options.update(old_options)
        if not leave_linear_active:
            for c in to_deactivate:
                c.activate()
        else:
            for c in MindtPy.nonlinear_constraints:
                c.activate()
        MindtPy.MindtPy_linear_cuts.deactivate()
        getattr(m, 'ipopt_zL_out', _DoNothing()).activate()
        getattr(m, 'ipopt_zU_out', _DoNothing()).activate()

        # Process master problem result
        if master_terminate_cond is tc.optimal:
            # proceed. Just need integer values
            m.solutions.load_from(results)
            self._copy_values(m, solve_data.working_model, config)

            if MindtPy.obj.sense == minimize:
                solve_data.LB = max(value(MindtPy.obj.expr), solve_data.LB)
                solve_data.LB_progress.append(solve_data.LB)
            else:
                solve_data.UB = min(value(MindtPy.obj.expr), solve_data.UB)
                solve_data.UB_progress.append(solve_data.UB)
            config.logger.info(
                'MIP %s: OBJ: %s  LB: %s  UB: %s'
                % (solve_data.mip_iter, value(MindtPy.obj.expr),
                   solve_data.LB, solve_data.UB))
        elif master_terminate_cond is tc.infeasible:
            print('MILP master problem is infeasible. '
                  'Problem may have no more feasible binary configurations.')
            if solve_data.mip_iter == 1:
                print('MindtPy initialization may have generated poor '
                      'quality cuts.')
            # set optimistic bound to infinity
            if MindtPy.obj.sense == minimize:
                solve_data.LB = float('inf')
            else:
                solve_data.UB = float('-inf')
        elif master_terminate_cond is tc.unbounded:
            print('MILP master problem is unbounded. ')
            # Change the integer values to something new, re-solve.
            MindtPy.MindtPy_linear_cuts.activate()
            MindtPy.MindtPy_linear_cuts.feasible_integer_cuts.activate()
            self._init_max_binaries()
            MindtPy.MindtPy_linear_cuts.deactivate()
            MindtPy.MindtPy_linear_cuts.feasible_integer_cuts.deactivate()
        else:
            raise ValueError(
                'MindtPy unable to handle MILP master termination condition '
                'of {}. Solver message: {}'.format(
                    master_terminate_cond, results.solver.message))

        #
        # MindtPy.MindtPy_linear_cuts.deactivate()
        # Call the MILP post-solve callback
        config.master_postsolve(m, solve_data)
