"""Implementation of the MindtPy solver.

The MindtPy (MINLP Decomposition Tookit) solver applies a variety of
decomposition-based approaches to solve nonlinear continuous-discrete problems.
These approaches include:

- Outer approximation
- Benders decomposition (questionable)
- Partial surrogate cuts

This solver implementation was developed by Carnegie Mellon University in the
research group of Ignacio Grossmann.

For nonconvex problems, the bounds self.LB and self.UB may not be rigorous.

Questions: Qi Chen <https://github.com/qtothec>

"""
from copy import deepcopy
from math import copysign
from pprint import pprint

import pyomo.util.plugin
from pyomo.core.base import expr as EXPR
from pyomo.core.base import (Block, ComponentUID, Constraint, ConstraintList,
                             Expression, Objective, RangeSet, Set, Suffix, Var,
                             maximize, minimize, value)
from pyomo.core.base.expr_common import clone_expression
from pyomo.core.base.numvalue import NumericConstant
from pyomo.core.base.symbolic import differentiate
from pyomo.core.kernel import Binary, NonNegativeReals, Reals
from pyomo.opt import TerminationCondition as tc
from pyomo.opt import SolverFactory, SolverStatus
from pyomo.opt.base import IOptSolver
from pyomo.repn.canonical_repn import generate_canonical_repn

from six.moves import range


class MindtPySolver(pyomo.util.plugin.Plugin):
    """A decomposition-based MINLP solver."""

    pyomo.util.plugin.implements(IOptSolver)
    pyomo.util.plugin.alias('mindtpy',
                            doc='The MindtPy decomposition-based MINLP solver')

    def available(self, exception_flag=True):
        """Check if solver is available.

        TODO: For now, it is always available. However, sub-solvers may not
        always be available, and so this should reflect that possibility.

        """
        return True

    def solve(self, model, **kwds):
        """Solve the model.

        Warning: this solver is still in beta. Keyword arguments subject to
        change. Undocumented keyword arguments definitely subject to change.

        Warning: at this point in time, if you try to use PSC or GBD with
        anything other than IPOPT as the NLP solver, bad things will happen.
        This is because the suffixes are not in place to extract dual values
        from the variable bounds for any other solver.

        TODO: something is wrong with the GBD implementation, I think...

        Args:
            model (Block): a Pyomo model or block to be solved

        Kwargs:
            tol (float): bound tolerance
            iterlim (int): maximum number of master iterations
            strategy (str): decomposition strategy to use. Possible values:
                OA, PSC, GBD, hPSC
            init_strategy (str): initialization strategy to use when generating
                the initial cuts to construct the master problem.
            max_slack (float): upper bound on slack variable values
            OA_penalty (float): multiplier on objective penalization for slack
                variables.
            nlp (str): Solver to use for nonlinear subproblems
            nlp_kwargs (dict): Keyword arguments to pass to NLP solver
            mip (str): Solver to use for linear discrete problems
            mip_kwargs (dict): Keyword arguments to pass to MIP solver
            solve_in_place (bool): If true, MindtPy manipulations are performed
                directly upon the model. Otherwise, the model is first copied
                and solution values are copied over afterwards.
            master_postsolve (func): callback hook after a solution of the
                master problem
            subprob_postsolve (func): callback hook after a solution of the
                nonlinear subproblem
            subprob_postfeas (func): callback hook after feasible solution of
                the nonlinear subproblem
            load_solutions (bool): if True, load solutions back into the model.
                This is only relevant if solve_in_place is not True.

        """
        self.bound_tolerance = kwds.pop('tol', 1E-6)
        self.iteration_limit = kwds.pop('iterlim', 30)
        self.decomposition_strategy = kwds.pop('strategy', 'OA')
        self.initialization_strategy = kwds.pop('init_strategy', None)
        self.max_slack = kwds.pop('max_slack', 1000)
        self.OA_penalty_factor = kwds.pop('OA_penalty', 1000)
        self.nlp_solver_name = kwds.pop('nlp', 'ipopt')
        self.nlp_solver_kwargs = kwds.pop('nlp_kwargs', {})
        self.mip_solver_name = kwds.pop('mip', 'gurobi')
        self.mip_solver_kwargs = kwds.pop('mip_kwargs', {})
        self.modify_in_place = kwds.pop('solve_in_place', True)
        self.master_postsolve = kwds.pop('master_postsolve', _DoNothing())
        self.subproblem_postsolve = kwds.pop('subprob_postsolve', _DoNothing())
        self.subproblem_postfeasible = kwds.pop('subprob_postfeas',
                                                _DoNothing())
        self.load_solutions = kwds.pop('load_solutions', True)
        if kwds:
            print("Unrecognized arguments passed to MindtPy solver:")
            pprint(kwds)

        # If decomposition strategy is a hybrid, set the initial strategy
        if self.decomposition_strategy == 'hPSC':
            self._decomposition_strategy = 'PSC'
        else:
            self._decomposition_strategy = self.decomposition_strategy

        # When generating cuts, small duals multiplied by expressions can cause
        # problems. Exclude all duals smaller in absolue value than the
        # following.
        self.small_dual_tolerance = 1E-8
        self.integer_tolerance = 1E-5

        # Modify in place decides whether to run the algorithm on a copy of the
        # originally model passed to the solver, or whether to manipulate the
        # original model directly.
        if self.modify_in_place:
            self.m = m = model
        else:
            self.m = m = model.clone()

        # Store the initial model state as the best solution found. If we find
        # no better solution, then we will restore from this copy.
        self.best_solution_found = model.clone()

        # Save model initial values. These are used later to initialize NLP
        # subproblems.
        self.initial_variable_values = {
            id(v): v.value for v in m.component_data_objects(
                ctype=Var, descend_into=True)}

        # Validate the model to ensure that MindtPy is able to solve it.
        #
        # This needs to take place before the detection of nonlinear
        # constraints, because if the objective is nonlinear, it will be moved
        # to the constraints.
        assert(not hasattr(self, 'nonlinear_constraints'))
        self._validate_model()

        # Create a model block in which to store the generated linear
        # constraints. Do not leave the constraints on by default.
        lin = m.MindtPy_linear_cuts = Block()
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

        # Build a list of binary variables
        self.binary_vars = [v for v in m.component_data_objects(
            ctype=Var, descend_into=True)
            if v.is_binary() and not v.fixed]

        # Build list of nonlinear constraints
        self.nonlinear_constraints = [
            v for v in m.component_data_objects(
                ctype=Constraint, active=True, descend_into=True)
            if v.body.polynomial_degree() not in (0, 1)]

        # Set up iteration counters
        self.nlp_iter = 0
        self.mip_iter = 0
        self.mip_subiter = 0

        # Set of NLP iterations for which cuts were generated
        lin.nlp_iters = Set(dimen=1)

        # Create an integer index set over the nonlinear constraints
        lin.nl_constraint_set = RangeSet(len(self.nonlinear_constraints))
        # Mapping Constraint -> integer index
        self.nl_map = {}
        # Mapping integer index -> Constraint
        self.nl_inverse_map = {}
        # Generate the two maps. These maps may be helpful for later
        # interpreting indices on the slack variables or generated cuts.
        for c, n in zip(self.nonlinear_constraints, lin.nl_constraint_set):
            self.nl_map[c] = n
            self.nl_inverse_map[n] = c

        # Create slack variables
        lin.slack_vars = Var(lin.nlp_iters, lin.nl_constraint_set,
                             domain=NonNegativeReals,
                             bounds=(0, self.max_slack), initialize=0)

        # set up bounds
        self.LB = float('-inf')
        self.UB = float('inf')
        self.LB_progress = [self.LB]
        self.UB_progress = [self.UB]

        # Flag indicating whether the solution improved in the past iteration
        # or not
        self.solution_improved = False

        # Set up solvers
        self.nlp_solver = SolverFactory(self.nlp_solver_name)
        self.mip_solver = SolverFactory(self.mip_solver_name)

        # Initialize the master problem
        self._MindtPy_initialize_master()

        # Algorithm main loop
        self._MindtPy_iteration_loop()

        # Update values in original model
        if self.load_solutions:
            for v in self.best_solution_found.component_data_objects(
                    ctype=Var, descend_into=True):
                uid = ComponentUID(v)
                orig_model_var = uid.find_component_on(model)
                if orig_model_var is not None:
                    try:
                        orig_model_var.set_value(v.value)
                    except ValueError as err:
                        if 'is not in domain Binary' in err.message:
                            # check to see whether this is just a tolerance
                            # issue
                            if (value(abs(v - 1)) <= self.integer_tolerance or
                                    value(abs(v)) <= self.integer_tolerance):
                                orig_model_var.set_value(round(v.value))
                            else:
                                raise

    def _validate_model(self):
        m = self.m
        # Check for any integer variables
        if any(True for v in m.component_data_objects(
                ctype=Var, descend_into=True)
                if v.is_integer() and not v.fixed):
            raise ValueError('Model contains unfixed integer variables. '
                             'MindtPy does not currently support solution of '
                             'such problems.')
            # TODO add in the reformulation using base 2

        # Handle missing or multiple objectives
        objs = m.component_data_objects(
            ctype=Objective, active=True, descend_into=True)
        # Fetch the first active objective in the model
        main_obj = next(objs, None)
        if main_obj is None:
            raise ValueError('Model has no active objectives.')
        # Fetch the next active objective in the model
        if next(objs, None) is not None:
            raise ValueError('Model has multiple active objectives.')

        # Move the objective to the constraints
        m.MindtPy_objective_value = Var(domain=Reals)
        if main_obj.sense == minimize:
            m.MindtPy_objective_expr = Constraint(
                expr=m.MindtPy_objective_value >= main_obj.expr)
        else:
            m.MindtPy_objective_expr = Constraint(
                expr=m.MindtPy_objective_value <= main_obj.expr)
        main_obj.deactivate()
        self.obj = m.MindtPy_objective = Objective(
            expr=m.MindtPy_objective_value, sense=main_obj.sense)

        # TODO if any continuous variables are multipled with binary ones, need
        # to do some kind of transformation (Glover?) or throw an error message

    def _MindtPy_initialize_master(self):
        """Initialize the decomposition algorithm.

        This includes generating the initial cuts require to build the master
        problem.

        """
        m = self.m
        if (self._decomposition_strategy == 'OA' or
                self.decomposition_strategy == 'hPSC'):
            if not hasattr(m, 'dual'):  # Set up dual value reporting
                m.dual = Suffix(direction=Suffix.IMPORT)
            # Map Constraint, nlp_iter -> generated OA Constraint
            self.OA_constr_map = {}
            self._calc_jacobians()  # preload jacobians
            self.m.MindtPy_linear_cuts.oa_cuts = ConstraintList(
                doc='Outer approximation cuts')
        if self._decomposition_strategy == 'PSC':
            if not hasattr(m, 'dual'):  # Set up dual value reporting
                m.dual = Suffix(direction=Suffix.IMPORT)
            if not hasattr(m, 'ipopt_zL_out'):
                m.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
            if not hasattr(m, 'ipopt_zU_out'):
                m.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
            self._detect_nonlinear_vars()
            self.m.MindtPy_linear_cuts.psc_cuts = ConstraintList(
                doc='Partial surrogate cuts')
        if self._decomposition_strategy == 'GBD':
            if not hasattr(m, 'dual'):
                m.dual = Suffix(direction=Suffix.IMPORT)
            if not hasattr(m, 'ipopt_zL_out'):
                m.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
            if not hasattr(m, 'ipopt_zU_out'):
                m.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
            self.m.MindtPy_linear_cuts.gbd_cuts = ConstraintList(
                doc='Generalized Benders cuts')

        # Set default initialization_strategy
        if self.initialization_strategy is None:
            if self._decomposition_strategy == 'OA':
                self.initialization_strategy = 'rNLP'
            else:
                self.initialization_strategy = 'max_binary'

        # Do the initialization
        if self.initialization_strategy == 'rNLP':
            self._init_rNLP()
        elif self.initialization_strategy == 'max_binary':
            self._init_max_binaries()
            self._solve_NLP_subproblem()

    def _init_rNLP(self):
        """Initialize by solving the rNLP (relaxed binary variables)."""
        self.nlp_iter += 1
        print("NLP {}: Solve relaxed integrality.".format(self.nlp_iter))
        for v in self.binary_vars:
            v.domain = NonNegativeReals
            v.setlb(0)
            v.setub(1)
        results = self.nlp_solver.solve(self.m, **self.nlp_solver_kwargs)
        for v in self.binary_vars:
            v.domain = Binary
        subprob_terminate_cond = results.solver.termination_condition
        if subprob_terminate_cond is tc.optimal:
            # Add OA cut
            if self.obj.sense == minimize:
                self.LB = value(self.obj.expr)
            else:
                self.UB = value(self.obj.expr)
            print('Objective {}'.format(value(self.obj.expr)))
            self._add_oa_cut()
        elif subprob_terminate_cond is tc.infeasible:
            # TODO fail? try something else?
            raise ValueError('Initial relaxed NLP infeasible. '
                             'Problem may be infeasible.')
        else:
            raise ValueError(
                'MindtPy unable to handle relaxed NLP termination condition '
                'of {}'.format(subprob_terminate_cond))

    def _init_max_binaries(self):
        """Initialize by turning on as many binary variables as possible.

        The user would usually want to call _solve_NLP_subproblem after an
        invocation of this function.

        """
        self.mip_subiter += 1
        m = self.m
        print("MILP {}.{}: maximize value of binaries".format(
            self.mip_iter, self.mip_subiter))
        for c in self.nonlinear_constraints:
            c.deactivate()
        self.obj.deactivate()
        m.MindtPy_max_binary_obj = Objective(
            expr=sum(v for v in self.binary_vars), sense=maximize)
        getattr(m, 'ipopt_zL_out', _DoNothing()).deactivate()
        getattr(m, 'ipopt_zU_out', _DoNothing()).deactivate()
        results = self.mip_solver.solve(m, **self.mip_solver_kwargs)
        getattr(m, 'ipopt_zL_out', _DoNothing()).activate()
        getattr(m, 'ipopt_zU_out', _DoNothing()).activate()
        m.del_component(m.MindtPy_max_binary_obj)
        self.obj.activate()
        for c in self.nonlinear_constraints:
            c.activate()
        solve_terminate_cond = results.solver.termination_condition
        if solve_terminate_cond is tc.optimal:
            pass  # good
        elif solve_terminate_cond is tc.infeasible:
            raise ValueError('Linear relaxation is infeasible. '
                             'Problem is infeasible.')
        else:
            raise ValueError('Cannot handle termination condition {}'.format(
                solve_terminate_cond))

    def _MindtPy_iteration_loop(self):
        # Backup counter to prevent infinite loop
        backup_max_iter = max(1000, self.iteration_limit)
        backup_iter = 0
        while backup_iter < backup_max_iter:
            print('\n')  # print blank lines for visual display
            backup_iter += 1
            # Check bound convergence
            if self.LB + self.bound_tolerance >= self.UB:
                print('MindtPy exiting on bound convergence. '
                      'LB: {} + (tol {}) >= UB: {}'.format(
                          self.LB, self.bound_tolerance, self.UB))
                break
            # Check iteration limit
            if self.mip_iter >= self.iteration_limit:
                print('MindtPy unable to converge bounds '
                      'after {} master iterations.'.format(self.mip_iter))
                print('Final bound values: LB: {}  UB: {}'.
                      format(self.LB, self.UB))
                break
            self.mip_subiter = 0
            # solve MILP master problem
            if self._decomposition_strategy == 'OA':
                self._solve_OA_master()
            elif self._decomposition_strategy == 'PSC':
                self._solve_PSC_master()
            elif self._decomposition_strategy == 'GBD':
                self._solve_GBD_master()
            # Check bound convergence
            if self.LB + self.bound_tolerance >= self.UB:
                print('MindtPy exiting on bound convergence. '
                      'LB: {} + (tol {}) >= UB: {}'.format(
                          self.LB, self.bound_tolerance, self.UB))
                break
            # Solve NLP subproblem
            self._solve_NLP_subproblem()

            # If the hybrid algorithm is not making progress, switch to OA.
            progress_required = 1E-6
            if self.obj.sense == minimize:
                log = self.LB_progress
                sign_adjust = 1
            else:
                log = self.UB_progress
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
                    self.decomposition_strategy == 'hPSC' and
                    self._decomposition_strategy == 'PSC'):
                print('Not making enough progress for {} iterations. '
                      'Switching to OA.'.format(max_nonimprove_iter))
                self._decomposition_strategy = 'OA'

    def _solve_OA_master(self):
        m = self.m
        self.mip_iter += 1
        print('MILP {}: Solve master problem.'.format(self.mip_iter))
        # Set up MILP
        for c in self.nonlinear_constraints:
            c.deactivate()
        m.MindtPy_linear_cuts.activate()
        self.obj.deactivate()
        m.del_component('MindtPy_penalty_expr')
        sign_adjust = 1 if self.obj.sense == minimize else -1
        m.MindtPy_penalty_expr = Expression(
            expr=sign_adjust * self.OA_penalty_factor * sum(
                v for v in m.MindtPy_linear_cuts.slack_vars[...]))
        m.del_component('MindtPy_oa_obj')
        m.MindtPy_oa_obj = Objective(
            expr=self.obj.expr + m.MindtPy_penalty_expr,
            sense=self.obj.sense)
        getattr(m, 'ipopt_zL_out', _DoNothing()).deactivate()
        getattr(m, 'ipopt_zU_out', _DoNothing()).deactivate()
        results = self.mip_solver.solve(m, load_solutions=False,
                                        **self.mip_solver_kwargs)
        master_terminate_cond = results.solver.termination_condition
        if master_terminate_cond is tc.infeasibleOrUnbounded:
            # Linear solvers will sometimes tell me that it's infeasible or
            # unbounded during presolve, but fails to distinguish. We need to
            # resolve with a solver option flag on.
            from copy import deepcopy
            old_options = deepcopy(self.mip_solver.options)
            # This solver option is specific to Gurobi.
            self.mip_solver.options['DualReductions'] = 0
            results = self.mip_solver.solve(m, load_solutions=False,
                                            **self.mip_solver_kwargs)
            master_terminate_cond = results.solver.termination_condition
            self.mip_solver.options.update(old_options)
        self.obj.activate()
        for c in self.nonlinear_constraints:
            c.activate()
        m.MindtPy_linear_cuts.deactivate()
        m.MindtPy_oa_obj.deactivate()
        getattr(m, 'ipopt_zL_out', _DoNothing()).activate()
        getattr(m, 'ipopt_zU_out', _DoNothing()).activate()

        # Process master problem result
        if master_terminate_cond is tc.optimal:
            # proceed. Just need integer values
            m.solutions.load_from(results)
            if self.obj.sense == minimize:
                self.LB = max(value(m.MindtPy_oa_obj.expr), self.LB)
                self.LB_progress.append(self.LB)
            else:
                self.UB = min(value(m.MindtPy_oa_obj.expr), self.UB)
                self.UB_progress.append(self.UB)
            print('MIP {}: OBJ: {}  LB: {}  UB: {}'
                  .format(self.mip_iter, value(m.MindtPy_oa_obj.expr), self.LB,
                          self.UB))
        elif master_terminate_cond is tc.infeasible:
            print('MILP master problem is infeasible. '
                  'Problem may have no more feasible binary configurations.')
            if self.mip_iter == 1:
                print('MindtPy initialization may have generated poor '
                      'quality cuts.')
            # set optimistic bound to infinity
            if self.obj.sense == minimize:
                self.LB = float('inf')
            else:
                self.UB = float('-inf')
        else:
            raise ValueError(
                'MindtPy unable to handle MILP master termination condition '
                'of {}. Solver message: {}'.format(
                    master_terminate_cond, results.solver.message))

        # Call the MILP post-solve callback
        self.master_postsolve(m, self)

    def _solve_PSC_master(self):
        m = self.m
        self.mip_iter += 1
        print('MILP {}: Solve master problem.'.format(self.mip_iter))
        # Set up MILP
        for c in self.nonlinear_constraints:
            c.deactivate()
        m.MindtPy_linear_cuts.activate()
        getattr(m, 'ipopt_zL_out', _DoNothing()).deactivate()
        getattr(m, 'ipopt_zU_out', _DoNothing()).deactivate()
        results = self.mip_solver.solve(m, load_solutions=False,
                                        **self.mip_solver_kwargs)
        for c in self.nonlinear_constraints:
            c.activate()
        m.MindtPy_linear_cuts.deactivate()
        getattr(m, 'ipopt_zL_out', _DoNothing()).activate()
        getattr(m, 'ipopt_zU_out', _DoNothing()).activate()

        # Process master problem result
        master_terminate_cond = results.solver.termination_condition
        if master_terminate_cond is tc.optimal:
            # proceed. Just need integer values
            m.solutions.load_from(results)
            if self.obj.sense == minimize:
                self.LB = max(value(self.obj.expr), self.LB)
                self.LB_progress.append(self.LB)
            else:
                self.UB = min(value(self.obj.expr), self.UB)
                self.UB_progress.append(self.UB)
            print('MIP {}: OBJ: {}  LB: {}  UB: {}'
                  .format(self.mip_iter, value(self.obj.expr), self.LB,
                          self.UB))
        elif master_terminate_cond is tc.infeasible:
            print('MILP master problem is infeasible. '
                  'Problem may have no more feasible binary configurations.')
            if self.mip_iter == 1:
                print('MindtPy initialization may have generated poor '
                      'quality cuts.')
            # set optimistic bound to infinity
            if self.obj.sense == minimize:
                self.LB = float('inf')
            else:
                self.UB = float('-inf')
        elif master_terminate_cond is tc.unbounded:
            print('MILP master problem is unbounded. ')
            m.solutions.load_from(results)
        else:
            raise ValueError(
                'MindtPy unable to handle MILP master termination condition '
                'of {}. Solver message: {}'.format(
                    master_terminate_cond, results.solver.message))

        # Call the MILP post-solve callback
        self.master_postsolve(m, self)

    def _solve_GBD_master(self, leave_linear_active=False):
        m = self.m
        self.mip_iter += 1
        print('MILP {}: Solve master problem.'.format(self.mip_iter))
        if not leave_linear_active:
            # Deactivate all constraints except those in MindtPy_linear_cuts
            _MindtPy_linear_cuts = set(
                c for c in m.MindtPy_linear_cuts.component_data_objects(
                    ctype=Constraint, descend_into=True))
            to_deactivate = set(c for c in m.component_data_objects(
                ctype=Constraint, active=True, descend_into=True)
                if c not in _MindtPy_linear_cuts)
            for c in to_deactivate:
                c.deactivate()
        else:
            for c in self.nonlinear_constraints:
                c.deactivate()
        m.MindtPy_linear_cuts.activate()
        m.MindtPy_objective_expr.activate()
        getattr(m, 'ipopt_zL_out', _DoNothing()).deactivate()
        getattr(m, 'ipopt_zU_out', _DoNothing()).deactivate()
        results = self.mip_solver.solve(m, load_solutions=False,
                                        **self.mip_solver_kwargs)
        master_terminate_cond = results.solver.termination_condition
        if master_terminate_cond is tc.infeasibleOrUnbounded:
            # Linear solvers will sometimes tell me that it is infeasible or
            # unbounded during presolve, but fails to distinguish. We need to
            # resolve with a solver option flag on.
            old_options = deepcopy(self.mip_solver.options)
            # This solver option is specific to Gurobi.
            self.mip_solver.options['DualReductions'] = 0
            results = self.mip_solver.solve(m, load_solutions=False,
                                            **self.mip_solver_kwargs)
            master_terminate_cond = results.solver.termination_condition
            self.mip_solver.options.update(old_options)
        if not leave_linear_active:
            for c in to_deactivate:
                c.activate()
        else:
            for c in self.nonlinear_constraints:
                c.activate()
        m.MindtPy_linear_cuts.deactivate()
        getattr(m, 'ipopt_zL_out', _DoNothing()).activate()
        getattr(m, 'ipopt_zU_out', _DoNothing()).activate()

        # Process master problem result
        if master_terminate_cond is tc.optimal:
            # proceed. Just need integer values
            m.solutions.load_from(results)
            if self.obj.sense == minimize:
                self.LB = max(value(self.obj.expr), self.LB)
                self.LB_progress.append(self.LB)
            else:
                self.UB = min(value(self.obj.expr), self.UB)
                self.UB_progress.append(self.UB)
            print('MIP {}: OBJ: {}  LB: {}  UB: {}'
                  .format(self.mip_iter, value(self.obj.expr), self.LB,
                          self.UB))
        elif master_terminate_cond is tc.infeasible:
            print('MILP master problem is infeasible. '
                  'Problem may have no more feasible binary configurations.')
            if self.mip_iter == 1:
                print('MindtPy initialization may have generated poor '
                      'quality cuts.')
            # set optimistic bound to infinity
            if self.obj.sense == minimize:
                self.LB = float('inf')
            else:
                self.UB = float('-inf')
        elif master_terminate_cond is tc.unbounded:
            print('MILP master problem is unbounded. ')
            # Change the integer values to something new, re-solve.
            m.MindtPy_linear_cuts.activate()
            m.MindtPy_linear_cuts.feasible_integer_cuts.activate()
            self._init_max_binaries()
            m.MindtPy_linear_cuts.deactivate()
            m.MindtPy_linear_cuts.feasible_integer_cuts.deactivate()
        else:
            raise ValueError(
                'MindtPy unable to handle MILP master termination condition '
                'of {}. Solver message: {}'.format(
                    master_terminate_cond, results.solver.message))

        # Call the MILP post-solve callback
        self.master_postsolve(m, self)

    def _solve_NLP_subproblem(self):
        m = self.m
        self.nlp_iter += 1
        print('NLP {}: Solve subproblem for fixed binaries.'
              .format(self.nlp_iter))
        # Set up NLP
        for v in self.binary_vars:
            v.fix()
        # restore original variable values
        for v in m.component_data_objects(ctype=Var, descend_into=True):
            if not v.fixed and not v.is_binary():
                try:
                    v.set_value(self.initial_variable_values[id(v)])
                except KeyError:
                    continue
        # Solve the NLP
        results = self.nlp_solver.solve(m, load_solutions=False,
                                        **self.nlp_solver_kwargs)
        for v in self.binary_vars:
            v.unfix()
        subprob_terminate_cond = results.solver.termination_condition
        if subprob_terminate_cond is tc.optimal:
            m.solutions.load_from(results)
            if self.obj.sense == minimize:
                self.UB = min(value(self.obj.expr), self.UB)
                self.solution_improved = self.UB < self.UB_progress[-1]
                self.UB_progress.append(self.UB)
            else:
                self.LB = max(value(self.obj.expr), self.LB)
                self.solution_improved = self.LB > self.LB_progress[-1]
                self.LB_progress.append(self.LB)
            print('NLP {}: OBJ: {}  LB: {}  UB: {}'
                  .format(self.nlp_iter, value(self.obj.expr), self.LB,
                          self.UB))
            if self.solution_improved:
                self.best_solution_found = m.clone()
            # Add the linear cut
            if self._decomposition_strategy == 'OA':
                self._add_oa_cut()
            elif self._decomposition_strategy == 'PSC':
                self._add_psc_cut()
            elif self._decomposition_strategy == 'GBD':
                self._add_gbd_cut()

            # This adds an integer cut to the feasible_integer_cuts
            # ConstraintList, which is not activated by default. However, it
            # may be activated as needed in certain situations or for certain
            # values of option flags.
            self._add_int_cut(feasible=True)

            self.subproblem_postfeasible(m, self)
        elif subprob_terminate_cond is tc.infeasible:
            # TODO try something else? Reinitialize with different initial
            # value?
            print('NLP subproblem was locally infeasible.')
            # load the solution and suppress the warning message by setting
            # solver status to ok.
            results.solver.status = SolverStatus.ok
            m.solutions.load_from(results)
            if self._decomposition_strategy == 'PSC':
                print('Adding PSC feasibility cut.')
                self._add_psc_cut(nlp_feasible=False)
            elif self._decomposition_strategy == 'GBD':
                print('Adding GBD feasibility cut.')
                self._add_gbd_cut(nlp_feasible=False)
            # Add an integer cut to exclude this discrete option
            self._add_int_cut()
        elif subprob_terminate_cond is tc.maxIterations:
            # TODO try something else? Reinitialize with different initial
            # value?
            print('NLP subproblem failed to converge within iteration limit.')
            # Add an integer cut to exclude this discrete option
            self._add_int_cut()
        else:
            raise ValueError(
                'MindtPy unable to handle NLP subproblem termination '
                'condition of {}'.format(subprob_terminate_cond))

        # Call the NLP post-solve callback
        self.subproblem_postsolve(m, self)

    def _calc_jacobians(self):
        self.jacs = {}
        for c in self.nonlinear_constraints:
            constraint_vars = list(EXPR.identify_variables(c.body))
            jac_list = differentiate(c.body, wrt_list=constraint_vars)
            self.jacs[c] = {id(var): jac
                            for var, jac in zip(constraint_vars, jac_list)}

    def _add_oa_cut(self):
        m = self.m
        m.MindtPy_linear_cuts.nlp_iters.add(self.nlp_iter)
        sign_adjust = -1 if self.obj.sense == minimize else 1

        # generate new constraints
        # TODO some kind of special handling if the dual is phenomenally small?
        for constr in self.jacs:
            c = m.MindtPy_linear_cuts.oa_cuts.add(
                expr=copysign(1, sign_adjust * m.dual[constr]) * sum(
                    value(self.jacs[constr][id(var)]) * (var - value(var))
                    for var in list(EXPR.identify_variables(constr.body))) +
                m.MindtPy_linear_cuts.slack_vars[self.nlp_iter,
                                                 self.nl_map[constr]] <= 0)
            self.OA_constr_map[constr, self.nlp_iter] = c

    def _add_psc_cut(self, nlp_feasible=True):
        m = self.m

        sign_adjust = 1 if self.obj.sense == minimize else -1

        # generate the sum of all multipliers with the nonlinear constraints
        var_to_val = {id(var): NumericConstant(value(var))
                      for var in self.nonlinear_variables}
        sum_nonlinear = (
            # Address constraints of form f(x) <= upper
            sum(value(m.dual[c]) * -1 *
                (clone_expression(c.body, substitute=var_to_val) - c.upper)
                for c in self.nonlinear_constraints
                if value(abs(m.dual[c])) > self.small_dual_tolerance
                and c.upper is not None) +
            # Address constraints of form f(x) >= lower
            sum(value(m.dual[c]) *
                (c.lower - clone_expression(c.body, substitute=var_to_val))
                for c in self.nonlinear_constraints
                if value(abs(m.dual[c])) > self.small_dual_tolerance
                and c.lower is not None and not c.upper == c.lower))

        # Generate the sum of all multipliers with linear constraints
        # containing nonlinear variables
        #
        # For now, need to generate canonical representation in order to get
        # the coefficients on the linear terms.
        lin_cons = [c for c in m.component_data_objects(
            ctype=Constraint, active=True, descend_into=True)
            if c.body.polynomial_degree() in (0, 1)]
        # Create a coefficient dictionary mapping variables to their
        # coefficient in the expression. Constraint -> (id(Var) -> coefficient)
        coef_dict = {}
        constr_vars = {}
        for constr in lin_cons:
            repn = generate_canonical_repn(constr.body)
            if repn.variables is None or repn.linear is None:
                repn.variables = []
                repn.linear = []
            coef_dict[constr] = {id(var): coef for var, coef in
                                 zip(repn.variables, repn.linear)}
            constr_vars[constr] = repn.variables
        sum_linear = sum(
            m.dual[c] *
            sum(coef_dict[c][id(var)] * (var - value(var))
                for var in constr_vars[c]
                if id(var) in self.nonlinear_variable_IDs)
            for c in lin_cons
            if value(abs(m.dual[c])) > self.small_dual_tolerance)

        # Generate the sum of all bound multipliers with nonlinear variables
        sum_var_bounds = (
            sum(m.ipopt_zL_out.get(var, 0) * (var - value(var))
                for var in self.nonlinear_variables
                if value(abs(m.ipopt_zL_out.get(var, 0))) >
                self.small_dual_tolerance) +
            sum(m.ipopt_zU_out.get(var, 0) * (var - value(var))
                for var in self.nonlinear_variables
                if value(abs(m.ipopt_zU_out.get(var, 0))) >
                self.small_dual_tolerance))

        if nlp_feasible:
            # Optimality cut (for feasible NLP)
            m.MindtPy_linear_cuts.psc_cuts.add(
                expr=self.obj.expr * sign_adjust >= sign_adjust * (
                    self.obj.expr + sum_nonlinear + sum_linear +
                    sum_var_bounds))
        else:
            # Feasibility cut (for infeasible NLP)
            m.MindtPy_linear_cuts.psc_cuts.add(
                expr=sign_adjust * (
                    sum_nonlinear + sum_linear + sum_var_bounds) <= 0)

    def _add_gbd_cut(self, nlp_feasible=True):
        m = self.m

        sign_adjust = 1 if self.obj.sense == minimize else -1

        for c in m.component_data_objects(ctype=Constraint, active=True,
                                          descend_into=True):
            if value(c.upper) is None:
                raise ValueError(
                    'Oh no, Pyomo did something MindtPy does not expect. '
                    'The value of c.upper for {} is None: {} <= {} <= {}'
                    .format(c.name, c.lower, c.body, c.upper))
        # TODO handle the case where constraint upper is None

        # only substitute non-binary variables to their values
        binary_var_ids = set(id(var) for var in self.binary_vars)
        var_to_val = {id(var): NumericConstant(value(var))
                      for var in m.component_data_objects(ctype=Var,
                                                          descend_into=True)
                      if id(var) not in binary_var_ids}
        # generate the sum of all multipliers with the active (from a duality
        # sense) constraints
        sum_constraints = (
            sum(value(m.dual[c]) * -1 *
                (clone_expression(c.body, substitute=var_to_val) - c.upper)
                for c in m.component_data_objects(
                    ctype=Constraint, active=True, descend_into=True)
                if value(abs(m.dual[c])) > self.small_dual_tolerance
                and c.upper is not None) +
            sum(value(m.dual[c]) *
                (c.lower - clone_expression(c.body, substitute=var_to_val))
                for c in m.component_data_objects(
                    ctype=Constraint, active=True, descend_into=True)
                if value(abs(m.dual[c])) > self.small_dual_tolerance
                and c.lower is not None and not c.upper == c.lower))

        # add in variable bound dual contributions
        #
        # Generate the sum of all bound multipliers with nonlinear variables
        sum_var_bounds = (
            sum(m.ipopt_zL_out.get(var, 0) * (var - value(var))
                for var in m.component_data_objects(ctype=Var,
                                                    descend_into=True)
                if (id(var) not in binary_var_ids and
                    value(abs(m.ipopt_zL_out.get(var, 0))) >
                    self.small_dual_tolerance)) +
            sum(m.ipopt_zU_out.get(var, 0) * (var - value(var))
                for var in m.component_data_objects(ctype=Var,
                                                    descend_into=True)
                if (id(var) not in binary_var_ids and
                    value(abs(m.ipopt_zU_out.get(var, 0))) >
                    self.small_dual_tolerance)))

        if nlp_feasible:
            m.MindtPy_linear_cuts.gbd_cuts.add(
                expr=self.obj.expr * sign_adjust >= sign_adjust * (
                    self.obj.expr + sum_constraints + sum_var_bounds))
        else:
            m.MindtPy_linear_cuts.gbd_cuts.add(
                expr=sign_adjust * (sum_constraints + sum_var_bounds) <= 0)

    def _add_int_cut(self, feasible=False):
        m = self.m
        int_tol = self.integer_tolerance
        # check to make sure that binary variables are all 0 or 1
        for v in self.binary_vars:
            if value(abs(v - 1)) > int_tol and value(abs(v)) > int_tol:
                raise ValueError('Binary {} = {} is not 0 or 1'.format(
                    v.name, value(v)))

        if not self.binary_vars:  # if no binary variables, skip.
            return

        int_cut = (sum(1 - v for v in self.binary_vars
                       if value(abs(v - 1)) <= int_tol) +
                   sum(v for v in self.binary_vars
                       if value(abs(v)) <= int_tol) >= 1)

        if not feasible:
            # Add the integer cut
            m.MindtPy_linear_cuts.integer_cuts.add(expr=int_cut)
        else:
            m.MindtPy_linear_cuts.feasible_integer_cuts.add(expr=int_cut)

    def _detect_nonlinear_vars(self):
        """Identify the variables that participate in nonlinear terms."""
        self.nonlinear_variables = []
        # This is a workaround because Var is not hashable, and I do not want
        # duplicates in self.nonlinear_variables.
        seen = set()
        for constr in self.nonlinear_constraints:
            if isinstance(constr.body, EXPR._SumExpression):
                # go through each term and check to see if the term is
                # nonlinear
                for expr in constr.body._args:
                    # Check to see if the expression is nonlinear
                    if expr.polynomial_degree() not in (0, 1):
                        # collect variables
                        for var in EXPR.identify_variables(
                                expr, include_fixed=False):
                            if id(var) not in seen:
                                seen.add(id(var))
                                self.nonlinear_variables.append(var)
            # if the root expression object is not a summation, then something
            # else is the cause of the nonlinearity. Collect all participating
            # variables.
            else:
                # collect variables
                for var in EXPR.identify_variables(constr.body,
                                                   include_fixed=False):
                    if id(var) not in seen:
                        seen.add(id(var))
                        self.nonlinear_variables.append(var)
        self.nonlinear_variable_IDs = set(
            id(v) for v in self.nonlinear_variables)


class _DoNothing(object):
    """Do nothing, literally.

    This class is used in situations of "do something if attribute exists."
    """

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        pass

    def __getattr__(self, attr):
        def _do_nothing(*args, **kwargs):
            pass
        return _do_nothing
