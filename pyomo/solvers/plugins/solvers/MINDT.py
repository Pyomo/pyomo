"""Implementation of the MINDT solver.

The MINDT (MINLP Decomposition Tookit) solver applies a variety of
decomposition-based approaches to solve nonlinear continuous-discrete problems.
These approaches include:

- Outer approximation
- Benders decomposition (unimplemented)
- Partial surrogate cuts (unimplemented)

This solver implementation was developed by Carnegie Mellon University in the
research group of Ignacio Grossmann.

For nonconvex problems, the bounds self.LB and self.UB may not be rigorous.

TODO: handle nonlinear constraint objects of form a <= f(x) <= b.

Questions: Qi Chen <qichen at andrew.cmu.edu>

"""
from math import copysign
from pprint import pprint

import pyomo.util.plugin
from pyomo.core.base import expr as EXPR
from pyomo.core.base.symbolic import differentiate
from pyomo.environ import (Binary, Block, ComponentUID, Constraint,
                           ConstraintList, Expression, NonNegativeReals,
                           Objective, RangeSet, Reals, Set, Suffix, Var,
                           maximize, minimize, value)
from pyomo.opt import TerminationCondition as tc
from pyomo.opt import SolverFactory, SolverStatus
from pyomo.opt.base import IOptSolver


def _do_nothing(*args, **kwargs):
    """Do nothing, literally."""
    pass


class MINDTSolver(pyomo.util.plugin.Plugin):
    """A decomposition-based MINLP solver."""

    pyomo.util.plugin.implements(IOptSolver)
    pyomo.util.plugin.alias('mindt',
                            doc='The MINDT decomposition-based MINLP solver')

    def available(self, exception_flag=True):
        """Check if solver is available.

        TODO: For now, it is always available. However, sub-solvers may not
        always be available, and so this should reflect that possibility.

        """
        return True

    def solve(self, model, **kwds):
        """Solve everything.

        All the problems. World peace? Done. Cure for cancer? You bet.

        Warning: this solver is still in beta. Keyword arguments subject to
        change. Undocumented keyword arguments definitely subject to change.

        Args:
            model (Block): a Pyomo model or block to be solved

        Kwargs:
            tol (float): bound tolerance
            iterlim (int): maximum number of master iterations
            strategy (str): decomposition strategy to use. Possible values:
                OA, PSC, GBD
            init_strategy (str): initialization strategy to use when generating
                the initial cuts to construct the master problem.
            max_slack (float): upper bound on slack variable values
            OA_penalty (float): multiplier on objective penalization for slack
                variables.
            nlp (str): Solver to use for nonlinear subproblems
            mip (str): Solver to use for linear discrete problems
            mip_kwargs (dict): Keyword arguments to pass to MIP solver

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
        self.master_postsolve = kwds.pop('master_postsolve', _do_nothing)
        self.subproblem_postsolve = kwds.pop('subprob_postsolve', _do_nothing)
        self.subproblem_postfeasible = kwds.pop('subprob_postfeas',
                                                _do_nothing)
        self.load_solutions = kwds.pop('load_solutions', True)
        if kwds:
            print("Unrecognized arguments passed to MINDT solver:")
            pprint(kwds)

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
        # Save model initial values
        self.initial_variable_values = {
            id(v): v.value for v in m.component_data_objects(
                ctype=Var, descend_into=True)}

        # Validate the model to ensure that MINDT is able to solve it.
        self._validate_model()

        # Create a model block in which to store the generated linear
        # constraints. Do not leave the constraints on by default.
        lin = m.MINDT_linear_cuts = Block()
        lin.deactivate()

        # Integer cuts exclude particular discrete decisions
        lin.integer_cuts = ConstraintList(doc='integer cuts')

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

        # Set up dual value reporting
        if not hasattr(m, 'dual'):
            m.dual = Suffix(direction=Suffix.IMPORT_EXPORT)

        # set up bounds
        self.LB = float('-inf')
        self.LB_old = float('-inf')
        self.UB = float('inf')
        self.UB_old = float('inf')
        # whether the solution improved in the past iteration or not
        self.solution_improved = False

        # Set up solvers
        self.nlp_solver = SolverFactory(self.nlp_solver_name)
        self.mip_solver = SolverFactory(self.mip_solver_name)

        # Set up solution strategy
        if self.decomposition_strategy == 'OA':
            self._calc_jacobians()  # preload jacobians
            lin.oa_cuts = ConstraintList(doc='OA_cuts')

        # Initialize the master problem
        self._MINDT_initialize_master()

        # Algorithm main loop
        self._MINDT_iteration_loop()

        # Update values in original model
        if self.load_solutions:
            for v in self.best_solution_found.component_data_objects(
                    ctype=Var, descend_into=True):
                uid = ComponentUID(v)
                orig_model_var = uid.find_component_on(model)
                if orig_model_var is not None:
                    orig_model_var.set_value(v.value)

    def _validate_model(self):
        m = self.m
        # Check for any integer variables
        if any(True for v in m.component_data_objects(
                ctype=Var, descend_into=True)
                if v.is_integer() and not v.fixed):
            raise ValueError('Model contains unfixed integer variables. '
                             'MINDT does not currently support solution of '
                             'such problems.')
            # TODO add in a reformulation?

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

        # If objective is nonlinear, move it into the constraints.
        if main_obj.expr.polynomial_degree() not in (0, 1):
            m.MINDT_objective_value = Var(domain=Reals)
            m.MINDT_nonlinear_objective = Constraint(
                expr=m.MINDT_objective_value == main_obj.expr)
            main_obj.deactivate()
            self.obj = m.MINDT_objective = Objective(
                expr=m.MINDT_objective_value, sense=main_obj.sense)
        else:
            self.obj = main_obj

        # TODO if any continuous variables are multipled with binary ones, need
        # to do some kind of transformation or throw an error message

    def _MINDT_initialize_master(self):
        """Initialize the decomposition algorithm.

        This includes generating the initial cuts require to build the master
        problem.

        """
        if self.decomposition_strategy == 'OA':
            self.OA_constr_map = {}
            if self.initialization_strategy in ('rNLP', None):
                self._init_OA_rNLP()
            elif self.initialization_strategy in ('max_binary',):
                self._init_OA_max_binaries()

    def _init_OA_rNLP(self):
        """Initialize OA by solving the rNLP (relaxed binary variables)."""
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
                'MINDT unable to handle relaxed NLP termination condition '
                'of {}'.format(subprob_terminate_cond))

    def _init_OA_max_binaries(self):
        m = self.m
        print("MILP 0: maximize value of binaries")
        for c in self.nonlinear_constraints:
            c.deactivate()
        self.obj.deactivate()
        m.MINDT_max_binary_obj = Objective(
            expr=sum(v for v in self.binary_vars), sense=maximize)
        results = self.mip_solver.solve(m, **self.mip_solver_kwargs)
        m.del_component(m.MINDT_max_binary_obj)
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
        # Solve the NLP subproblem at this binary realization
        self._solve_OA_subproblem()

    def _MINDT_iteration_loop(self):
        # Backup counter to prevent infinite loop
        backup_max_iter = max(1000, self.iteration_limit)
        backup_iter = 0
        while backup_iter < backup_max_iter:
            backup_iter += 1
            # Check bound convergence
            if self.LB + self.bound_tolerance >= self.UB:
                print('MINDT exiting on bound convergence. '
                      'LB: {} + (tol {}) >= UB: {}'.format(
                          self.LB, self.bound_tolerance, self.UB))
                break
            # Check iteration limit
            if self.mip_iter >= self.iteration_limit:
                print('MINDT unable to solve problem '
                      'after {} master iterations'.format(self.mip_iter))
                break
            # solve MILP master problem
            if self.decomposition_strategy == 'OA':
                self._solve_OA_master()
            # Check bound convergence
            if self.LB + self.bound_tolerance >= self.UB:
                print('MINDT exiting on bound convergence. '
                      'LB: {} + (tol {}) >= UB: {}'.format(
                          self.LB, self.bound_tolerance, self.UB))
                break
            # Solve NLP subproblem
            if self.decomposition_strategy == 'OA':
                self._solve_OA_subproblem()

    def _solve_OA_master(self):
        m = self.m
        self.mip_iter += 1
        print('MILP {}: Solve master problem.'.format(self.mip_iter))
        # Set up MILP
        for c in self.nonlinear_constraints:
            c.deactivate()
        m.MINDT_linear_cuts.activate()
        self.obj.deactivate()
        m.del_component('MINDT_penalty_expr')
        sign_adjust = 1 if self.obj.sense == minimize else -1
        m.MINDT_penalty_expr = Expression(
            expr=sign_adjust * self.OA_penalty_factor * sum(
                v for v in m.MINDT_linear_cuts.slack_vars[:, :]))
        m.del_component('MINDT_oa_obj')
        m.MINDT_oa_obj = Objective(
            expr=self.obj.expr + m.MINDT_penalty_expr,
            sense=self.obj.sense)
        results = self.mip_solver.solve(m, load_solutions=False,
                                        **self.mip_solver_kwargs)
        self.obj.activate()
        for c in self.nonlinear_constraints:
            c.activate()
        m.MINDT_linear_cuts.deactivate()
        m.MINDT_oa_obj.deactivate()

        # Process master problem result
        master_terminate_cond = results.solver.termination_condition
        if master_terminate_cond is tc.optimal:
            # proceed. Just need integer values
            m.solutions.load_from(results)
            if self.obj.sense == minimize:
                self.LB = max(value(m.MINDT_oa_obj.expr), self.LB)
            else:
                self.UB = min(value(m.MINDT_oa_obj.expr), self.UB)
            print('MIP {}: OBJ: {}  LB: {}  UB: {}'
                  .format(self.mip_iter, value(self.obj.expr), self.LB,
                          self.UB))
        elif master_terminate_cond is tc.infeasible:
            print('MILP master problem is infeasible. '
                  'Problem may have no more feasible binary configurations.')
            if self.mip_iter == 1:
                print('MINDT initialization may have generated poor '
                      'quality cuts.')
            # set optimistic bound to infinity
            if self.obj.sense == minimize:
                self.LB = float('inf')
            else:
                self.UB = float('-inf')
        else:
            raise ValueError(
                'MINDT unable to handle MILP master termination condition '
                'of {}. Solver message: {}'.format(
                    master_terminate_cond, results.solver.message))

        # Call the MILP post-solve callback
        self.master_postsolve(m, self)

    def _solve_OA_subproblem(self):
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
                self.UB_old = self.UB
                self.UB = min(value(self.obj.expr), self.UB)
                self.solution_improved = self.UB < self.UB_old
            else:
                self.LB_old = self.LB
                self.LB = max(value(self.obj.expr), self.LB)
                self.solution_improved = self.LB > self.LB_old
            print('NLP {}: OBJ: {}  LB: {}  UB: {}'
                  .format(self.nlp_iter, value(self.obj.expr), self.LB,
                          self.UB))
            if self.solution_improved:
                self.best_solution_found = m.clone()
            self._add_oa_cut()

            self.subproblem_postfeasible(m, self)
        elif subprob_terminate_cond is tc.infeasible:
            # load the solution, but suppress the warning message by setting
            # solver status to ok
            print('NLP subproblem was locally infeasible.')
            results.solver.status = SolverStatus.ok
            m.solutions.load_from(results)

            # TODO try something else? Reinitialize?
            self._add_int_cut()
        elif subprob_terminate_cond is tc.maxIterations:
            print('NLP subproblem failed to converge within iteration limit.')
            self._add_int_cut()
        else:
            raise ValueError(
                'MINDT unable to handle NLP subproblem termination '
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
        m.MINDT_linear_cuts.nlp_iters.add(self.nlp_iter)
        if self.obj.sense == minimize:
            sign_adjust = -1
        else:
            sign_adjust = 1

        # generate new constraints
        # TODO some kind of special handling if the dual is phenomenally small?
        for constr in self.jacs:
            c = m.MINDT_linear_cuts.oa_cuts.add(
                expr=copysign(1, sign_adjust * m.dual[constr]) * sum(
                    value(self.jacs[constr][id(var)]) * (var - value(var))
                    for var in list(EXPR.identify_variables(constr.body))) +
                m.MINDT_linear_cuts.slack_vars[self.nlp_iter,
                                               self.nl_map[constr]] <= 0)
            self.OA_constr_map[constr, self.nlp_iter] = c

    def _add_int_cut(self):
        m = self.m
        int_tol = 1E-6
        # check to make sure that binary variables are all 0 or 1
        for v in self.binary_vars:
            if value(abs(v - 1)) > int_tol and value(abs(v)) > int_tol:
                raise ValueError('Binary {} = {} is not 0 or 1'.format(
                    v.name, value(v)))

        # Add the integer cut
        m.MINDT_linear_cuts.integer_cuts.add(
            expr=sum(1 - v for v in self.binary_vars
                     if value(abs(v - 1)) <= int_tol) +
            sum(v for v in self.binary_vars if value(abs(v)) <= int_tol) >= 1)

    def _nonlinear_vars(self, blk):
        """Return the variables that participate in nonlinear terms.

        Args:
            blk (Block): Pyomo block

        Returns:
            list(_VarData): list of Pyomo var_data objects

        """
        nl_vars = []
        # This is a workaround because Var is not hashable, and I do not want
        # duplicates in nl_vars.
        seen = set()
        for constr in self.nonlinear_constraints:
            if isinstance(constr.body, EXPR._SumExpression):
                # go through each term and check to see if the term is
                # nonlinear
                for expr in EXPR._SumExpression._args:
                    # Check to see if the expression is nonlinear
                    if expr.polynomial_degree() not in (0, 1):
                        # collect variables
                        for var in EXPR.identify_variables(
                                expr, include_fixed=False):
                            if id(var) not in seen:
                                seen.add(id(var))
                                nl_vars.append(var)
            # if the root expression object is not a summation, then something
            # else is the cause of the nonlinearity. Collect all participating
            # variables.
            else:
                # collect variables
                for var in EXPR.identify_variables(constr.body,
                                                   include_fixed=False):
                    if id(var) not in seen:
                        seen.add(id(var))
                        nl_vars.append(var)

        return nl_vars
