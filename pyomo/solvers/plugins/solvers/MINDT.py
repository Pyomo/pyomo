"""Implementation of the MINDT solver.

The MINDT (MINLP Decomposition Tookit) solver applies a variety of
decomposition-based approaches to solve nonlinear continuous-discrete problems.
These approaches include:

- Outer approximation
- Benders decomposition (unimplemented)
- Partial surrogate cuts (unimplemented)

This solver implementation was developed by Carnegie Mellon University in the
research group of Ignacio Grossmann.

Questions: Qi Chen <qichen at andrew.cmu.edu>

"""
from math import copysign

import pyomo.util.plugin
from pyomo.core.base import expr as EXPR
from pyomo.core.base.symbolic import differentiate
from pyomo.environ import (Binary, Block, Constraint, ConstraintList,
                           Expression, Integers, NonNegativeReals, Objective,
                           RangeSet, Reals, Set, Suffix, Var, minimize, value)
from pyomo.opt import SolverFactory, TerminationCondition
from pyomo.opt.base import IOptSolver

optimal = TerminationCondition.optimal
infeasible = TerminationCondition.infeasible
maxIterations = TerminationCondition.maxIterations


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

    def solve(self, m, **kwds):
        """Solve everything.

        All the problems. World peace? Done. Cure for cancer? You bet.

        Args:
            m (Block): a Pyomo model or block to be solved

        Kwargs:
            tol (float): bound tolerance
            iterlim (int): maximum number of master iterations
            strategy (str): decomposition strategy to use. Possible values:
                OA, PSC, GBD
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
        self.max_slack = kwds.pop('max_slack', 1000)
        self.OA_penalty_factor = kwds.pop('OA_penalty', 1000)
        self.nlp_solver_name = kwds.pop('nlp', 'ipopt')
        self.nlp_solver_kwargs = kwds.pop('nlp_kwargs', {})
        self.mip_solver_name = kwds.pop('mip', 'gurobi')
        self.mip_solver_kwargs = kwds.pop('mip_kwargs', {})
        self.mip_post_solve = kwds.pop('mip_post_solve', _do_nothing)

        # Check for any integer variables
        if any(True for v in m.component_data_objects(
                ctype=Var, active=True, descend_into=True)
                if v.domain is Integers and not v.fixed):
            raise ValueError('Model contains unfixed integer variables. '
                             'MINDT does not currently support solution of '
                             'such problems.')
            # TODO add in a reformulation?

        # Create a model block in which to store the generated linear
        # constraints
        lin = m.MINDT_linear_cuts = Block()
        lin.deactivate()  # do not leave the constraints on by default
        # Integer cuts exclude particular discrete decisions
        lin.integer_cuts = ConstraintList(doc='integer cuts')
        # Build a list of binary variables
        self.binary_vars = [v for v in m.component_data_objects(
            ctype=Var, active=True, descend_into=True)
            if v.domain is Binary and not v.fixed]

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

        # Build list of nonlinear constraints
        self.nonlinear_constraints = [
            v for v in m.component_data_objects(
                ctype=Constraint, active=True, descend_into=True)
            if v.body.polynomial_degree() not in (0, 1)]

        if self.decomposition_strategy == 'OA':
            self._calc_jacobians()  # preload jacobians
            lin.oa_cuts = ConstraintList(doc='OA_cuts')

        # Set of NLP iterations for which cuts were generated
        lin.nlp_iters = Set(dimen=1)

        # Create an integer index set over the nonlinear constraints
        lin.nl_constraint_set = RangeSet(len(self.nonlinear_constraints))
        self.nl_map = {}  # dict to map constraint to integer index
        self.nl_inverse_map = {}  # dict to map integer index to constraint
        for c, n in zip(self.nonlinear_constraints, lin.nl_constraint_set):
            self.nl_map[c] = n
            self.nl_inverse_map[n] = c

        # Create slack variables
        lin.slack_vars = Var(lin.nlp_iters, lin.nl_constraint_set,
                             domain=NonNegativeReals,
                             bounds=(0, self.max_slack), initialize=0)

        # Set up iteration counters
        self.nlp_iter = 0
        self.mip_iter = 0

        # Set up dual value reporting
        if not hasattr(m, 'dual'):
            m.dual = Suffix(direction=Suffix.IMPORT_EXPORT)

        # set up bounds
        self.LB = float('-inf')
        self.UB = float('inf')

        # Set up solvers
        self.nlp_solver = SolverFactory(self.nlp_solver_name)
        self.mip_solver = SolverFactory(self.mip_solver_name)

        if self.decomposition_strategy == 'OA':
            self.OA_constr_map = {}
            self._initialize_OA(m)  # Initialization

        # Algorithm main loop
        while self.LB + self.bound_tolerance < self.UB:
            if self.mip_iter >= self.iteration_limit:
                print('MINDT unable to solve problem '
                      'after {} master iterations'.format(self.mip_iter))
                break
            # solve MILP master problem
            if self.decomposition_strategy == 'OA':
                self._solve_OA_master(m)
            # Call the MILP post-solve callback
            self.mip_post_solve(m, self)
            # Solve NLP subproblem
            if self.decomposition_strategy == 'OA':
                self._solve_OA_subproblem(m)

    def _initialize_OA(self, m):
        # solve rNLP (relaxation of binary variables)
        self.nlp_iter += 1
        print("NLP {}: Solve relaxed integrality.".format(self.nlp_iter))
        for v in self.binary_vars:
            v.domain = NonNegativeReals
            v.setlb(0)
            v.setub(1)
        results = self.nlp_solver.solve(m, **self.nlp_solver_kwargs)
        for v in self.binary_vars:
            v.domain = Binary
        subprob_terminate_cond = results.solver.termination_condition
        if subprob_terminate_cond is optimal:
            # Add OA cut
            if self.obj.sense == minimize:
                self.LB = value(self.obj.expr)
            else:
                self.UB = value(self.obj.expr)
            print('Objective {}'.format(value(self.obj.expr)))
            self._add_oa_cut(m)
        elif subprob_terminate_cond is infeasible:
            # TODO fail? try something else?
            raise ValueError('Initial relaxed NLP infeasible. '
                             'Problem may be infeasible.')
        else:
            raise ValueError(
                'MINDT unable to handle relaxed NLP termination condition '
                'of {}'.format(subprob_terminate_cond))

    def _solve_OA_master(self, m):
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
        results = self.mip_solver.solve(m, **self.mip_solver_kwargs)
        self.obj.activate()
        for c in self.nonlinear_constraints:
            c.activate()
        m.MINDT_linear_cuts.deactivate()
        m.MINDT_oa_obj.deactivate()

        # Process master problem result
        master_terminate_cond = results.solver.termination_condition
        if master_terminate_cond is optimal:
            # proceed. Just need integer values
            print('Objective {}'.format(value(m.MINDT_oa_obj.expr)))
            if self.obj.sense == minimize:
                self.LB = max(value(m.MINDT_oa_obj.expr), self.LB)
            else:
                self.UB = min(value(m.MINDT_oa_obj.expr), self.UB)
        elif master_terminate_cond is infeasible:
            # set optimistic bound to infinity
            if self.obj.sense == minimize:
                self.LB = float('inf')
            else:
                self.UB = float('-inf')
        else:
            raise ValueError(
                'MINDT unable to handle MILP master termination condition '
                'of {}'.format(master_terminate_cond))

    def _solve_OA_subproblem(self, m):
        self.nlp_iter += 1
        print('NLP {}: Solve subproblem.'.format(self.nlp_iter))
        # Set up NLP
        for v in self.binary_vars:
            v.fix()
        results = self.nlp_solver.solve(m, **self.nlp_solver_kwargs)
        for v in self.binary_vars:
            v.unfix()
        subprob_terminate_cond = results.solver.termination_condition
        if subprob_terminate_cond is optimal:
            if self.obj.sense == minimize:
                self.UB = min(value(self.obj.expr), self.UB)
            else:
                self.LB = max(value(self.obj.expr), self.LB)
            print('Objective {}'.format(value(self.obj.expr)))
            self._add_oa_cut(m)
        elif subprob_terminate_cond is infeasible:
            # TODO try something else? Reinitialize?
            self._add_int_cut(m)
        elif subprob_terminate_cond is maxIterations:
            self._add_int_cut(m)
        else:
            raise ValueError(
                'MINDT unable to handle NLP subproblem termination '
                'condition of {}'.format(subprob_terminate_cond))

    def _calc_jacobians(self):
        self.jacs = {}
        for c in self.nonlinear_constraints:
            constraint_vars = list(EXPR.identify_variables(c.body))
            jac_list = differentiate(c.body, wrt_list=constraint_vars)
            self.jacs[c] = {id(var): jac
                            for var, jac in zip(constraint_vars, jac_list)}

    def _add_oa_cut(self, m):
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

    def _add_int_cut(self, m):
        int_tol = 1E-6
        # TODO raise an error if any of the binary variables are not 0 or 1
        m.MINDT_linear_cuts.integer_cuts.add(
            expr=sum(1 - v for v in self.binary_vars
                     if value(abs(v) - 1) <= int_tol) +
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
