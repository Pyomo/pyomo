"""Implementation of the MINDT solver.

The MINDT (MINLP Decomposition Tookit) solver applies a variety of
decomposition-based approaches to solve nonlinear continuous-discrete problems.
These approaches include:

- Outer approximation
- Benders decomposition (unimplemented)
- Partial surrogate cuts (unimplemented)

TODO: currently no support for integer variables. Need to raise exception.

This solver implementation was developed by Carnegie Mellon University in the
research group of Ignacio Grossmann.

Questions: Qi Chen <qichen at andrew.cmu.edu>

"""
from math import copysign

import pyomo.util.plugin
from pyomo.core.base import expr as EXPR
from pyomo.core.base.symbolic import differentiate
from pyomo.environ import (Binary, Block, Constraint, ConstraintList,
                           NonNegativeReals, Objective, RangeSet, Reals, Set,
                           Suffix, Var, minimize, value)
from pyomo.opt import SolverFactory, TerminationCondition
from pyomo.opt.base import IOptSolver

optimal = TerminationCondition.optimal
infeasible = TerminationCondition.infeasible


class MINDTSolver(pyomo.util.plugin.Plugin):
    """A decomposition-based MINLP solver."""

    pyomo.util.plugin.implements(IOptSolver)
    pyomo.util.plugin.alias('mindt',
                            doc='The MINDT decomposition-based MINLP solver')

    # def available(self, exception_flag=True):
    #     return True

    def solve(self, m, tol=1E-6, iterlim=30, **kwds):
        """Solve everything.

        All the problems. World peace? Done. Cure for cancer? You bet.

        Args:
            m (Block): a Pyomo model or block to be solved
        """
        lin = m.MINDT_linear_cuts = Block()
        lin.deactivate()
        lin.integer_cuts = ConstraintList(doc='integer cuts')
        self.binary_vars = [v for v in m.component_data_objects(
            ctype=Var, active=True, descend_into=True)
            if v.domain is Binary and not v.fixed]

        # Prepare problem objective
        objs = m.component_data_objects(
            ctype=Objective, active=True, descend_into=True)
        main_obj = next(objs, None)
        if main_obj is None:
            raise ValueError('Model has no active objectives.')
        if next(objs, None) is not None:
            raise ValueError('Model has multiple active objectives.')

        # If objective is nonlinear, move it into the constraints
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

        self.nonlinear_constraints = [
            v for v in m.component_data_objects(
                ctype=Constraint, active=True, descend_into=True)
            if v.body.polynomial_degree() not in (0, 1)]
        # preload jacobians
        self._calc_jacobians()

        lin.oa_iters = Set(dimen=1)
        lin.oa_cuts = ConstraintList(doc='OA_cuts')
        lin.nl_constraint_set = RangeSet(len(self.nonlinear_constraints))
        self.nl_map = {}
        self.nl_inverse_map = {}
        for c, n in zip(self.nonlinear_constraints, lin.nl_constraint_set):
            self.nl_map[c] = n
            self.nl_inverse_map[n] = c
        max_slack = kwds.pop('max_slack', 1000)
        lin.slack_vars = Var(lin.oa_iters, lin.nl_constraint_set,
                             domain=NonNegativeReals, bounds=(0, max_slack))

        self.nlp_iter = 0
        self.mip_iter = 0
        if not hasattr(m, 'dual'):
            m.dual = Suffix(direction=Suffix.IMPORT_EXPORT)

        # set up bounds
        self.LB = float('-inf')
        self.UB = float('inf')

        # Set up solvers
        nlp_solver = SolverFactory('ipopt')
        mip_solver = SolverFactory('gurobi')

        # Initialization

        # solve rNLP (relaxation of binary variables)
        self.nlp_iter += 1
        print("NLP {}: Solve relaxed integrality.".format(self.nlp_iter))
        for v in self.binary_vars:
            v.domain = NonNegativeReals
            v.setlb(0)
            v.setub(1)
        results = nlp_solver.solve(m)
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

        # Algorithm main loop
        while self.LB + tol < self.UB:
            if self.mip_iter >= iterlim:
                print('MINDT unable to solve problem '
                      'after {} master iterations'.format(self.mip_iter))
                break
            # solve MILP master problem
            self.mip_iter += 1
            print('MILP {}: Solve master problem.'.format(self.mip_iter))
            # Set up MILP
            for c in self.nonlinear_constraints:
                c.deactivate()
            m.MINDT_linear_cuts.activate()
            results = mip_solver.solve(m)
            for c in self.nonlinear_constraints:
                c.activate()
            m.MINDT_linear_cuts.deactivate()
            master_terminate_cond = results.solver.termination_condition
            if master_terminate_cond is optimal:
                # proceed. Just need integer values
                print('Objective {}'.format(value(self.obj.expr)))
                if self.obj.sense == minimize:
                    self.LB = max(value(self.obj.expr), self.LB)
                else:
                    self.UB = min(value(self.obj.expr), self.UB)
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

            # Solve NLP subproblem
            self.nlp_iter += 1
            print('NLP {}: Solve subproblem.'.format(self.nlp_iter))
            # Set up NLP
            for v in self.binary_vars:
                v.fix()
            results = nlp_solver.solve(m)
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
        m.MINDT_linear_cuts.oa_iters.add(self.nlp_iter)
        if self.obj.sense == minimize:
            sign_adjust = -1
        else:
            sign_adjust = 1

        # generate new constraints and slack variables
        # TODO some kind of special handling if the dual is phenomenally small?
        for constr in self.jacs:
            m.MINDT_linear_cuts.oa_cuts.add(
                expr=copysign(1, sign_adjust * m.dual[constr]) * sum(
                    value(self.jacs[constr][id(var)]) * (var - value(var))
                    for var in list(EXPR.identify_variables(constr.body))) +
                m.MINDT_linear_cuts.slack_vars[self.nlp_iter,
                                               self.nl_map[constr]] <= 0
            )

    def _add_int_cut(self, m):
        int_tol = 1E-6
        # TODO raise an error if any of the binary variables are not 0 or 1
        m.MINDT_linear_cuts.integer_cuts.add(
            expr=sum(1 - v for v in self.binary_vars
                     if value(abs(v) - 1) <= int_tol) +
            sum(v for v in self.binary_vars if value(abs(v)) <= int_tol) >= 1)
