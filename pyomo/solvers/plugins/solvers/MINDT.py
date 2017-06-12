"""Interface to the MINDT solver.

The MINDT (MINLP Decomposition Tookit) solver applies a variety of
decomposition-based approaches to solve nonlinear continuous-discrete problems.
These approaches include:

- Outer approximation
- Benders decomposition (unimplemented)
- Partial surrogate cuts (unimplemented)

TODO: currently no support for integer variables. Need to raise exception.

"""
import pyomo.util.plugin
from pyomo.opt.base import IOptSolver
from pyomo.environ import (Block, ConstraintList, Suffix, Binary, Var, value,
                           NonNegativeReals, Constraint, minimize,
                           Objective, Param, Reals)
from pyomo.opt import SolverFactory, TerminationCondition
from pyomo.core.base import expr as EXPR
from pyomo.core.base.symbolic import differentiate
optimal = TerminationCondition.optimal
infeasible = TerminationCondition.infeasible


class MINDTSolver(pyomo.util.plugin.Plugin):
    """A decomposition-based MINLP solver."""

    pyomo.util.plugin.implements(IOptSolver)
    pyomo.util.plugin.alias('mindt',
                            doc='The MINDT decomposition-based MINLP solver')

    # def available(self, exception_flag=True):
    #     return True

    def solve(self, m, tol=1E-6, iterlim=1000, **kwds):
        """Solve everything.

        All the problems. World peace? Done. Cure for cancer? You bet.

        Args:
            m (Block): a Pyomo model or block to be solved
        """
        lin = m.MINDT_linear_cuts = Block()
        lin.deactivate()
        lin.integer_cuts = ConstraintList(doc='integer cuts')
        # lin.
        lin.oa_cuts = ConstraintList(doc='OA_cuts')
        binary_vars = [v for v in m.component_data_objects(
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
            obj = Objective(expr=m.MINDT_objective_value, sense=main_obj.sense)
        else:
            obj = main_obj

        # TODO if any continuous variables are multipled with binary ones, need
        # to do some kind of transformation or throw an error message

        nonlinear_constraints = [
            v for v in m.component_data_objects(
                ctype=Constraint, active=True, descend_into=True)
            if v.body.polynomial_degree() not in (0, 1)]
        # preload jacobians
        jacs = self._get_jacobians(nonlinear_constraints)

        nlp_iter = 0
        mip_iter = 0
        if not hasattr(m, 'dual'):
            m.dual = Suffix(direction=Suffix.IMPORT_EXPORT)

        # set up bounds
        lower_bound = m.MINDT_LB = Param(initialize=float('-inf'),
                                         mutable=True)
        upper_bound = m.MINDT_UB = Param(initialize=float('inf'), mutable=True)
        if obj.sense == minimize:
            optimistic_bound = lower_bound
            feasible_bound = upper_bound
        else:
            optimistic_bound = upper_bound
            feasible_bound = lower_bound

        # Set up solvers
        nlp_solver = SolverFactory('ipopt')
        mip_solver = SolverFactory('gurobi')

        # Initialization

        # solve rNLP (relaxation of binary variables)
        nlp_iter += 1
        print("Solve relaxed NLP")
        for v in binary_vars:
            v.domain = NonNegativeReals
            v.setlb(0)
            v.setub(1)
        results = nlp_solver.solve(m)
        for v in binary_vars:
            v.domain = Binary
        subprob_terminate_cond = results.solver.termination_condition
        if subprob_terminate_cond is optimal:
            # Add OA cut
            optimistic_bound = value(obj.expr)
            self._add_oa_cut(m, nlp_iter, jacs, obj.sense == minimize)
        elif subprob_terminate_cond is infeasible:
            # TODO fail? try something else?
            raise ValueError('Initial relaxed NLP infeasible. '
                             'Problem may be infeasible.')
        else:
            raise ValueError(
                'MINDT unable to handle relaxed NLP termination condition '
                'of {}'.format(subprob_terminate_cond))

        # Algorithm main loop
        while value(lower_bound) + tol < value(upper_bound) \
                and mip_iter < iterlim:
            # solve MILP master problem
            mip_iter += 1
            print('Solve MILP master problem. Iter {}'.format(mip_iter))
            # Set up MILP
            for c in nonlinear_constraints:
                c.deactivate()
            results = mip_solver.solve(m)
            for c in nonlinear_constraints:
                c.activate()
            master_terminate_cond = results.solver.termination_condition
            if master_terminate_cond is optimal:
                # proceed. Just need integer values
                pass
            elif master_terminate_cond is infeasible:
                # set optimistic bound to infinity
                if optimistic_bound is lower_bound:
                    lower_bound = float('inf')
                else:
                    upper_bound = float('-inf')
            else:
                raise ValueError(
                    'MINDT unable to handle MILP master termination condition '
                    'of {}'.format(master_terminate_cond))

            # Solve NLP subproblem
            nlp_iter += 1
            print('Solve NLP subproblem. Iter {}'.format(nlp_iter))
            # Set up NLP
            for v in binary_vars:
                v.fix()
            results = nlp_solver.solve(m)
            subprob_terminate_cond = results.solver.termination_condition
            if subprob_terminate_cond is optimal:
                if feasible_bound is upper_bound:
                    upper_bound = min(value(obj.expr), value(upper_bound))
                else:
                    lower_bound = max(value(obj.expr), value(lower_bound))
                self._add_oa_cut(m, nlp_iter, jacs, obj.sense == minimize)
            elif subprob_terminate_cond is infeasible:
                # TODO try something else? Reinitialize?
                self._add_int_cut(m, binary_vars)
            else:
                raise ValueError(
                    'MINDT unable to handle NLP subproblem termination '
                    'condition of {}'.format(subprob_terminate_cond))

    def _get_jacobians(nonlinear_constraints):
        jacs = {}
        for c in nonlinear_constraints:
            constraint_vars = list(EXPR.identify_variables(c.body))
            jac_list = differentiate(c.body, wrt=constraint_vars)
            jacs[c] = {id(var): jac
                       for var, jac in zip(constraint_vars, jac_list)}
        return jacs

    def _add_oa_cut(m, nlp_iter, jacs, obj_is_minimize):
        if obj_is_minimize:
            sign_adjust = -1
        else:
            sign_adjust = 1

        # generate new constraints and slack variables
        for constr in jacs:
            pass

    def _add_int_cut(m, binary_vars):
        int_tol = 1E-6
        # TODO raise an error if any of the binary variables are not 0 or 1
        m.MINDT_linear_cuts.integer_cuts.add(
            expr=sum(1 - v for v in binary_vars
                     if value(abs(v) - 1) <= int_tol) +
            sum(v for v in binary_vars if value(abs(v)) <= int_tol) >= 1)
