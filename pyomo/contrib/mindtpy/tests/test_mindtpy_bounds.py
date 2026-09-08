# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Check the result when a no-good cut exhausts feasible integer assignments."""

import pyomo.common.unittest as unittest
import pyomo.environ as pyo


@unittest.skipUnless(
    all(
        pyo.SolverFactory(solver).available(exception_flag=False)
        for solver in ('glpk', 'ipopt')
    ),
    'GLPK and IPOPT are required',
)
class TestExhaustedAssignments(unittest.TestCase):
    def _solve(self, sense, mip_solver='glpk', single_tree=False, use_tabu_list=False):
        # Both models are convex optimization problems with one feasible binary
        # assignment and a known optimum of +1 (minimize) or -1 (maximize).
        model = pyo.ConcreteModel()
        model.y = pyo.Var(domain=pyo.Binary, initialize=0)
        model.x = pyo.Var(bounds=(1, 2), initialize=1.5)
        model.only_assignment = pyo.Constraint(expr=model.y <= 0)
        model.objective = pyo.Objective(
            expr=int(sense) * (model.x**2 + model.y), sense=sense
        )
        results = pyo.SolverFactory('mindtpy').solve(
            model,
            strategy='OA',
            mip_solver=mip_solver,
            nlp_solver='ipopt',
            init_strategy='initial_binary',
            add_no_good_cuts=not use_tabu_list,
            use_tabu_list=use_tabu_list,
            single_tree=single_tree,
            use_fbbt=False,
        )
        self.assertAlmostEqual(pyo.value(model.objective), int(sense), places=6)
        self.assertEqual(
            results.solver.termination_condition, pyo.TerminationCondition.optimal
        )
        self.assertAlmostEqual(results.problem.lower_bound, int(sense), places=6)
        self.assertAlmostEqual(results.problem.upper_bound, int(sense), places=6)

    def test_minimize(self):
        self._solve(pyo.minimize)

    def test_maximize(self):
        self._solve(pyo.maximize)

    @unittest.skipUnless(
        pyo.SolverFactory('cplex_persistent').available(exception_flag=False),
        'CPLEX is required',
    )
    def test_cplex_exclusions(self):
        for sense in (pyo.minimize, pyo.maximize):
            for single_tree in (False, True):
                for use_tabu_list in (False, True):
                    with self.subTest(
                        sense=sense,
                        single_tree=single_tree,
                        use_tabu_list=use_tabu_list,
                    ):
                        self._solve(
                            sense,
                            mip_solver='cplex_persistent',
                            single_tree=single_tree,
                            use_tabu_list=use_tabu_list,
                        )

    def test_unsolved_assignment_is_not_infeasible(self):
        model = pyo.ConcreteModel()
        model.y = pyo.Var(domain=pyo.Binary, initialize=0)
        model.x = pyo.Var(bounds=(1, 2), initialize=1.5)
        model.only_assignment = pyo.Constraint(expr=model.y <= 0)
        model.objective = pyo.Objective(expr=model.x**2 + model.y)
        results = pyo.SolverFactory('mindtpy').solve(
            model,
            mip_solver='glpk',
            nlp_solver='ipopt',
            nlp_solver_args={'options': {'max_iter': 0}},
            init_strategy='initial_binary',
            add_no_good_cuts=True,
            use_fbbt=False,
        )
        self.assertEqual(
            results.solver.termination_condition, pyo.TerminationCondition.noSolution
        )
        self.assertEqual(results.problem.lower_bound, -float('inf'))
        self.assertEqual(results.problem.upper_bound, float('inf'))

    def test_unsolved_assignment_does_not_close_gap(self):
        for sense in (pyo.minimize, pyo.maximize):
            with self.subTest(sense=sense), pyo.SolverFactory('mindtpy.oa') as opt:
                model = pyo.ConcreteModel()
                model.y = pyo.Var(domain=pyo.Binary, initialize=1)
                model.z = pyo.Var(domain=pyo.Binary, initialize=0)
                model.x = pyo.Var(bounds=(1, 2), initialize=1.5)
                model.assignments = pyo.Constraint(expr=model.y + model.z <= 1)
                model.objective = pyo.Objective(
                    expr=int(sense) * (model.x**2 + 2 * model.y + model.z), sense=sense
                )
                limited_assignments = []

                def limit_best_assignment(fixed_nlp):
                    # Interrupt the NLP for the true best assignment (0, 0).
                    # The other two assignments can still produce incumbents.
                    best_assignment = (
                        pyo.value(fixed_nlp.y) + pyo.value(fixed_nlp.z) < 0.5
                    )
                    opt.nlp_opt.options['max_iter'] = 0 if best_assignment else 1000
                    if best_assignment:
                        limited_assignments.append((0, 0))

                results = opt.solve(
                    model,
                    mip_solver='glpk',
                    nlp_solver='ipopt',
                    init_strategy='initial_binary',
                    add_no_good_cuts=True,
                    use_fbbt=False,
                    call_before_subproblem_solve=limit_best_assignment,
                )
                self.assertEqual(limited_assignments, [(0, 0)])
                self.assertAlmostEqual(
                    pyo.value(model.objective), 2 * int(sense), places=6
                )
                self.assertEqual(
                    results.solver.termination_condition,
                    pyo.TerminationCondition.feasible,
                )
                # The reported interval must still contain the true optimum,
                # even though that assignment was excluded after the NLP limit.
                self.assertLessEqual(results.problem.lower_bound, int(sense) + 1e-6)
                self.assertGreaterEqual(results.problem.upper_bound, int(sense) - 1e-6)
