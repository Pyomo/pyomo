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

import pyomo.common.unittest as unittest
from pyomo.common.fileutils import Executable

from pyomo.contrib.cp import IntervalVar, Pulse, Step, AlwaysIn
from pyomo.contrib.cp.repn.docplex_writer import LogicalToDoCplex
from pyomo.environ import (
    ConcreteModel,
    Set,
    Var,
    Integers,
    LogicalConstraint,
    implies,
    value,
    TerminationCondition,
    Constraint,
    PositiveIntegers,
    maximize,
    minimize,
    Objective,
)
from pyomo.opt import WriterFactory, SolverFactory

try:
    import docplex.cp.model as cp

    docplex_available = True
except:
    docplex_available = False

cpoptimizer_available = Executable('cpoptimizer').available()


@unittest.skipIf(not docplex_available, "docplex is not available")
class TestWriteModel(unittest.TestCase):
    def test_write_scheduling_model_only_interval_vars(self):
        m = ConcreteModel()
        m.i = IntervalVar(start=(2, 4), end=(5, 19), length=7, optional=False)
        m.tasks = Set(initialize=range(2))
        m.h = IntervalVar(m.tasks, optional=True, length=(4, 5), start=(1, 2))

        cpx_mod, var_map = WriterFactory('docplex_model').write(m)

        # We have nothing on this model other than interval vars
        exprs = cpx_mod.get_all_expressions()
        self.assertEqual(len(exprs), 3)

        # We should have the three interval vars above
        variables = cpx_mod.get_all_variables()
        self.assertEqual(len(variables), 3)
        # I'm assuming that the lists of exprs and vars are in a deterministic
        # order. If they're not, this will fail periodically, so I guess we'll
        # find out.
        self.assertIs(variables[0], var_map[m.h[1]])
        self.assertIs(exprs[2][0], var_map[m.h[1]])
        self.assertIs(variables[1], var_map[m.h[0]])
        self.assertIs(exprs[1][0], var_map[m.h[0]])

        for i in [0, 1]:
            self.assertTrue(variables[i].is_optional())
            self.assertEqual(variables[i].get_start(), (1, 2))
            self.assertEqual(variables[i].get_length(), (4, 5))

        self.assertIs(variables[2], var_map[m.i])
        self.assertIs(exprs[0][0], var_map[m.i])
        self.assertTrue(variables[2].is_present())
        self.assertEqual(variables[2].get_start(), (2, 4))
        self.assertEqual(variables[2].get_end(), (5, 19))
        self.assertEqual(variables[2].get_length(), (7, 7))

    def test_write_model_with_bool_expr_as_constraint(self):
        # This tests our handling of a quirk with docplex that even some things
        # that are boolean-valued can't be added to the model as constraints. We
        # need to explicitly recognize them and add an "== True" right-hand
        # side.
        m = ConcreteModel()
        m.i = IntervalVar([1, 2], optional=True)
        m.x = Var(within={1, 2})
        # This is a perfectly reasonable constraint in a context where x is some
        # variable that decides what needs to be scheduled.
        m.cons = LogicalConstraint(expr=m.i[m.x].is_present)

        cpx_mod, var_map = WriterFactory('docplex_model').write(m)

        variables = cpx_mod.get_all_variables()
        self.assertEqual(len(variables), 3)
        # The three variables plus the one constraint:
        exprs = cpx_mod.get_all_expressions()
        self.assertEqual(len(exprs), 4)

        x = var_map[m.x]
        i1 = var_map[m.i[1]]
        i2 = var_map[m.i[2]]

        self.assertIs(variables[0], x)
        self.assertIs(variables[1], i2)
        self.assertIs(variables[2], i1)

        self.assertTrue(
            exprs[3][0].equals(
                cp.element(
                    [cp.presence_of(i1), cp.presence_of(i2)], 0 + 1 * (x - 1) // 1
                )
                == True
            )
        )


@unittest.skipIf(not docplex_available, "docplex is not available")
@unittest.skipIf(not cpoptimizer_available, "CP optimizer is not available")
class TestSolveModel(unittest.TestCase):
    def test_solve_scheduling_problem(self):
        m = ConcreteModel()
        m.eat_cookie = IntervalVar([0, 1], length=8, end=(0, 24), optional=False)
        m.eat_cookie[0].start_time.bounds = (0, 4)
        m.eat_cookie[1].start_time.bounds = (5, 20)

        m.read_story = IntervalVar(start=(15, 24), end=(0, 24), length=(2, 3))
        m.sweep_crumbs = IntervalVar(optional=True, length=1, end=(0, 24))
        m.do_dishes = IntervalVar(optional=True, length=5, end=(0, 24))

        m.num_crumbs = Var(domain=Integers, bounds=(0, 100))

        ## Precedence
        m.cookies = LogicalConstraint(
            expr=m.eat_cookie[1].start_time.after(m.eat_cookie[0].end_time)
        )
        m.cookies_imply_crumbs = LogicalConstraint(
            expr=m.eat_cookie[0].is_present.implies(m.num_crumbs == 5)
        )
        m.good_mouse = LogicalConstraint(
            expr=implies(m.num_crumbs >= 3, m.sweep_crumbs.is_present)
        )
        m.sweep_after = LogicalConstraint(
            expr=m.sweep_crumbs.start_time.after(m.eat_cookie[1].end_time)
        )

        m.mice_occupied = (
            sum(Pulse((m.eat_cookie[i], 1)) for i in range(2))
            + Step(m.read_story.start_time, 1)
            + Pulse((m.sweep_crumbs, 1))
            - Pulse((m.do_dishes, 1))
        )

        # Must keep exactly one mouse occupied for a 25-hour day
        m.treat_your_mouse_well = LogicalConstraint(
            expr=AlwaysIn(cumul_func=m.mice_occupied, bounds=(1, 1), times=(0, 24))
        )

        results = SolverFactory('cp_optimizer').solve(
            m, symbolic_solver_labels=True, tee=True
        )

        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.feasible
        )

        # check solution
        self.assertTrue(value(m.eat_cookie[0].is_present))
        self.assertTrue(value(m.eat_cookie[1].is_present))
        # That means there were crumbs:
        self.assertEqual(value(m.num_crumbs), 5)
        # So there was sweeping:
        self.assertTrue(value(m.sweep_crumbs.is_present))

        # start with the first cookie:
        self.assertEqual(value(m.eat_cookie[0].start_time), 0)
        self.assertEqual(value(m.eat_cookie[0].end_time), 8)
        self.assertEqual(value(m.eat_cookie[0].length), 8)
        # Proceed to second cookie:
        self.assertEqual(value(m.eat_cookie[1].start_time), 8)
        self.assertEqual(value(m.eat_cookie[1].end_time), 16)
        self.assertEqual(value(m.eat_cookie[1].length), 8)
        # Sweep
        self.assertEqual(value(m.sweep_crumbs.start_time), 16)
        self.assertEqual(value(m.sweep_crumbs.end_time), 17)
        self.assertEqual(value(m.sweep_crumbs.length), 1)
        # End with read story, as it keeps exactly one mouse occupied
        # indefinitely (in this particular retelling)
        self.assertEqual(value(m.read_story.start_time), 17)

        # Since doing the dishes actually *bores* a mouse, we leave the dishes
        # in the sink
        self.assertFalse(value(m.do_dishes.is_present))

        self.assertEqual(results.problem.number_of_objectives, 0)
        self.assertEqual(results.problem.number_of_constraints, 5)
        self.assertEqual(results.problem.number_of_integer_vars, 1)
        self.assertEqual(results.problem.number_of_interval_vars, 5)

    def test_solve_infeasible_problem(self):
        m = ConcreteModel()
        m.x = Var(within=[1, 2, 3, 5])
        m.c = Constraint(expr=m.x == 0)

        result = SolverFactory('cp_optimizer').solve(m)
        self.assertEqual(
            result.solver.termination_condition, TerminationCondition.infeasible
        )

        self.assertIsNone(m.x.value)

    def test_solve_max_problem(self):
        m = ConcreteModel()
        m.cookies = Var(domain=PositiveIntegers, bounds=(7, 10))
        m.chocolate_chip_equity = Constraint(expr=m.cookies <= 9)

        m.obj = Objective(expr=m.cookies, sense=maximize)

        results = SolverFactory('cp_optimizer').solve(m)

        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.optimal
        )
        self.assertEqual(value(m.cookies), 9)

        self.assertEqual(results.problem.number_of_objectives, 1)
        self.assertEqual(results.problem.sense, maximize)
        self.assertEqual(results.problem.lower_bound, 9)
        self.assertEqual(results.problem.upper_bound, 9)

    def test_solve_min_problem(self):
        m = ConcreteModel()
        m.x = Var([1, 2, 3], bounds=(4, 6), domain=Integers)
        m.y = Var(within=[1, 2, 3])

        m.c1 = Constraint(expr=m.y >= 2.5)

        @m.Constraint([1, 2, 3])
        def x_bounds(m, i):
            return m.x[i] >= 3 * (i - 1)

        m.obj = Objective(expr=m.x[m.y])

        results = SolverFactory('cp_optimizer').solve(m)

        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.optimal
        )
        self.assertEqual(value(m.x[3]), 6)
        self.assertEqual(value(m.y), 3)

        self.assertEqual(results.problem.number_of_objectives, 1)
        self.assertEqual(results.problem.sense, minimize)
        self.assertEqual(results.problem.lower_bound, 6)
        self.assertEqual(results.problem.upper_bound, 6)
