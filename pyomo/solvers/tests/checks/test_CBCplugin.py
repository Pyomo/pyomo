#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import os
import sys
from os.path import dirname, abspath

import pyutilib.th as unittest

from pyomo.environ import (ConcreteModel, Var, Objective, RangeSet,
                           Constraint, Reals, NonNegativeIntegers,
                           NonNegativeReals, Integers, Binary,
                           maximize, minimize)
from pyomo.opt import (SolverFactory, ProblemSense,
                       TerminationCondition, SolverStatus)

cbc_available = SolverFactory('cbc', solver_io='lp').available(exception_flag=False)

data_dir = '{}/data'.format(dirname(abspath(__file__)))


class TestCBC(unittest.TestCase):
    """
    These tests are here to test the general functionality of the cbc solver when using the lp solverio, which will
    ensure that we have a consistent output from CBC for some simple problems
    """

    def setUp(self):
        self.stderr = sys.stderr
        sys.stderr = None
        self.model = ConcreteModel()
        # Do we need to pass in seeds to ensure consistent behaviour? options={'randomSeed: 42, 'randomCbcSeed': 42}
        self.opt = SolverFactory("cbc", solver_io="lp")

    def tearDown(self):
        sys.stderr = self.stderr

    @unittest.skipIf(not cbc_available, "The 'cbc' solver is not available")
    def test_infeasible_lp(self):
        self.model.X = Var(within=Reals)
        self.model.C1 = Constraint(expr=self.model.X <= 1)
        self.model.C2 = Constraint(expr=self.model.X >= 2)
        self.model.Obj = Objective(expr=self.model.X, sense=minimize)

        results = self.opt.solve(self.model)

        self.assertEqual(ProblemSense.minimize, results.problem.sense)
        self.assertEqual(TerminationCondition.infeasible, results.solver.termination_condition)
        self.assertEqual('Model was proven to be infeasible.', results.solver.termination_message)
        self.assertEqual(SolverStatus.warning, results.solver.status)

    @unittest.skipIf(not cbc_available, "The 'cbc' solver is not available")
    def test_unbounded_lp(self):
        self.model.Idx = RangeSet(2)
        self.model.X = Var(self.model.Idx, within=Reals)
        self.model.Obj = Objective(expr=self.model.X[1] + self.model.X[2], sense=maximize)

        results = self.opt.solve(self.model)

        self.assertEqual(ProblemSense.maximize, results.problem.sense)
        self.assertEqual(TerminationCondition.unbounded, results.solver.termination_condition)
        self.assertEqual('Model was proven to be unbounded.', results.solver.termination_message)
        self.assertEqual(SolverStatus.warning, results.solver.status)

    @unittest.skipIf(not cbc_available, "The 'cbc' solver is not available")
    def test_optimal_lp(self):
        self.model.X = Var(within=NonNegativeReals)
        self.model.Obj = Objective(expr=self.model.X, sense=minimize)

        results = self.opt.solve(self.model)

        self.assertEqual(0.0, results.problem.lower_bound)
        self.assertEqual(0.0, results.problem.upper_bound)
        self.assertEqual(ProblemSense.minimize, results.problem.sense)
        self.assertEqual(TerminationCondition.optimal, results.solver.termination_condition)
        self.assertEqual(
            'Model was solved to optimality (subject to tolerances), and an optimal solution is available.',
            results.solver.termination_message)
        self.assertEqual(SolverStatus.ok, results.solver.status)

    @unittest.skipIf(not cbc_available, "The 'cbc' solver is not available")
    def test_infeasible_mip(self):
        self.model.X = Var(within=NonNegativeIntegers)
        self.model.C1 = Constraint(expr=self.model.X <= 1)
        self.model.C2 = Constraint(expr=self.model.X >= 2)
        self.model.Obj = Objective(expr=self.model.X, sense=minimize)

        results = self.opt.solve(self.model)

        self.assertEqual(ProblemSense.minimize, results.problem.sense)
        self.assertEqual(TerminationCondition.infeasible, results.solver.termination_condition)
        self.assertEqual('Model was proven to be infeasible.', results.solver.termination_message)
        self.assertEqual(SolverStatus.warning, results.solver.status)

    @unittest.skipIf(not cbc_available, "The 'cbc' solver is not available")
    def test_unbounded_mip(self):
        self.model.X = Var(within=Integers)
        self.model.Obj = Objective(expr=self.model.X, sense=minimize)

        instance = self.model.create_instance()
        results = self.opt.solve(instance)

        self.assertEqual(ProblemSense.minimize, results.problem.sense)
        self.assertEqual(TerminationCondition.unbounded, results.solver.termination_condition)
        self.assertEqual('Model was proven to be unbounded.', results.solver.termination_message)
        self.assertEqual(SolverStatus.warning, results.solver.status)

    @unittest.skipIf(not cbc_available, "The 'cbc' solver is not available")
    def test_optimal_mip(self):
        self.model.Idx = RangeSet(2)
        self.model.X = Var(self.model.Idx, within=NonNegativeIntegers)
        self.model.Y = Var(self.model.Idx, within=Binary)
        self.model.C1 = Constraint(expr=self.model.X[1] == self.model.X[2] + 1)
        self.model.Obj = Objective(expr=self.model.Y[1] + self.model.Y[2] - self.model.X[1],
                                   sense=maximize)

        results = self.opt.solve(self.model)

        self.assertEqual(1.0, results.problem.lower_bound)
        self.assertEqual(1.0, results.problem.upper_bound)
        self.assertEqual(results.problem.number_of_binary_variables, 2)
        self.assertEqual(results.problem.number_of_integer_variables, 4)
        self.assertEqual(ProblemSense.maximize, results.problem.sense)
        self.assertEqual(TerminationCondition.optimal, results.solver.termination_condition)
        self.assertEqual(
            'Model was solved to optimality (subject to tolerances), and an optimal solution is available.',
            results.solver.termination_message)
        self.assertEqual(SolverStatus.ok, results.solver.status)


class TestCBCUsingMock(unittest.TestCase):
    """
    These tests cover various abnormal exit conditions from CBC (notably various time or solution limits).
    In order to be able to reliably compare solutions on different platforms, we will compare against cached solutions.
    Effectively we are just testing how we parse the output files.
    The files were produced by using the following configuration:
    macOS Mojave Version 10.14.3
    CBC Version 2.9.9
    appdirs==1.4.3
    certifi==2018.11.29
    nose==1.3.7
    numpy==1.16.1
    ply==3.11
    Pyomo==5.6.1
    PyUtilib==5.6.5
    six==1.12.0


    n = 19
    np.random.seed(42)
    distance_matrix = np.random.rand(n, n)
    model = ConcreteModel()
    model.N = RangeSet(n)
    model.c = Param(model.N, model.N, initialize=lambda a_model, i, j: distance_matrix[i - 1][j - 1])
    model.x = Var(model.N, model.N, within=Binary)
    model.u = Var(model.N, within=NonNegativeReals)

    # Remove arcs that go to and from same node
    for n in model.N:
        model.x[n, n].fix(0)

    def obj_rule(a_model):
        return sum(a_model.c[i, j] * a_model.x[i, j] for i in a_model.N for j in a_model.N)

    model.obj = Objective(rule=obj_rule, sense=minimize)

    def only_leave_each_node_once_constraints(a_model, i):
        return sum(a_model.x[i, j] for j in a_model.N) == 1

    def only_arrive_at_each_node_once_constraints(a_model, j):
        return sum(a_model.x[i, j] for i in a_model.N) == 1

    def miller_tucker_zemlin_constraints(a_model, i, j):
        if i != j and i >= 2 and j >= 2:
            return a_model.u[i] - a_model.u[j] + a_model.x[i, j] * n <= n - 1
        return Constraint.NoConstraint

    model.con1 = Constraint(model.N, rule=only_leave_each_node_once_constraints)
    model.con2 = Constraint(model.N, rule=only_arrive_at_each_node_once_constraints)
    model.con3 = Constraint(model.N, model.N, rule=miller_tucker_zemlin_constraints)

    opt = SolverFactory('cbc', executable=_get_path_for_solver(), options={'randomCbcSeed': 42, 'randomSeed': 42})
    results = opt.solve(self.model, tee=True, **solver_kwargs)

    """

    def setUp(self):
        self.stderr = sys.stderr
        sys.stderr = None
        self.opt = SolverFactory("_mock_cbc", solver_io="lp")

    def tearDown(self):
        sys.stderr = self.stderr

    def test_optimal_mip(self):
        """
        solver_kwargs={}
        """
        lp_file = 'optimal.out.lp'
        results = self.opt.solve(os.path.join(data_dir, lp_file))

        self.assertEqual(1.20645976, results.problem.lower_bound)
        self.assertEqual(1.20645976, results.problem.upper_bound)
        self.assertEqual(SolverStatus.ok, results.solver.status)
        self.assertEqual(0.34, results.solver.system_time)
        self.assertEqual(0.72, results.solver.wallclock_time)
        self.assertEqual(TerminationCondition.optimal, results.solver.termination_condition)
        self.assertEqual(
            'Model was solved to optimality (subject to tolerances), and an optimal solution is available.',
            results.solver.termination_message)
        self.assertEqual(results.solver.statistics.branch_and_bound.number_of_bounded_subproblems, 2)
        self.assertEqual(results.solver.statistics.branch_and_bound.number_of_created_subproblems, 2)
        self.assertEqual(results.solver.statistics.black_box.number_of_iterations, 625)

    def test_max_time_limit_mip(self):
        """
        solver_kwargs={'timelimit': 0.1}
        """
        lp_file = 'max_time_limit.out.lp'
        results = self.opt.solve(os.path.join(data_dir, lp_file))

        self.assertEqual(1.1084706, results.problem.lower_bound)  # Note that we ignore the lower bound given at the end
        self.assertEqual(1.35481947, results.problem.upper_bound)
        self.assertEqual(SolverStatus.aborted, results.solver.status)
        self.assertEqual(0.1, results.solver.system_time)
        self.assertEqual(0.11, results.solver.wallclock_time)
        self.assertEqual(TerminationCondition.maxTimeLimit, results.solver.termination_condition)
        self.assertEqual(
            'Optimization terminated because the time expended exceeded the value specified in the seconds parameter.',
            results.solver.termination_message)
        self.assertEqual(results.solver.statistics.branch_and_bound.number_of_bounded_subproblems, 0)
        self.assertEqual(results.solver.statistics.branch_and_bound.number_of_created_subproblems, 0)
        self.assertEqual(results.solver.statistics.black_box.number_of_iterations, 82)

    def test_intermediate_non_integer_mip(self):
        """
        solver_kwargs={'timelimit': 0.0001}
        """
        lp_file = 'intermediate_non_integer.out.lp'
        results = self.opt.solve(os.path.join(data_dir, lp_file))

        self.assertEqual(0.92543678, results.problem.lower_bound)
        self.assertEqual(SolverStatus.aborted, results.solver.status)
        self.assertEqual(0.02, results.solver.system_time)
        self.assertEqual(0.02, results.solver.wallclock_time)
        self.assertEqual(TerminationCondition.intermediateNonInteger, results.solver.termination_condition)
        self.assertEqual(
            'Optimization terminated because a limit was hit, however it had not found an integer solution yet.',
            results.solver.termination_message)
        self.assertEqual(results.solver.statistics.branch_and_bound.number_of_bounded_subproblems, 0)
        self.assertEqual(results.solver.statistics.branch_and_bound.number_of_created_subproblems, 0)
        self.assertEqual(results.solver.statistics.black_box.number_of_iterations, 0)

    def test_max_solutions(self):
        """
        solver_kwargs={'options': {'maxSolutions': 1}}
        """
        lp_file = 'max_solutions.out.lp'
        results = self.opt.solve(os.path.join(data_dir, lp_file))

        self.assertEqual(0.92543678, results.problem.lower_bound)
        self.assertEqual(1.35481947, results.problem.upper_bound)
        self.assertEqual(SolverStatus.aborted, results.solver.status)
        self.assertEqual(0.03, results.solver.system_time)
        self.assertEqual(0.03, results.solver.wallclock_time)
        self.assertEqual(TerminationCondition.other, results.solver.termination_condition)
        self.assertEqual(
            'Optimization terminated because the number of solutions found reached the value specified in the '
            'maxSolutions parameter.', results.solver.termination_message)
        self.assertEqual(results.solver.statistics.branch_and_bound.number_of_bounded_subproblems, 0)
        self.assertEqual(results.solver.statistics.branch_and_bound.number_of_created_subproblems, 0)
        self.assertEqual(results.solver.statistics.black_box.number_of_iterations, 0)

    def test_within_gap_tolerance(self):
        """
        solver_kwargs={'options': {'allowableGap': 1000000}}
        """
        lp_file = 'within_gap_tolerance.out.lp'
        results = self.opt.solve(os.path.join(data_dir, lp_file))

        self.assertEqual(0.925437, results.problem.lower_bound)
        self.assertEqual(1.35481947, results.problem.upper_bound)
        self.assertEqual(SolverStatus.ok, results.solver.status)
        self.assertEqual(0.07, results.solver.system_time)
        self.assertEqual(0.07, results.solver.wallclock_time)
        self.assertEqual(TerminationCondition.optimal, results.solver.termination_condition)
        self.assertEqual(
            'Model was solved to optimality (subject to tolerances), and an optimal solution is available.',
            results.solver.termination_message)
        self.assertEqual(results.solver.statistics.branch_and_bound.number_of_bounded_subproblems, 0)
        self.assertEqual(results.solver.statistics.branch_and_bound.number_of_created_subproblems, 0)
        self.assertEqual(results.solver.statistics.black_box.number_of_iterations, 0)

    def test_max_evaluations(self):
        """
        solver_kwargs={'options': {'maxNodes': 1}}
        """
        lp_file = 'max_evaluations.out.lp'
        results = self.opt.solve(os.path.join(data_dir, lp_file))

        self.assertEqual(1.2052223, results.problem.lower_bound)
        self.assertEqual(1.20645976, results.problem.upper_bound)
        self.assertEqual(SolverStatus.aborted, results.solver.status)
        self.assertEqual(0.16, results.solver.system_time)
        self.assertEqual(0.18, results.solver.wallclock_time)
        self.assertEqual(TerminationCondition.maxEvaluations, results.solver.termination_condition)
        self.assertEqual(
            'Optimization terminated because the total number of branch-and-cut nodes explored exceeded the value '
            'specified in the maxNodes parameter', results.solver.termination_message)
        self.assertEqual(results.solver.statistics.branch_and_bound.number_of_bounded_subproblems, 1)
        self.assertEqual(results.solver.statistics.branch_and_bound.number_of_created_subproblems, 1)
        self.assertEqual(results.solver.statistics.black_box.number_of_iterations, 602)

    def test_fix_parsing_bug(self):
        """
        The test wasn't generated using the method in the class docstring
        See https://github.com/Pyomo/pyomo/issues/1001
        """
        lp_file = 'fix_parsing_bug.out.lp'
        results = self.opt.solve(os.path.join(data_dir, lp_file))

        self.assertEqual(3.0, results.problem.lower_bound)
        self.assertEqual(3.0, results.problem.upper_bound)
        self.assertEqual(SolverStatus.aborted, results.solver.status)
        self.assertEqual(0.08, results.solver.system_time)
        self.assertEqual(0.09, results.solver.wallclock_time)
        self.assertEqual(TerminationCondition.other, results.solver.termination_condition)
        self.assertEqual(
            'Optimization terminated because the number of solutions found reached the value specified in the '
            'maxSolutions parameter.', results.solver.termination_message)
        self.assertEqual(results.solver.statistics.branch_and_bound.number_of_bounded_subproblems, 0)
        self.assertEqual(results.solver.statistics.branch_and_bound.number_of_created_subproblems, 0)
        self.assertEqual(results.solver.statistics.black_box.number_of_iterations, 0)


if __name__ == "__main__":
    unittest.main()
