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

import sys
from itertools import product
from random import random, seed

import pyomo.common.unittest as unittest

from pyomo.environ import (ConcreteModel, AbstractModel, Var, Objective,
                           Block, Constraint, Suffix, NonNegativeIntegers,
                           NonNegativeReals, Integers, Binary, is_fixed,
                           value)
from pyomo.opt import SolverFactory, TerminationCondition, SolutionStatus
from pyomo.solvers.plugins.solvers.cplex_direct import (_CplexExpr,
                                                        _LinearConstraintData,
                                                        _VariableData)

from pyomo.solvers.plugins.solvers.cplex_direct import (_CplexExpr,
                                                        _LinearConstraintData,
                                                        _VariableData)

try:
    import cplex
    cplexpy_available = True
except ImportError:
    cplexpy_available = False

diff_tol = 1e-4

class CPLEXDirectTests(unittest.TestCase):

    @staticmethod
    def build_mtz_tsp_model(nodes, links, distances):
        # Taken from examples/pyomo/callbacks/tsp.py
        model = ConcreteModel()

        model.POINTS = Set(initialize=nodes, ordered=True)
        model.POINTS_LESS_FIRST = Set(initialize=nodes[1:], ordered=True)
        model.LINKS = Set(initialize=links, ordered=True)
        model.LINKS_LESS_FIRST = Set(
            initialize=[
                (i, j) for (i, j) in links if i in nodes[1:] and j in nodes[1:]
            ],
            ordered=True,
        )

        model.N = len(nodes)
        model.d = Param(model.LINKS, initialize=distances)

        model.Z = Var(model.LINKS, domain=Binary)
        model.FLOW = Var(
            model.POINTS_LESS_FIRST,
            domain=NonNegativeReals,
            bounds=(0, model.N - 1),
        )

        model.InDegrees = Constraint(
            model.POINTS,
            rule=lambda m, i: sum(
                model.Z[i, j] for (i_, j) in model.LINKS if i == i_
            )
                              == 1,
        )
        model.OutDegrees = Constraint(
            model.POINTS,
            rule=lambda m, i: sum(
                model.Z[j, i] for (j, i_) in model.LINKS if i == i_
            )
                              == 1,
        )

        model.FlowCon = Constraint(
            model.LINKS_LESS_FIRST,
            rule=lambda m, i, j: model.FLOW[i] - model.FLOW[j] + m.N * m.Z[i, j]
                                 <= m.N - 1,
        )

        model.tour_length = Objective(
            expr=sum_product(model.d, model.Z), sense=minimize
        )
        return model

    def setUp(self):
        self.stderr = sys.stderr
        sys.stderr = None

    def tearDown(self):
        sys.stderr = self.stderr

    @unittest.skipIf(not cplexpy_available,
                     "The 'cplex' python bindings are not available")
    def test_infeasible_lp(self):
        with SolverFactory("cplex", solver_io="python") as opt:

            model = ConcreteModel()
            model.X = Var(within=NonNegativeReals)
            model.C1 = Constraint(expr= model.X==1)
            model.C2 = Constraint(expr= model.X==2)
            model.O = Objective(expr= model.X)

            results = opt.solve(model)

            self.assertEqual(results.solver.termination_condition,
                             TerminationCondition.infeasible)

    @unittest.skipIf(not cplexpy_available,
                     "The 'cplex' python bindings are not available")
    def test_unbounded_lp(self):
        with SolverFactory("cplex", solver_io="python") as opt:

            model = ConcreteModel()
            model.X = Var()
            model.O = Objective(expr= model.X)

            results = opt.solve(model)

            self.assertIn(results.solver.termination_condition,
                          (TerminationCondition.unbounded,
                           TerminationCondition.infeasibleOrUnbounded))

    @unittest.skipIf(not cplexpy_available,
                     "The 'cplex' python bindings are not available")
    def test_optimal_lp(self):
        with SolverFactory("cplex", solver_io="python") as opt:

            model = ConcreteModel()
            model.X = Var(within=NonNegativeReals)
            model.O = Objective(expr= model.X)

            results = opt.solve(model, load_solutions=False)

            self.assertEqual(results.solution.status,
                             SolutionStatus.optimal)

    @unittest.skipIf(not cplexpy_available,
                     "The 'cplex' python bindings are not available")
    def test_get_duals_lp(self):
        with SolverFactory("cplex", solver_io="python") as opt:

            model = ConcreteModel()
            model.X = Var(within=NonNegativeReals)
            model.Y = Var(within=NonNegativeReals)

            model.C1 = Constraint(expr= 2*model.X + model.Y >= 8 )
            model.C2 = Constraint(expr= model.X + 3*model.Y >= 6 )

            model.O = Objective(expr= model.X + model.Y)

            results = opt.solve(model, suffixes=['dual'], load_solutions=False)

            model.dual = Suffix(direction=Suffix.IMPORT)
            model.solutions.load_from(results)

            self.assertAlmostEqual(model.dual[model.C1], 0.4)
            self.assertAlmostEqual(model.dual[model.C2], 0.2)

    @unittest.skipIf(not cplexpy_available,
                     "The 'cplex' python bindings are not available")
    def test_infeasible_mip(self):
        with SolverFactory("cplex", solver_io="python") as opt:

            model = ConcreteModel()
            model.X = Var(within=NonNegativeIntegers)
            model.C1 = Constraint(expr= model.X==1)
            model.C2 = Constraint(expr= model.X==2)
            model.O = Objective(expr= model.X)

            results = opt.solve(model)

            self.assertEqual(results.solver.termination_condition,
                             TerminationCondition.infeasible)

    @unittest.skipIf(not cplexpy_available,
                     "The 'cplex' python bindings are not available")
    def test_no_solution_mip(self):
        with SolverFactory("cplex", solver_io="python") as opt:
            # Set the `options` such that CPLEX cannot determine the problem as infeasible within the time allowed
            opt.options["dettimelimit"] = 1
            opt.options["lpmethod"] = 1
            opt.options["threads"] = 1

            opt.options["mip_limits_nodes"] = 0
            opt.options["mip_limits_eachcutlimit"] = 0
            opt.options["mip_limits_cutsfactor"] = 0
            opt.options["mip_limits_auxrootthreads"] = -1

            opt.options["preprocessing_presolve"] = 0
            opt.options["preprocessing_reduce"] = 0
            opt.options["preprocessing_relax"] = 0

            opt.options["mip_strategy_heuristicfreq"] = -1
            opt.options["mip_strategy_presolvenode"] = -1
            opt.options["mip_strategy_probe"] = -1

            opt.options["mip_cuts_mircut"] = -1
            opt.options["mip_cuts_implied"] = -1
            opt.options["mip_cuts_gomory"] = -1
            opt.options["mip_cuts_flowcovers"] = -1
            opt.options["mip_cuts_pathcut"] = -1
            opt.options["mip_cuts_liftproj"] = -1
            opt.options["mip_cuts_zerohalfcut"] = -1
            opt.options["mip_cuts_cliques"] = -1
            opt.options["mip_cuts_covers"] = -1

            nodes = list(range(15))
            links = list((i, j) for i, j in product(nodes, nodes) if i != j)

            seed(0)
            distances = {link: random() for link in links}

            model = self.build_mtz_tsp_model(nodes, links, distances)

            results = opt.solve(model)

            self.assertEqual(results.solver.status, SolverStatus.warning)
            self.assertEqual(
                results.solver.termination_condition, TerminationCondition.noSolution
            )

    @unittest.skipIf(not cplexpy_available,
                     "The 'cplex' python bindings are not available")
    def test_unbounded_mip(self):
        with SolverFactory("cplex", solver_io="python") as opt:

            model = AbstractModel()
            model.X = Var(within=Integers)
            model.O = Objective(expr= model.X)

            instance = model.create_instance()
            results = opt.solve(instance)

            self.assertIn(results.solver.termination_condition,
                          (TerminationCondition.unbounded,
                           TerminationCondition.infeasibleOrUnbounded))

    @unittest.skipIf(not cplexpy_available,
                     "The 'cplex' python bindings are not available")
    def test_optimal_mip(self):
        with SolverFactory("cplex", solver_io="python") as opt:

            model = ConcreteModel()
            model.X = Var(within=NonNegativeIntegers)
            model.O = Objective(expr= model.X)

            results = opt.solve(model, load_solutions=False)

            self.assertEqual(results.solution.status,
                             SolutionStatus.optimal)


@unittest.skipIf(not cplexpy_available, "The 'cplex' python bindings are not available")
class TestIsFixedCallCount(unittest.TestCase):
    """ Tests for PR#1402 (669e7b2b) """

    def setup(self, skip_trivial_constraints):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.c1 = Constraint(expr=m.x + m.y == 1)
        m.c2 = Constraint(expr=m.x <= 1)
        self.assertFalse(m.c2.has_lb())
        self.assertTrue(m.c2.has_ub())
        self._model = m

        self._opt = SolverFactory("cplex_persistent")
        self._opt.set_instance(
            self._model, skip_trivial_constraints=skip_trivial_constraints
        )

    def test_skip_trivial_and_call_count_for_fixed_con_is_one(self):
        self.setup(skip_trivial_constraints=True)
        self._model.x.fix(1)
        self.assertTrue(self._opt._skip_trivial_constraints)
        self.assertTrue(self._model.c2.body.is_fixed())

        with unittest.mock.patch(
            "pyomo.solvers.plugins.solvers.cplex_direct.is_fixed", wraps=is_fixed
        ) as mock_is_fixed:
            self.assertEqual(mock_is_fixed.call_count, 0)
            self._opt.add_constraint(self._model.c2)
            self.assertEqual(mock_is_fixed.call_count, 1)

    def test_skip_trivial_and_call_count_for_unfixed_con_is_two(self):
        self.setup(skip_trivial_constraints=True)
        self.assertTrue(self._opt._skip_trivial_constraints)
        self.assertFalse(self._model.c2.body.is_fixed())

        with unittest.mock.patch(
            "pyomo.solvers.plugins.solvers.cplex_direct.is_fixed", wraps=is_fixed
        ) as mock_is_fixed:
            self.assertEqual(mock_is_fixed.call_count, 0)
            self._opt.add_constraint(self._model.c2)
            self.assertEqual(mock_is_fixed.call_count, 2)

    def test_skip_trivial_and_call_count_for_unfixed_equality_con_is_three(self):
        self.setup(skip_trivial_constraints=True)
        self._model.c2 = Constraint(expr=self._model.x == 1)
        self.assertTrue(self._opt._skip_trivial_constraints)
        self.assertFalse(self._model.c2.body.is_fixed())

        with unittest.mock.patch(
            "pyomo.solvers.plugins.solvers.cplex_direct.is_fixed", wraps=is_fixed
        ) as mock_is_fixed:
            self.assertEqual(mock_is_fixed.call_count, 0)
            self._opt.add_constraint(self._model.c2)
            self.assertEqual(mock_is_fixed.call_count, 3)

    def test_dont_skip_trivial_and_call_count_for_fixed_con_is_one(self):
        self.setup(skip_trivial_constraints=False)
        self._model.x.fix(1)
        self.assertFalse(self._opt._skip_trivial_constraints)
        self.assertTrue(self._model.c2.body.is_fixed())

        with unittest.mock.patch(
            "pyomo.solvers.plugins.solvers.cplex_direct.is_fixed", wraps=is_fixed
        ) as mock_is_fixed:
            self.assertEqual(mock_is_fixed.call_count, 0)
            self._opt.add_constraint(self._model.c2)
            self.assertEqual(mock_is_fixed.call_count, 1)

    def test_dont_skip_trivial_and_call_count_for_unfixed_con_is_one(self):
        self.setup(skip_trivial_constraints=False)
        self.assertFalse(self._opt._skip_trivial_constraints)
        self.assertFalse(self._model.c2.body.is_fixed())

        with unittest.mock.patch(
            "pyomo.solvers.plugins.solvers.cplex_direct.is_fixed", wraps=is_fixed
        ) as mock_is_fixed:
            self.assertEqual(mock_is_fixed.call_count, 0)
            self._opt.add_constraint(self._model.c2)
            self.assertEqual(mock_is_fixed.call_count, 1)


@unittest.skipIf(not cplexpy_available, "The 'cplex' python bindings are not available")
class TestDataContainers(unittest.TestCase):
    def test_variable_data(self):
        solver_model = cplex.Cplex()
        var_data = _VariableData(solver_model)
        var_data.add(lb=0, ub=1, type_=solver_model.variables.type.binary, name="var1")
        var_data.add(
            lb=0, ub=10, type_=solver_model.variables.type.integer, name="var2"
        )
        var_data.add(
            lb=-cplex.infinity,
            ub=cplex.infinity,
            type_=solver_model.variables.type.continuous,
            name="var3",
        )
        self.assertEqual(solver_model.variables.get_num(), 0)
        var_data.store_in_cplex()
        self.assertEqual(solver_model.variables.get_num(), 3)

    def test_constraint_data(self):
        solver_model = cplex.Cplex()

        solver_model.variables.add(
            lb=[-cplex.infinity, -cplex.infinity, -cplex.infinity],
            ub=[cplex.infinity, cplex.infinity, cplex.infinity],
            types=[
                solver_model.variables.type.continuous,
                solver_model.variables.type.continuous,
                solver_model.variables.type.continuous,
            ],
            names=["var1", "var2", "var3"],
        )
        con_data = _LinearConstraintData(solver_model)
        con_data.add(
            cplex_expr=_CplexExpr(variables=[0, 1], coefficients=[10, 100]),
            sense="L",
            rhs=0,
            range_values=0,
            name="c1",
        )
        con_data.add(
            cplex_expr=_CplexExpr(variables=[0], coefficients=[-30]),
            sense="G",
            rhs=1,
            range_values=0,
            name="c2",
        )
        con_data.add(
            cplex_expr=_CplexExpr(variables=[1], coefficients=[80]),
            sense="E",
            rhs=2,
            range_values=0,
            name="c3",
        )
        con_data.add(
            cplex_expr=_CplexExpr(variables=[2], coefficients=[50]),
            sense="R",
            rhs=3,
            range_values=10,
            name="c4",
        )

        self.assertEqual(solver_model.linear_constraints.get_num(), 0)
        con_data.store_in_cplex()
        self.assertEqual(solver_model.linear_constraints.get_num(), 4)


@unittest.skipIf(not cplexpy_available, "The 'cplex' python bindings are not available")
class TestAddVar(unittest.TestCase):
    def test_add_single_variable(self):
        """ Test that the variable is added correctly to `solver_model`. """
        model = ConcreteModel()

        opt = SolverFactory("cplex", solver_io="python")
        opt._set_instance(model)

        self.assertEqual(opt._solver_model.variables.get_num(), 0)
        self.assertEqual(opt._solver_model.variables.get_num_binary(), 0)

        model.X = Var(within=Binary)

        var_interface = opt._solver_model.variables
        with unittest.mock.patch.object(
            var_interface, "add", wraps=var_interface.add
        ) as wrapped_add_call, unittest.mock.patch.object(
            var_interface, "set_lower_bounds", wraps=var_interface.set_lower_bounds
        ) as wrapped_lb_call, unittest.mock.patch.object(
            var_interface, "set_upper_bounds", wraps=var_interface.set_upper_bounds
        ) as wrapped_ub_call:
            opt._add_var(model.X)

            self.assertEqual(wrapped_add_call.call_count, 1)
            self.assertEqual(
                wrapped_add_call.call_args,
                ({"lb": [0], "names": ["x1"], "types": ["B"], "ub": [1]},),
            )

            self.assertFalse(wrapped_lb_call.called)
            self.assertFalse(wrapped_ub_call.called)

        self.assertEqual(opt._solver_model.variables.get_num(), 1)
        self.assertEqual(opt._solver_model.variables.get_num_binary(), 1)

    def test_add_block_containing_single_variable(self):
        """ Test that the variable is added correctly to `solver_model`. """
        model = ConcreteModel()

        opt = SolverFactory("cplex", solver_io="python")
        opt._set_instance(model)

        self.assertEqual(opt._solver_model.variables.get_num(), 0)
        self.assertEqual(opt._solver_model.variables.get_num_binary(), 0)

        model.X = Var(within=Binary)

        with unittest.mock.patch.object(
            opt._solver_model.variables, "add", wraps=opt._solver_model.variables.add
        ) as wrapped_add_call:
            opt._add_block(model)

            self.assertEqual(wrapped_add_call.call_count, 1)
            self.assertEqual(
                wrapped_add_call.call_args,
                ({"lb": [0], "names": ["x1"], "types": ["B"], "ub": [1]},),
            )

        self.assertEqual(opt._solver_model.variables.get_num(), 1)
        self.assertEqual(opt._solver_model.variables.get_num_binary(), 1)

    def test_add_block_containing_multiple_variables(self):
        """ Test that:
            - The variable is added correctly to `solver_model`
            - The CPLEX `variables` interface is called only once
            - Fixed variable bounds are set correctly
        """
        model = ConcreteModel()

        opt = SolverFactory("cplex", solver_io="python")
        opt._set_instance(model)

        self.assertEqual(opt._solver_model.variables.get_num(), 0)

        model.X1 = Var(within=Binary)
        model.X2 = Var(within=NonNegativeReals)
        model.X3 = Var(within=NonNegativeIntegers)

        model.X3.fix(5)

        with unittest.mock.patch.object(
            opt._solver_model.variables, "add", wraps=opt._solver_model.variables.add
        ) as wrapped_add_call:
            opt._add_block(model)

            self.assertEqual(wrapped_add_call.call_count, 1)
            self.assertEqual(
                wrapped_add_call.call_args,
                (
                    {
                        "lb": [0, 0, 5],
                        "names": ["x1", "x2", "x3"],
                        "types": ["B", "C", "I"],
                        "ub": [1, cplex.infinity, 5],
                    },
                ),
            )

        self.assertEqual(opt._solver_model.variables.get_num(), 3)


@unittest.skipIf(not cplexpy_available, "The 'cplex' python bindings are not available")
class TestAddCon(unittest.TestCase):
    def test_add_single_constraint(self):
        model = ConcreteModel()
        model.X = Var(within=Binary)

        opt = SolverFactory("cplex", solver_io="python")
        opt._set_instance(model)

        self.assertEqual(opt._solver_model.linear_constraints.get_num(), 0)

        model.C = Constraint(expr=model.X == 1)

        con_interface = opt._solver_model.linear_constraints
        with unittest.mock.patch.object(
            con_interface, "add", wraps=con_interface.add
        ) as wrapped_add_call:
            opt._add_constraint(model.C)

            self.assertEqual(wrapped_add_call.call_count, 1)
            self.assertEqual(
                wrapped_add_call.call_args,
                (
                    {
                        "lin_expr": [[[0], (1,)]],
                        "names": ["x2"],
                        "range_values": [0.0],
                        "rhs": [1.0],
                        "senses": ["E"],
                    },
                ),
            )

        self.assertEqual(opt._solver_model.linear_constraints.get_num(), 1)

    def test_add_block_containing_single_constraint(self):
        model = ConcreteModel()
        model.X = Var(within=Binary)

        opt = SolverFactory("cplex", solver_io="python")
        opt._set_instance(model)

        self.assertEqual(opt._solver_model.linear_constraints.get_num(), 0)

        model.B = Block()
        model.B.C = Constraint(expr=model.X == 1)

        con_interface = opt._solver_model.linear_constraints
        with unittest.mock.patch.object(
            con_interface, "add", wraps=con_interface.add
        ) as wrapped_add_call:
            opt._add_block(model.B)

            self.assertEqual(wrapped_add_call.call_count, 1)
            self.assertEqual(
                wrapped_add_call.call_args,
                (
                    {
                        "lin_expr": [[[0], (1,)]],
                        "names": ["x2"],
                        "range_values": [0.0],
                        "rhs": [1.0],
                        "senses": ["E"],
                    },
                ),
            )

        self.assertEqual(opt._solver_model.linear_constraints.get_num(), 1)

    def test_add_block_containing_multiple_constraints(self):
        model = ConcreteModel()
        model.X = Var(within=Binary)

        opt = SolverFactory("cplex", solver_io="python")
        opt._set_instance(model)

        self.assertEqual(opt._solver_model.linear_constraints.get_num(), 0)

        model.B = Block()
        model.B.C1 = Constraint(expr=model.X == 1)
        model.B.C2 = Constraint(expr=model.X <= 1)
        model.B.C3 = Constraint(expr=model.X >= 1)

        con_interface = opt._solver_model.linear_constraints
        with unittest.mock.patch.object(
            con_interface, "add", wraps=con_interface.add
        ) as wrapped_add_call:
            opt._add_block(model.B)

            self.assertEqual(wrapped_add_call.call_count, 1)
            self.assertEqual(
                wrapped_add_call.call_args,
                (
                    {
                        "lin_expr": [[[0], (1,)], [[0], (1,)], [[0], (1,)]],
                        "names": ["x2", "x3", "x4"],
                        "range_values": [0.0, 0.0, 0.0],
                        "rhs": [1.0, 1.0, 1.0],
                        "senses": ["E", "L", "G"],
                    },
                ),
            )

        self.assertEqual(opt._solver_model.linear_constraints.get_num(), 3)


@unittest.skipIf(not cplexpy_available, "The 'cplex' python bindings are not available")
class TestLoadVars(unittest.TestCase):
    def setUp(self):
        opt = SolverFactory("cplex", solver_io="python")
        model = ConcreteModel()
        model.X = Var(within=NonNegativeReals, initialize=0)
        model.Y = Var(within=NonNegativeReals, initialize=0)

        model.C1 = Constraint(expr=2 * model.X + model.Y >= 8)
        model.C2 = Constraint(expr=model.X + 3 * model.Y >= 6)

        model.O = Objective(expr=model.X + model.Y)

        opt.solve(model, load_solutions=False, save_results=False)

        self._model = model
        self._opt = opt

    def test_all_vars_are_loaded(self):
        self.assertTrue(self._model.X.stale)
        self.assertTrue(self._model.Y.stale)
        self.assertEqual(value(self._model.X), 0)
        self.assertEqual(value(self._model.Y), 0)

        with unittest.mock.patch.object(
            self._opt._solver_model.solution,
            "get_values",
            wraps=self._opt._solver_model.solution.get_values,
        ) as wrapped_values_call:
            self._opt.load_vars()

            self.assertEqual(wrapped_values_call.call_count, 1)
            self.assertEqual(wrapped_values_call.call_args, tuple())

        self.assertFalse(self._model.X.stale)
        self.assertFalse(self._model.Y.stale)
        self.assertAlmostEqual(value(self._model.X), 3.6)
        self.assertAlmostEqual(value(self._model.Y), 0.8)

    def test_only_specified_vars_are_loaded(self):
        self.assertTrue(self._model.X.stale)
        self.assertTrue(self._model.Y.stale)
        self.assertEqual(value(self._model.X), 0)
        self.assertEqual(value(self._model.Y), 0)

        with unittest.mock.patch.object(
            self._opt._solver_model.solution,
            "get_values",
            wraps=self._opt._solver_model.solution.get_values,
        ) as wrapped_values_call:
            self._opt.load_vars([self._model.X])

            self.assertEqual(wrapped_values_call.call_count, 1)
            self.assertEqual(wrapped_values_call.call_args, (([0],), {}))

        self.assertFalse(self._model.X.stale)
        self.assertTrue(self._model.Y.stale)
        self.assertAlmostEqual(value(self._model.X), 3.6)
        self.assertEqual(value(self._model.Y), 0)

        with unittest.mock.patch.object(
            self._opt._solver_model.solution,
            "get_values",
            wraps=self._opt._solver_model.solution.get_values,
        ) as wrapped_values_call:
            self._opt.load_vars([self._model.Y])

            self.assertEqual(wrapped_values_call.call_count, 1)
            self.assertEqual(wrapped_values_call.call_args, (([1],), {}))

        self.assertFalse(self._model.X.stale)
        self.assertFalse(self._model.Y.stale)
        self.assertAlmostEqual(value(self._model.X), 3.6)
        self.assertAlmostEqual(value(self._model.Y), 0.8)

    @unittest.skipIf(not cplexpy_available,
                     "The 'cplex' python bindings are not available")
    def test_soln_limit_mip(self):
        with SolverFactory("cplex", solver_io="python") as opt:
            model = ConcreteModel()
            model.X = Var(within=Binary)
            model.C1 = Constraint(expr=model.X == 1)
            model.O = Objective(expr=model.X)

            opt.options['mip_limits_solutions'] = 1
            results = opt.solve(model)

            self.assertEqual(results.solver.status,
                             SolverStatus.aborted)
            self.assertEqual(results.solver.termination_condition,
                             TerminationCondition.unknown)
            self.assertEqual(model.solutions[0].status,
                             SolutionStatus.stoppedByLimit)

    @unittest.skipIf(not cplexpy_available,
                     "The 'cplex' python bindings are not available")
    def test_dettime_limit_mip(self):
        with SolverFactory("cplex", solver_io="python") as opt:
            nodes = list(range(20))
            links = list((i, j) for i, j in product(nodes, nodes) if i != j)
            seed(0)
            distances = {link: random() for link in links}
            model = self.build_mtz_tsp_model(nodes, links, distances)

            opt.options["dettimelimit"] = 10

            results = opt.solve(model)

            self.assertEqual(results.solver.status,
                             SolverStatus.aborted)
            self.assertEqual(results.solver.termination_condition,
                             TerminationCondition.maxTimeLimit)
            self.assertEqual(results.solver.termination_message, 'deterministic time limit exceeded')
            self.assertEqual(model.solutions[0].status,
                             SolutionStatus.stoppedByLimit)
            self.assertTrue(9 <= results.solver.deterministic_time <= 11)


@unittest.skipIf(not unittest.mock_available, "'mock' is not available")
@unittest.skipIf(not cplexpy_available, "The 'cplex' python bindings are not available")
class TestIsFixedCallCount(unittest.TestCase):
    """ Tests for PR#1402 (669e7b2b) """

    def setup(self, skip_trivial_constraints):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.c1 = Constraint(expr=m.x + m.y == 1)
        m.c2 = Constraint(expr=m.x <= 1)
        self.assertFalse(m.c2.has_lb())
        self.assertTrue(m.c2.has_ub())
        self._model = m

        self._opt = SolverFactory("cplex_persistent")
        self._opt.set_instance(
            self._model, skip_trivial_constraints=skip_trivial_constraints
        )

    def test_skip_trivial_and_call_count_for_fixed_con_is_one(self):
        self.setup(skip_trivial_constraints=True)
        self._model.x.fix(1)
        self.assertTrue(self._opt._skip_trivial_constraints)
        self.assertTrue(self._model.c2.body.is_fixed())

        with unittest.mock.patch(
            "pyomo.solvers.plugins.solvers.cplex_direct.is_fixed", wraps=is_fixed
        ) as mock_is_fixed:
            self.assertEqual(mock_is_fixed.call_count, 0)
            self._opt.add_constraint(self._model.c2)
            self.assertEqual(mock_is_fixed.call_count, 1)

    def test_skip_trivial_and_call_count_for_unfixed_con_is_two(self):
        self.setup(skip_trivial_constraints=True)
        self.assertTrue(self._opt._skip_trivial_constraints)
        self.assertFalse(self._model.c2.body.is_fixed())

        with unittest.mock.patch(
            "pyomo.solvers.plugins.solvers.cplex_direct.is_fixed", wraps=is_fixed
        ) as mock_is_fixed:
            self.assertEqual(mock_is_fixed.call_count, 0)
            self._opt.add_constraint(self._model.c2)
            self.assertEqual(mock_is_fixed.call_count, 2)

    def test_skip_trivial_and_call_count_for_unfixed_equality_con_is_three(self):
        self.setup(skip_trivial_constraints=True)
        self._model.c2 = Constraint(expr=self._model.x == 1)
        self.assertTrue(self._opt._skip_trivial_constraints)
        self.assertFalse(self._model.c2.body.is_fixed())

        with unittest.mock.patch(
            "pyomo.solvers.plugins.solvers.cplex_direct.is_fixed", wraps=is_fixed
        ) as mock_is_fixed:
            self.assertEqual(mock_is_fixed.call_count, 0)
            self._opt.add_constraint(self._model.c2)
            self.assertEqual(mock_is_fixed.call_count, 3)

    def test_dont_skip_trivial_and_call_count_for_fixed_con_is_one(self):
        self.setup(skip_trivial_constraints=False)
        self._model.x.fix(1)
        self.assertFalse(self._opt._skip_trivial_constraints)
        self.assertTrue(self._model.c2.body.is_fixed())

        with unittest.mock.patch(
            "pyomo.solvers.plugins.solvers.cplex_direct.is_fixed", wraps=is_fixed
        ) as mock_is_fixed:
            self.assertEqual(mock_is_fixed.call_count, 0)
            self._opt.add_constraint(self._model.c2)
            self.assertEqual(mock_is_fixed.call_count, 1)

    def test_dont_skip_trivial_and_call_count_for_unfixed_con_is_one(self):
        self.setup(skip_trivial_constraints=False)
        self.assertFalse(self._opt._skip_trivial_constraints)
        self.assertFalse(self._model.c2.body.is_fixed())

        with unittest.mock.patch(
            "pyomo.solvers.plugins.solvers.cplex_direct.is_fixed", wraps=is_fixed
        ) as mock_is_fixed:
            self.assertEqual(mock_is_fixed.call_count, 0)
            self._opt.add_constraint(self._model.c2)
            self.assertEqual(mock_is_fixed.call_count, 1)


@unittest.skipIf(not cplexpy_available, "The 'cplex' python bindings are not available")
class TestDataContainers(unittest.TestCase):
    def test_variable_data(self):
        solver_model = cplex.Cplex()
        var_data = _VariableData(solver_model)
        var_data.add(lb=0, ub=1, type_=solver_model.variables.type.binary, name="var1")
        var_data.add(
            lb=0, ub=10, type_=solver_model.variables.type.integer, name="var2"
        )
        var_data.add(
            lb=-cplex.infinity,
            ub=cplex.infinity,
            type_=solver_model.variables.type.continuous,
            name="var3",
        )
        self.assertEqual(solver_model.variables.get_num(), 0)
        var_data.store_in_cplex()
        self.assertEqual(solver_model.variables.get_num(), 3)

    def test_constraint_data(self):
        solver_model = cplex.Cplex()

        solver_model.variables.add(
            lb=[-cplex.infinity, -cplex.infinity, -cplex.infinity],
            ub=[cplex.infinity, cplex.infinity, cplex.infinity],
            types=[
                solver_model.variables.type.continuous,
                solver_model.variables.type.continuous,
                solver_model.variables.type.continuous,
            ],
            names=["var1", "var2", "var3"],
        )
        con_data = _LinearConstraintData(solver_model)
        con_data.add(
            cplex_expr=_CplexExpr(variables=[0, 1], coefficients=[10, 100]),
            sense="L",
            rhs=0,
            range_values=0,
            name="c1",
        )
        con_data.add(
            cplex_expr=_CplexExpr(variables=[0], coefficients=[-30]),
            sense="G",
            rhs=1,
            range_values=0,
            name="c2",
        )
        con_data.add(
            cplex_expr=_CplexExpr(variables=[1], coefficients=[80]),
            sense="E",
            rhs=2,
            range_values=0,
            name="c3",
        )
        con_data.add(
            cplex_expr=_CplexExpr(variables=[2], coefficients=[50]),
            sense="R",
            rhs=3,
            range_values=10,
            name="c4",
        )

        self.assertEqual(solver_model.linear_constraints.get_num(), 0)
        con_data.store_in_cplex()
        self.assertEqual(solver_model.linear_constraints.get_num(), 4)


@unittest.skipIf(not unittest.mock_available, "'mock' is not available")
@unittest.skipIf(not cplexpy_available, "The 'cplex' python bindings are not available")
class TestAddVar(unittest.TestCase):
    def test_add_single_variable(self):
        """ Test that the variable is added correctly to `solver_model`. """
        model = ConcreteModel()

        opt = SolverFactory("cplex", solver_io="python")
        opt._set_instance(model)

        self.assertEqual(opt._solver_model.variables.get_num(), 0)
        self.assertEqual(opt._solver_model.variables.get_num_binary(), 0)

        model.X = Var(within=Binary)

        var_interface = opt._solver_model.variables
        with unittest.mock.patch.object(
            var_interface, "add", wraps=var_interface.add
        ) as wrapped_add_call, unittest.mock.patch.object(
            var_interface, "set_lower_bounds", wraps=var_interface.set_lower_bounds
        ) as wrapped_lb_call, unittest.mock.patch.object(
            var_interface, "set_upper_bounds", wraps=var_interface.set_upper_bounds
        ) as wrapped_ub_call:
            opt._add_var(model.X)

            self.assertEqual(wrapped_add_call.call_count, 1)
            self.assertEqual(
                wrapped_add_call.call_args,
                ({"lb": [0], "names": ["x1"], "types": ["B"], "ub": [1]},),
            )

            self.assertFalse(wrapped_lb_call.called)
            self.assertFalse(wrapped_ub_call.called)

        self.assertEqual(opt._solver_model.variables.get_num(), 1)
        self.assertEqual(opt._solver_model.variables.get_num_binary(), 1)

    def test_add_block_containing_single_variable(self):
        """ Test that the variable is added correctly to `solver_model`. """
        model = ConcreteModel()

        opt = SolverFactory("cplex", solver_io="python")
        opt._set_instance(model)

        self.assertEqual(opt._solver_model.variables.get_num(), 0)
        self.assertEqual(opt._solver_model.variables.get_num_binary(), 0)

        model.X = Var(within=Binary)

        with unittest.mock.patch.object(
            opt._solver_model.variables, "add", wraps=opt._solver_model.variables.add
        ) as wrapped_add_call:
            opt._add_block(model)

            self.assertEqual(wrapped_add_call.call_count, 1)
            self.assertEqual(
                wrapped_add_call.call_args,
                ({"lb": [0], "names": ["x1"], "types": ["B"], "ub": [1]},),
            )

        self.assertEqual(opt._solver_model.variables.get_num(), 1)
        self.assertEqual(opt._solver_model.variables.get_num_binary(), 1)

    def test_add_block_containing_multiple_variables(self):
        """ Test that:
            - The variable is added correctly to `solver_model`
            - The CPLEX `variables` interface is called only once
            - Fixed variable bounds are set correctly
        """
        model = ConcreteModel()

        opt = SolverFactory("cplex", solver_io="python")
        opt._set_instance(model)

        self.assertEqual(opt._solver_model.variables.get_num(), 0)

        model.X1 = Var(within=Binary)
        model.X2 = Var(within=NonNegativeReals)
        model.X3 = Var(within=NonNegativeIntegers)

        model.X3.fix(5)

        with unittest.mock.patch.object(
            opt._solver_model.variables, "add", wraps=opt._solver_model.variables.add
        ) as wrapped_add_call:
            opt._add_block(model)

            self.assertEqual(wrapped_add_call.call_count, 1)
            self.assertEqual(
                wrapped_add_call.call_args,
                (
                    {
                        "lb": [0, 0, 5],
                        "names": ["x1", "x2", "x3"],
                        "types": ["B", "C", "I"],
                        "ub": [1, cplex.infinity, 5],
                    },
                ),
            )

        self.assertEqual(opt._solver_model.variables.get_num(), 3)


@unittest.skipIf(not unittest.mock_available, "'mock' is not available")
@unittest.skipIf(not cplexpy_available, "The 'cplex' python bindings are not available")
class TestAddCon(unittest.TestCase):
    def test_add_single_constraint(self):
        model = ConcreteModel()
        model.X = Var(within=Binary)

        opt = SolverFactory("cplex", solver_io="python")
        opt._set_instance(model)

        self.assertEqual(opt._solver_model.linear_constraints.get_num(), 0)

        model.C = Constraint(expr=model.X == 1)

        con_interface = opt._solver_model.linear_constraints
        with unittest.mock.patch.object(
            con_interface, "add", wraps=con_interface.add
        ) as wrapped_add_call:
            opt._add_constraint(model.C)

            self.assertEqual(wrapped_add_call.call_count, 1)
            self.assertEqual(
                wrapped_add_call.call_args,
                (
                    {
                        "lin_expr": [[[0], (1,)]],
                        "names": ["x2"],
                        "range_values": [0.0],
                        "rhs": [1.0],
                        "senses": ["E"],
                    },
                ),
            )

        self.assertEqual(opt._solver_model.linear_constraints.get_num(), 1)

    def test_add_block_containing_single_constraint(self):
        model = ConcreteModel()
        model.X = Var(within=Binary)

        opt = SolverFactory("cplex", solver_io="python")
        opt._set_instance(model)

        self.assertEqual(opt._solver_model.linear_constraints.get_num(), 0)

        model.B = Block()
        model.B.C = Constraint(expr=model.X == 1)

        con_interface = opt._solver_model.linear_constraints
        with unittest.mock.patch.object(
            con_interface, "add", wraps=con_interface.add
        ) as wrapped_add_call:
            opt._add_block(model.B)

            self.assertEqual(wrapped_add_call.call_count, 1)
            self.assertEqual(
                wrapped_add_call.call_args,
                (
                    {
                        "lin_expr": [[[0], (1,)]],
                        "names": ["x2"],
                        "range_values": [0.0],
                        "rhs": [1.0],
                        "senses": ["E"],
                    },
                ),
            )

        self.assertEqual(opt._solver_model.linear_constraints.get_num(), 1)

    def test_add_block_containing_multiple_constraints(self):
        model = ConcreteModel()
        model.X = Var(within=Binary)

        opt = SolverFactory("cplex", solver_io="python")
        opt._set_instance(model)

        self.assertEqual(opt._solver_model.linear_constraints.get_num(), 0)

        model.B = Block()
        model.B.C1 = Constraint(expr=model.X == 1)
        model.B.C2 = Constraint(expr=model.X <= 1)
        model.B.C3 = Constraint(expr=model.X >= 1)

        con_interface = opt._solver_model.linear_constraints
        with unittest.mock.patch.object(
            con_interface, "add", wraps=con_interface.add
        ) as wrapped_add_call:
            opt._add_block(model.B)

            self.assertEqual(wrapped_add_call.call_count, 1)
            self.assertEqual(
                wrapped_add_call.call_args,
                (
                    {
                        "lin_expr": [[[0], (1,)], [[0], (1,)], [[0], (1,)]],
                        "names": ["x2", "x3", "x4"],
                        "range_values": [0.0, 0.0, 0.0],
                        "rhs": [1.0, 1.0, 1.0],
                        "senses": ["E", "L", "G"],
                    },
                ),
            )

        self.assertEqual(opt._solver_model.linear_constraints.get_num(), 3)


@unittest.skipIf(not unittest.mock_available, "'mock' is not available")
@unittest.skipIf(not cplexpy_available, "The 'cplex' python bindings are not available")
class TestLoadVars(unittest.TestCase):
    def setUp(self):
        opt = SolverFactory("cplex", solver_io="python")
        model = ConcreteModel()
        model.X = Var(within=NonNegativeReals, initialize=0)
        model.Y = Var(within=NonNegativeReals, initialize=0)

        model.C1 = Constraint(expr=2 * model.X + model.Y >= 8)
        model.C2 = Constraint(expr=model.X + 3 * model.Y >= 6)

        model.O = Objective(expr=model.X + model.Y)

        opt.solve(model, load_solutions=False, save_results=False)

        self._model = model
        self._opt = opt

    def test_all_vars_are_loaded(self):
        self.assertTrue(self._model.X.stale)
        self.assertTrue(self._model.Y.stale)
        self.assertEqual(value(self._model.X), 0)
        self.assertEqual(value(self._model.Y), 0)

        with unittest.mock.patch.object(
            self._opt._solver_model.solution,
            "get_values",
            wraps=self._opt._solver_model.solution.get_values,
        ) as wrapped_values_call:
            self._opt.load_vars()

            self.assertEqual(wrapped_values_call.call_count, 1)
            self.assertEqual(wrapped_values_call.call_args, tuple())

        self.assertFalse(self._model.X.stale)
        self.assertFalse(self._model.Y.stale)
        self.assertAlmostEqual(value(self._model.X), 3.6)
        self.assertAlmostEqual(value(self._model.Y), 0.8)

    def test_only_specified_vars_are_loaded(self):
        self.assertTrue(self._model.X.stale)
        self.assertTrue(self._model.Y.stale)
        self.assertEqual(value(self._model.X), 0)
        self.assertEqual(value(self._model.Y), 0)

        with unittest.mock.patch.object(
            self._opt._solver_model.solution,
            "get_values",
            wraps=self._opt._solver_model.solution.get_values,
        ) as wrapped_values_call:
            self._opt.load_vars([self._model.X])

            self.assertEqual(wrapped_values_call.call_count, 1)
            self.assertEqual(wrapped_values_call.call_args, (([0],), {}))

        self.assertFalse(self._model.X.stale)
        self.assertTrue(self._model.Y.stale)
        self.assertAlmostEqual(value(self._model.X), 3.6)
        self.assertEqual(value(self._model.Y), 0)

        with unittest.mock.patch.object(
            self._opt._solver_model.solution,
            "get_values",
            wraps=self._opt._solver_model.solution.get_values,
        ) as wrapped_values_call:
            self._opt.load_vars([self._model.Y])

            self.assertEqual(wrapped_values_call.call_count, 1)
            self.assertEqual(wrapped_values_call.call_args, (([1],), {}))

        self.assertFalse(self._model.X.stale)
        self.assertFalse(self._model.Y.stale)
        self.assertAlmostEqual(value(self._model.X), 3.6)
        self.assertAlmostEqual(value(self._model.Y), 0.8)


if __name__ == "__main__":
    unittest.main()
