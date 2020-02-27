#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import sys
from itertools import product
from random import random, seed

import pyutilib.th as unittest

from pyomo.environ import *
from pyomo.opt import *

try:
    import cplex
    cplexpy_available = True
except ImportError:
    cplexpy_available = False

diff_tol = 1e-4

class CPLEXDirectTests(unittest.TestCase):

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

            model = build_mtz_tsp_model(nodes, links, distances)

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

if __name__ == "__main__":
    unittest.main()
