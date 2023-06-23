import os
import pyomo.common.unittest as unittest
from pyomo.environ import (
    ConcreteModel,
    Var,
    Objective,
    Constraint,
    NonNegativeIntegers,
    NonNegativeReals,
    Reals,
    Integers,
    maximize,
    minimize,
    Suffix,
)
from pyomo.opt.results import (
    SolverStatus,
    TerminationCondition,
    ProblemSense,
)
from pyomo.opt import (
    SolverFactory,
    check_available_solvers,
)


CAS_OPTIONS = {
    "hostname": os.environ.get('CAS_SERVER', None),
    "port": os.environ.get('CAS_PORT', None),
    "authinfo": os.environ.get('CAS_AUTHINFO', None),
}


sas_available = check_available_solvers('sas')


class SASTestAbc:
    solver_io = '_sas94'
    base_options = {}

    def setObj(self):
        X = self.instance.X
        self.instance.Obj = Objective(
            expr=2 * X[1] - 3 * X[2] - 4 * X[3], sense=minimize
        )

    def setX(self):
        self.instance.X = Var([1, 2, 3], within=NonNegativeReals)

    def setUp(self):
        instance = self.instance = ConcreteModel()
        self.setX()
        X = instance.X
        instance.R1 = Constraint(expr=-2 * X[2] - 3 * X[3] >= -5)
        instance.R2 = Constraint(expr=X[1] + X[2] + 2 * X[3] <= 4)
        instance.R3 = Constraint(expr=X[1] + 2 * X[2] + 3 * X[3] <= 7)
        self.setObj()

        # Declare suffixes for solution information
        instance.status = Suffix(direction=Suffix.IMPORT)
        instance.slack = Suffix(direction=Suffix.IMPORT)
        instance.rc = Suffix(direction=Suffix.IMPORT)
        instance.dual = Suffix(direction=Suffix.IMPORT)

        self.opt_sas = SolverFactory('sas', solver_io=self.solver_io)

    def tearDown(self):
        del self.opt_sas
        del self.instance

    def run_solver(self, **kwargs):
        opt_sas = self.opt_sas
        instance = self.instance

        # Add base options for connection data etc.
        options = kwargs.get("options", {})
        if self.base_options:
            kwargs["options"] = {**options, **self.base_options}

        # Call the solver
        self.results = opt_sas.solve(instance, **kwargs)


class SASTestLP(SASTestAbc, unittest.TestCase):
    def checkSolution(self):
        instance = self.instance
        results = self.results
        # Get the objective sense, we use the same code for minimization and maximization tests
        sense = instance.Obj.sense

        # Check status
        self.assertEqual(results.solver.status, SolverStatus.ok)
        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.optimal
        )

        # Check objective value
        self.assertAlmostEqual(instance.Obj(), sense * -7.5)

        # Check primal solution values
        self.assertAlmostEqual(instance.X[1].value, 0.0)
        self.assertAlmostEqual(instance.X[2].value, 2.5)
        self.assertAlmostEqual(instance.X[3].value, 0.0)

        # Check reduced cost
        self.assertAlmostEqual(instance.rc[instance.X[1]], sense * 2.0)
        self.assertAlmostEqual(instance.rc[instance.X[2]], sense * 0.0)
        self.assertAlmostEqual(instance.rc[instance.X[3]], sense * 0.5)

        # Check slack
        self.assertAlmostEqual(instance.slack[instance.R1], -5.0)
        self.assertAlmostEqual(instance.slack[instance.R2], 2.5)
        self.assertAlmostEqual(instance.slack[instance.R3], 5.0)

        # Check dual solution
        self.assertAlmostEqual(instance.dual[instance.R1], sense * 1.5)
        self.assertAlmostEqual(instance.dual[instance.R2], sense * 0.0)
        self.assertAlmostEqual(instance.dual[instance.R3], sense * 0.0)

        # Check basis status
        self.assertEqual(instance.status[instance.X[1]], 'L')
        self.assertEqual(instance.status[instance.X[2]], 'B')
        self.assertEqual(instance.status[instance.X[3]], 'L')
        self.assertEqual(instance.status[instance.R1], 'U')
        self.assertEqual(instance.status[instance.R2], 'B')
        self.assertEqual(instance.status[instance.R3], 'B')

    @unittest.skipIf(not sas_available, "The SAS solver is not available")
    def test_solver_default(self):
        self.run_solver()
        self.checkSolution()

    @unittest.skipIf(not sas_available, "The SAS solver is not available")
    def test_solver_primal(self):
        self.run_solver(options={"algorithm": "ps"})
        self.assertIn("NOTE: The Primal Simplex algorithm is used.", self.opt_sas._log)
        self.checkSolution()

    @unittest.skipIf(not sas_available, "The SAS solver is not available")
    def test_solver_ipm(self):
        self.run_solver(options={"algorithm": "ip"})
        self.assertIn("NOTE: The Interior Point algorithm is used.", self.opt_sas._log)
        self.checkSolution()

    @unittest.skipIf(not sas_available, "The SAS solver is not available")
    def test_solver_intoption(self):
        self.run_solver(options={"maxiter": 20})
        self.checkSolution()

    @unittest.skipIf(not sas_available, "The SAS solver is not available")
    def test_solver_invalidoption(self):
        with self.assertRaisesRegex(ValueError, "syntax error"):
            self.run_solver(options={"foo": "bar"})

    @unittest.skipIf(not sas_available, "The SAS solver is not available")
    def test_solver_max(self):
        X = self.instance.X
        self.instance.Obj.set_value(expr=-2 * X[1] + 3 * X[2] + 4 * X[3])
        self.instance.Obj.sense = maximize
        self.run_solver()
        self.checkSolution()
        self.assertEqual(self.results.problem.sense, ProblemSense.maximize)

    @unittest.skipIf(not sas_available, "The SAS solver is not available")
    def test_solver_infeasible(self):
        instance = self.instance
        X = instance.X
        instance.R4 = Constraint(expr=-2 * X[2] - 3 * X[3] <= -6)
        self.run_solver()
        results = self.results
        self.assertEqual(results.solver.status, SolverStatus.warning)
        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.infeasible
        )
        self.assertEqual(results.solver.message, "The problem is infeasible.")

    @unittest.skipIf(not sas_available, "The SAS solver is not available")
    def test_solver_infeasible_or_unbounded(self):
        self.instance.X.domain = Reals
        self.run_solver()
        results = self.results
        self.assertEqual(results.solver.status, SolverStatus.warning)
        self.assertEqual(
            results.solver.termination_condition,
            TerminationCondition.infeasibleOrUnbounded,
        )
        self.assertEqual(
            results.solver.message, "The problem is infeasible or unbounded."
        )

    @unittest.skipIf(not sas_available, "The SAS solver is not available")
    def test_solver_unbounded(self):
        self.instance.X.domain = Reals
        self.run_solver(options={"presolver": "none", "algorithm": "primal"})
        results = self.results
        self.assertEqual(results.solver.status, SolverStatus.warning)
        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.unbounded
        )
        self.assertEqual(results.solver.message, "The problem is unbounded.")

    def checkSolutionDecomp(self):
        instance = self.instance
        results = self.results
        # Get the objective sense, we use the same code for minimization and maximization tests
        sense = instance.Obj.sense

        # Check status
        self.assertEqual(results.solver.status, SolverStatus.ok)
        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.optimal
        )

        # Check objective value
        self.assertAlmostEqual(instance.Obj(), sense * -7.5)

        # Check primal solution values
        self.assertAlmostEqual(instance.X[1].value, 0.0)
        self.assertAlmostEqual(instance.X[2].value, 2.5)
        self.assertAlmostEqual(instance.X[3].value, 0.0)

        # Check reduced cost
        self.assertAlmostEqual(instance.rc[instance.X[1]], sense * 2.0)
        self.assertAlmostEqual(instance.rc[instance.X[2]], sense * 0.0)
        self.assertAlmostEqual(instance.rc[instance.X[3]], sense * 0.5)

        # Check slack
        self.assertAlmostEqual(instance.slack[instance.R1], -5.0)
        self.assertAlmostEqual(instance.slack[instance.R2], 2.5)
        self.assertAlmostEqual(instance.slack[instance.R3], 5.0)

        # Check dual solution
        self.assertAlmostEqual(instance.dual[instance.R1], sense * 1.5)
        self.assertAlmostEqual(instance.dual[instance.R2], sense * 0.0)
        self.assertAlmostEqual(instance.dual[instance.R3], sense * 0.0)

        # Don't check basis status for decomp

    @unittest.skipIf(not sas_available, "The SAS solver is not available")
    def test_solver_decomp(self):
        self.run_solver(
            options={
                "decomp": {"absobjgap": 0.0},
                "decompmaster": {"algorithm": "dual"},
                "decompsubprob": {"presolver": "none"},
            }
        )
        self.assertIn(
            "NOTE: The DECOMP method value DEFAULT is applied.", self.opt_sas._log
        )
        self.checkSolutionDecomp()

    @unittest.skipIf(not sas_available, "The SAS solver is not available")
    def test_solver_iis(self):
        self.run_solver(options={"iis": "true"})
        results = self.results
        self.assertEqual(results.solver.status, SolverStatus.ok)
        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.feasible
        )
        self.assertIn("NOTE: The IIS= option is enabled.", self.opt_sas._log)
        self.assertEqual(
            results.solver.message,
            "The problem is feasible. This status is displayed when the IIS=TRUE option is specified and the problem is feasible.",
        )

    @unittest.skipIf(not sas_available, "The SAS solver is not available")
    def test_solver_maxiter(self):
        self.run_solver(options={"maxiter": 1})
        results = self.results
        self.assertEqual(results.solver.status, SolverStatus.ok)
        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.maxIterations
        )
        self.assertEqual(
            results.solver.message,
            "The maximum allowable number of iterations was reached.",
        )


class SASTestLPCAS(SASTestLP):
    solver_io = '_sascas'
    base_options = CAS_OPTIONS


class SASTestMILP(SASTestAbc, unittest.TestCase):
    def setX(self):
        self.instance.X = Var([1, 2, 3], within=NonNegativeIntegers)

    def checkSolution(self):
        instance = self.instance
        results = self.results

        # Get the objective sense, we use the same code for minimization and maximization tests
        sense = instance.Obj.sense

        # Check status
        self.assertEqual(results.solver.status, SolverStatus.ok)
        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.optimal
        )

        # Check objective value
        self.assertAlmostEqual(instance.Obj(), sense * -7)

        # Check primal solution values
        self.assertAlmostEqual(instance.X[1].value, 0.0)
        self.assertAlmostEqual(instance.X[2].value, 1.0)
        self.assertAlmostEqual(instance.X[3].value, 1.0)

    @unittest.skipIf(not sas_available, "The SAS solver is not available")
    def test_solver_default(self):
        self.run_solver(options={})
        self.checkSolution()

    @unittest.skipIf(not sas_available, "The SAS solver is not available")
    def test_solver_presolve(self):
        self.run_solver(options={"presolver": "none"})
        self.assertIn(
            "NOTE: The MILP presolver value NONE is applied.", self.opt_sas._log
        )
        self.checkSolution()

    @unittest.skipIf(not sas_available, "The SAS solver is not available")
    def test_solver_intoption(self):
        self.run_solver(options={"maxnodes": 20})
        self.checkSolution()

    @unittest.skipIf(not sas_available, "The SAS solver is not available")
    def test_solver_invalidoption(self):
        with self.assertRaisesRegex(ValueError, "syntax error"):
            self.run_solver(options={"foo": "bar"})

    @unittest.skipIf(not sas_available, "The SAS solver is not available")
    def test_solver_max(self):
        X = self.instance.X
        self.instance.Obj.set_value(expr=-2 * X[1] + 3 * X[2] + 4 * X[3])
        self.instance.Obj.sense = maximize
        self.run_solver()
        self.checkSolution()

    @unittest.skipIf(not sas_available, "The SAS solver is not available")
    def test_solver_infeasible(self):
        instance = self.instance
        X = instance.X
        instance.R4 = Constraint(expr=-2 * X[2] - 3 * X[3] <= -6)
        self.run_solver()
        results = self.results
        self.assertEqual(results.solver.status, SolverStatus.warning)
        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.infeasible
        )
        self.assertEqual(results.solver.message, "The problem is infeasible.")

    @unittest.skipIf(not sas_available, "The SAS solver is not available")
    @unittest.skip("Returns wrong status for some versions.")
    def test_solver_infeasible_or_unbounded(self):
        self.instance.X.domain = Integers
        self.run_solver()
        results = self.results
        self.assertEqual(results.solver.status, SolverStatus.warning)
        self.assertEqual(
            results.solver.termination_condition,
            TerminationCondition.infeasibleOrUnbounded,
        )
        self.assertEqual(
            results.solver.message, "The problem is infeasible or unbounded."
        )

    @unittest.skipIf(not sas_available, "The SAS solver is not available")
    def test_solver_unbounded(self):
        self.instance.X.domain = Integers
        self.run_solver(
            options={"presolver": "none", "rootnode": {"algorithm": "primal"}}
        )
        results = self.results
        self.assertEqual(results.solver.status, SolverStatus.warning)
        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.unbounded
        )
        self.assertEqual(results.solver.message, "The problem is unbounded.")

    @unittest.skipIf(not sas_available, "The SAS solver is not available")
    def test_solver_decomp(self):
        self.run_solver(
            options={
                "decomp": {"hybrid": "off"},
                "decompmaster": {"algorithm": "dual"},
                "decompmasterip": {"presolver": "none"},
                "decompsubprob": {"presolver": "none"},
            }
        )
        self.assertIn(
            "NOTE: The DECOMP method value DEFAULT is applied.", self.opt_sas._log
        )
        self.checkSolution()

    @unittest.skipIf(not sas_available, "The SAS solver is not available")
    def test_solver_rootnode(self):
        self.run_solver(options={"rootnode": {"presolver": "automatic"}})
        self.checkSolution()

    @unittest.skipIf(not sas_available, "The SAS solver is not available")
    def test_solver_maxnodes(self):
        self.run_solver(options={"maxnodes": 0})
        results = self.results
        self.assertEqual(results.solver.status, SolverStatus.ok)
        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.maxEvaluations
        )
        self.assertEqual(
            results.solver.message,
            "The solver reached the maximum number of nodes specified by the MAXNODES= option and found a solution.",
        )

    @unittest.skipIf(not sas_available, "The SAS solver is not available")
    def test_solver_maxsols(self):
        self.run_solver(options={"maxsols": 1})
        results = self.results
        self.assertEqual(results.solver.status, SolverStatus.ok)
        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.maxEvaluations
        )
        self.assertEqual(
            results.solver.message,
            "The solver reached the maximum number of solutions specified by the MAXSOLS= option.",
        )

    @unittest.skipIf(not sas_available, "The SAS solver is not available")
    def test_solver_target(self):
        self.run_solver(options={"target": -6.0})
        results = self.results
        self.assertEqual(results.solver.status, SolverStatus.ok)
        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.optimal
        )
        self.assertEqual(
            results.solver.message,
            "The solution is not worse than the target specified by the TARGET= option.",
        )

    @unittest.skipIf(not sas_available, "The SAS solver is not available")
    def test_solver_primalin(self):
        X = self.instance.X
        X[1] = None
        X[2] = 3
        X[3] = 7
        self.run_solver(warmstart=True)
        self.checkSolution()
        self.assertIn(
            "NOTE: The input solution is infeasible or incomplete. Repair heuristics are applied.",
            self.opt_sas._log,
        )


class SASTestMILPCAS(SASTestMILP):
    solver_io = '_sascas'
    base_options = CAS_OPTIONS


if __name__ == '__main__':
    unittest.main()
