#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import os
import pyomo.common.unittest as unittest
from unittest import mock
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
from pyomo.opt.results import SolverStatus, TerminationCondition, ProblemSense
from pyomo.opt import SolverFactory
import warnings

CFGFILE = os.environ.get("SAS_CFG_FILE_PATH", None)

CAS_OPTIONS = {
    "hostname": os.environ.get("CASHOST", None),
    "port": os.environ.get("CASPORT", None),
    "authinfo": os.environ.get("CASAUTHINFO", None),
}


try:
    sas94_available = SolverFactory('_sas94').available()
except:
    sas94_available = False


class SASTestAbc:
    solver_io = "_sas94"
    session_options = {}
    cfgfile = CFGFILE

    @classmethod
    def setUpClass(cls):
        cls.opt_sas = SolverFactory(
            "sas", solver_io=cls.solver_io, cfgfile=cls.cfgfile, **cls.session_options
        )

    @classmethod
    def tearDownClass(cls):
        del cls.opt_sas

    def setObj(self):
        X = self.instance.X
        self.instance.Obj = Objective(
            expr=2 * X[1] - 3 * X[2] - 4 * X[3], sense=minimize
        )

    def setX(self):
        self.instance.X = Var([1, 2, 3], within=NonNegativeReals)

    def setUp(self):
        # Disable resource warnings
        warnings.filterwarnings("ignore", category=ResourceWarning)
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

    def tearDown(self):
        del self.instance

    def run_solver(self, **kwargs):
        opt_sas = self.opt_sas
        instance = self.instance

        # Call the solver
        self.results = opt_sas.solve(instance, **kwargs)


class SASTestLP(SASTestAbc):
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
        self.assertEqual(instance.status[instance.X[1]], "L")
        self.assertEqual(instance.status[instance.X[2]], "B")
        self.assertEqual(instance.status[instance.X[3]], "L")
        self.assertEqual(instance.status[instance.R1], "U")
        self.assertEqual(instance.status[instance.R2], "B")
        self.assertEqual(instance.status[instance.R3], "B")

    def test_solver_default(self):
        self.run_solver()
        self.checkSolution()

    def test_solver_tee(self):
        self.run_solver(tee=True)
        self.checkSolution()

    def test_solver_primal(self):
        self.run_solver(options={"algorithm": "ps"})
        self.assertIn("NOTE: The Primal Simplex algorithm is used.", self.opt_sas._log)
        self.checkSolution()

    def test_solver_ipm(self):
        self.run_solver(options={"algorithm": "ip"})
        self.assertIn("NOTE: The Interior Point algorithm is used.", self.opt_sas._log)
        self.checkSolution()

    def test_solver_intoption(self):
        self.run_solver(options={"maxiter": 20})
        self.checkSolution()

    def test_solver_invalidoption(self):
        with self.assertRaisesRegex(ValueError, "syntax error"):
            self.run_solver(options={"foo": "bar"})

    def test_solver_max(self):
        X = self.instance.X
        self.instance.Obj.set_value(expr=-2 * X[1] + 3 * X[2] + 4 * X[3])
        self.instance.Obj.sense = maximize
        self.run_solver()
        self.checkSolution()
        self.assertEqual(self.results.problem.sense, ProblemSense.maximize)

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

    def test_solver_infeasible_or_unbounded(self):
        self.instance.X.domain = Reals
        self.run_solver()
        results = self.results
        self.assertEqual(results.solver.status, SolverStatus.warning)
        self.assertIn(
            results.solver.termination_condition,
            [
                TerminationCondition.infeasibleOrUnbounded,
                TerminationCondition.unbounded,
            ],
        )
        self.assertIn(
            results.solver.message,
            ["The problem is infeasible or unbounded.", "The problem is unbounded."],
        )

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

    def test_solver_with_milp(self):
        self.run_solver(options={"with": "milp"})
        self.assertIn(
            "WARNING: The problem has no integer variables.", self.opt_sas._log
        )


@unittest.skipIf(not sas94_available, "The SAS94 solver interface is not available")
class SASTestLP94(SASTestLP, unittest.TestCase):
    @mock.patch(
        "pyomo.solvers.plugins.solvers.SAS.SAS94.sas_version",
        return_value="9.sd45s39M4234232",
    )
    def test_solver_versionM4(self, sas):
        with self.assertRaises(NotImplementedError):
            self.run_solver()

    @mock.patch(
        "pyomo.solvers.plugins.solvers.SAS.SAS94.sas_version",
        return_value="9.34897293M5324u98",
    )
    def test_solver_versionM5(self, sas):
        self.run_solver()
        self.checkSolution()

    @mock.patch("saspy.SASsession.submit", return_value={"LOG": "", "LST": ""})
    @mock.patch("saspy.SASsession.symget", return_value="STATUS=OUT_OF_MEMORY")
    def test_solver_out_of_memory(self, submit_mock, symget_mocks):
        self.run_solver(load_solutions=False)
        results = self.results
        self.assertEqual(results.solver.status, SolverStatus.aborted)

    @mock.patch("saspy.SASsession.submit", return_value={"LOG": "", "LST": ""})
    @mock.patch("saspy.SASsession.symget", return_value="STATUS=ERROR")
    def test_solver_error(self, submit_mock, symget_mock):
        self.run_solver(load_solutions=False)
        results = self.results
        self.assertEqual(results.solver.status, SolverStatus.error)


# @unittest.skipIf(not sascas_available, "The SAS solver is not available")
@unittest.skip("Tests not yet configured for SAS Viya interface.")
class SASTestLPCAS(SASTestLP, unittest.TestCase):
    solver_io = "_sascas"
    session_options = CAS_OPTIONS

    @mock.patch("pyomo.solvers.plugins.solvers.SAS.stat")
    def test_solver_large_file(self, os_stat):
        os_stat.return_value.st_size = 3 * 1024**3
        self.run_solver()
        self.checkSolution()


class SASTestMILP(SASTestAbc):
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

    def test_solver_default(self):
        self.run_solver()
        self.checkSolution()

    def test_solver_tee(self):
        self.run_solver(tee=True)
        self.checkSolution()

    def test_solver_presolve(self):
        self.run_solver(options={"presolver": "none"})
        self.assertIn(
            "NOTE: The MILP presolver value NONE is applied.", self.opt_sas._log
        )
        self.checkSolution()

    def test_solver_intoption(self):
        self.run_solver(options={"maxnodes": 20})
        self.checkSolution()

    def test_solver_invalidoption(self):
        with self.assertRaisesRegex(ValueError, "syntax error"):
            self.run_solver(options={"foo": "bar"})

    def test_solver_max(self):
        X = self.instance.X
        self.instance.Obj.set_value(expr=-2 * X[1] + 3 * X[2] + 4 * X[3])
        self.instance.Obj.sense = maximize
        self.run_solver()
        self.checkSolution()

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

    def test_solver_infeasible_or_unbounded(self):
        self.instance.X.domain = Integers
        self.run_solver()
        results = self.results
        self.assertEqual(results.solver.status, SolverStatus.warning)
        self.assertIn(
            results.solver.termination_condition,
            [
                TerminationCondition.infeasibleOrUnbounded,
                TerminationCondition.unbounded,
            ],
        )
        self.assertIn(
            results.solver.message,
            ["The problem is infeasible or unbounded.", "The problem is unbounded."],
        )

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

    def test_solver_rootnode(self):
        self.run_solver(options={"rootnode": {"presolver": "automatic"}})
        self.checkSolution()

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

    def test_solver_primalin_nosol(self):
        X = self.instance.X
        X[1] = None
        X[2] = None
        X[3] = None
        self.run_solver(warmstart=True)
        self.checkSolution()

    @mock.patch("pyomo.solvers.plugins.solvers.SAS.stat")
    def test_solver_large_file(self, os_stat):
        os_stat.return_value.st_size = 3 * 1024**3
        self.run_solver()
        self.checkSolution()

    def test_solver_with_lp(self):
        self.run_solver(options={"with": "lp"})
        self.assertIn(
            "contains integer variables; the linear relaxation will be solved.",
            self.opt_sas._log,
        )

    def test_solver_warmstart_capable(self):
        self.run_solver()
        self.assertTrue(self.opt_sas.warm_start_capable())


# @unittest.skipIf(not sas94_available, "The SAS solver is not available")
@unittest.skip("MILP94 tests disabled.")
class SASTestMILP94(SASTestMILP, unittest.TestCase):
    pass


# @unittest.skipIf(not sascas_available, "The SAS solver is not available")
@unittest.skip("Tests not yet configured for SAS Viya interface.")
class SASTestMILPCAS(SASTestMILP, unittest.TestCase):
    solver_io = "_sascas"
    session_options = CAS_OPTIONS


if __name__ == "__main__":
    unittest.main()
