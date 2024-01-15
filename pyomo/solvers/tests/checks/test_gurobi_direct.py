"""
Tests for working with Gurobi environments. Some require a single-use license
and are skipped if this isn't the case.
"""

import gc
from unittest.mock import patch

import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.environ import SolverFactory, ConcreteModel
from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.solvers.plugins.solvers.gurobi_direct import GurobiDirect


try:
    import gurobipy as gp

    NO_LICENSE = gp.GRB.Error.NO_LICENSE
    gurobipy_available = True
except ImportError:
    gurobipy_available = False

gurobi_available = GurobiDirect().available(exception_flag=False)


def clean_up_global_state():
    # Best efforts to dispose any gurobipy objects from previous tests
    # which might keep the default environment active
    gc.collect()
    gp.disposeDefaultEnv()
    # Reset flag to sync with default env state
    GurobiDirect._default_env_started = False


def single_use_license():
    # Return true if the current license is valid and single-use
    if not gurobipy_available:
        return False
    clean_up_global_state()
    try:
        with gp.Env():
            try:
                with gp.Env():
                    # License allows multiple uses
                    return False
            except gp.GurobiError:
                return True
    except gp.GurobiError:
        # No license available
        return False


class GurobiBase(unittest.TestCase):
    # Base class ensures the global environment is cleaned up

    def setUp(self):
        clean_up_global_state()

        # A simple model to solve
        model = ConcreteModel()
        model.x = pyo.Var([1, 2], domain=pyo.NonNegativeReals)
        model.OBJ = pyo.Objective(expr=model.x[1] + model.x[2], sense=pyo.maximize)
        model.Constraint1 = pyo.Constraint(expr=2 * model.x[1] + model.x[2] <= 1)
        model.Constraint2 = pyo.Constraint(expr=model.x[1] + 2 * model.x[2] <= 1)
        self.model = model

    def tearDown(self):
        clean_up_global_state()


@unittest.skipIf(gurobipy_available, "gurobipy is installed, skip import test")
class GurobiImportFailedTests(unittest.TestCase):
    def test_gurobipy_not_installed(self):
        # ApplicationError should be thrown if gurobipy is not available
        model = ConcreteModel()
        with SolverFactory("gurobi_direct") as opt:
            with self.assertRaisesRegex(ApplicationError, "No Python bindings"):
                opt.solve(model)


@unittest.skipIf(not gurobipy_available, "gurobipy is not available")
@unittest.skipIf(not gurobi_available, "gurobi license is not valid")
class GurobiParameterTests(GurobiBase):
    # Test parameter handling at the model and environment level

    def test_set_environment_parameters(self):
        # Solver options should handle parameters which must be set before the
        # environment is started (i.e. connection params, memory limits). This
        # can only work with a managed env.

        with SolverFactory(
            "gurobi_direct", manage_env=True, options={"ComputeServer": "my-cs-url"}
        ) as opt:
            # Check that the error comes from an attempted connection, (i.e. error
            # message reports the hostname) and not from setting the parameter after
            # the environment is started.
            with self.assertRaisesRegex(ApplicationError, "my-cs-url"):
                opt.solve(self.model)

    def test_set_once(self):
        # Make sure parameters aren't set twice. If they are set on the
        # environment, they shouldn't also be set on the model. This isn't an
        # issue for most parameters, but some license parameters (e.g. WLS)
        # will complain if set in both places.

        envparams = {}
        modelparams = {}

        class TempEnv(gp.Env):
            def setParam(self, param, value):
                envparams[param] = value

        class TempModel(gp.Model):
            def setParam(self, param, value):
                modelparams[param] = value

        with patch("gurobipy.Env", new=TempEnv), patch("gurobipy.Model", new=TempModel):
            with SolverFactory(
                "gurobi_direct", options={"Method": 2, "MIPFocus": 1}, manage_env=True
            ) as opt:
                opt.solve(self.model, options={"MIPFocus": 2})

        # Method should not be set again, but MIPFocus was changed.
        # OutputFlag is explicitly set on the model.
        assert envparams == {"Method": 2, "MIPFocus": 1}
        assert modelparams == {"MIPFocus": 2, "OutputFlag": 0}

    # Try an erroneous parameter setting to ensure parameters go through in all
    # cases. Expect an error to indicate pyomo tried to set the parameter.

    def test_param_changes_1(self):
        # Default env: parameters set on model at solve time
        with SolverFactory("gurobi_direct", options={"Method": -100}) as opt:
            with self.assertRaisesRegex(gp.GurobiError, "Unable to set"):
                opt.solve(self.model)

    def test_param_changes_2(self):
        # Note that this case throws an ApplicationError instead of a
        # GurobiError since the bad parameter value prevents the environment
        # from starting
        # Managed env: parameters set on env at solve time
        with SolverFactory(
            "gurobi_direct", options={"Method": -100}, manage_env=True
        ) as opt:
            with self.assertRaisesRegex(ApplicationError, "Unable to set"):
                opt.solve(self.model)

    def test_param_changes_3(self):
        # Default env: parameters passed to solve()
        with SolverFactory("gurobi_direct") as opt:
            with self.assertRaisesRegex(gp.GurobiError, "Unable to set"):
                opt.solve(self.model, options={"Method": -100})

    def test_param_changes_4(self):
        # Managed env: parameters passed to solve()
        with SolverFactory("gurobi_direct", manage_env=True) as opt:
            with self.assertRaisesRegex(gp.GurobiError, "Unable to set"):
                opt.solve(self.model, options={"Method": -100})


@unittest.skipIf(not gurobipy_available, "gurobipy is not available")
@unittest.skipIf(not gurobi_available, "gurobi license is not valid")
class GurobiEnvironmentTests(GurobiBase):
    # Test handling of gurobi environments

    def assert_optimal_result(self, results):
        self.assertEqual(results.solver.status, SolverStatus.ok)
        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.optimal
        )

    def test_init_default_env(self):
        # available() calls with the default env shouldn't need a repeat check
        with patch("gurobipy.Model") as PatchModel:
            with SolverFactory("gurobi_direct") as opt:
                opt.available()
                opt.available()
                PatchModel.assert_called_once_with()

    def test_close_global(self):
        # method releases the license and syncs the flag
        with patch("gurobipy.Model") as PatchModel, patch(
            "gurobipy.disposeDefaultEnv"
        ) as patch_dispose:
            with SolverFactory("gurobi_direct") as opt:
                opt.available()
                opt.available()
                PatchModel.assert_called_once_with()
            patch_dispose.assert_not_called()

            # close default environment
            opt.close_global()
            patch_dispose.assert_called_once_with()

        # _default_env_started flag was correctly synced, so available() is
        # checked again
        with patch("gurobipy.Model") as PatchModel, patch(
            "gurobipy.disposeDefaultEnv"
        ) as patch_dispose:
            with SolverFactory("gurobi_direct") as opt:
                opt.available()
                opt.available()
                PatchModel.assert_called_once_with()
            patch_dispose.assert_not_called()

    def test_persisted_license_failure(self):
        # Gurobi error message should come through in the exception
        # Failure to start an environment should not be persistent

        with patch(
            "gurobipy.Model", side_effect=gp.GurobiError(NO_LICENSE, "nolicense")
        ):
            with SolverFactory("gurobi_direct") as opt:
                with self.assertRaisesRegex(ApplicationError, "nolicense"):
                    opt.solve(self.model)

        with SolverFactory("gurobi_direct") as opt:
            results = opt.solve(self.model)
            self.assert_optimal_result(results)

    def test_persisted_license_failure_managed(self):
        # Gurobi error message should come through in the exception
        # Failure to start an environment should not be persistent

        with patch("gurobipy.Env", side_effect=gp.GurobiError(NO_LICENSE, "nolicense")):
            with SolverFactory("gurobi_direct", manage_env=True) as opt:
                with self.assertRaisesRegex(ApplicationError, "nolicense"):
                    opt.solve(self.model)

        with SolverFactory("gurobi_direct", manage_env=True) as opt:
            results = opt.solve(self.model)
            self.assert_optimal_result(results)
            self.assertEqual(results.solver.status, SolverStatus.ok)

    def test_context(self):
        # Context management should close the gurobi environment

        with gp.Env() as use_env:
            with patch("gurobipy.Env", return_value=use_env):
                with SolverFactory("gurobi_direct", manage_env=True) as opt:
                    results = opt.solve(self.model)
                    self.assert_optimal_result(results)

            # Environment was closed (cannot be restarted)
            with self.assertRaises(gp.GurobiError):
                use_env.start()

    def test_close(self):
        # Manual close() call should close the gurobi environment

        with gp.Env() as use_env:
            with patch("gurobipy.Env", return_value=use_env):
                opt = SolverFactory("gurobi_direct", manage_env=True)
                try:
                    results = opt.solve(self.model)
                    self.assert_optimal_result(results)
                finally:
                    opt.close()

            # Environment was closed (cannot be restarted)
            with self.assertRaises(gp.GurobiError):
                use_env.start()

    @unittest.skipIf(single_use_license(), reason="test requires multi-use license")
    def test_multiple_solvers_managed(self):
        # Multiple managed solvers will create their own envs

        with SolverFactory("gurobi_direct", manage_env=True) as opt1, SolverFactory(
            "gurobi_direct", manage_env=True
        ) as opt2:
            results1 = opt1.solve(self.model)
            self.assert_optimal_result(results1)
            results2 = opt2.solve(self.model)
            self.assert_optimal_result(results2)

    def test_multiple_solvers_nonmanaged(self):
        # Multiple solvers will share the default environment

        with SolverFactory("gurobi_direct") as opt1, SolverFactory(
            "gurobi_direct"
        ) as opt2:
            results1 = opt1.solve(self.model)
            self.assert_optimal_result(results1)
            results2 = opt2.solve(self.model)
            self.assert_optimal_result(results2)

    @unittest.skipIf(single_use_license(), reason="test requires multi-use license")
    def test_managed_env(self):
        # Test that manage_env=True creates its own environment

        # Set parameters on the default environment
        gp.setParam("IterationLimit", 100)

        # On the patched environment, solve times out due to parameter setting
        with gp.Env(params={"IterationLimit": 0, "Presolve": 0}) as use_env, patch(
            "gurobipy.Env", return_value=use_env
        ):
            with SolverFactory("gurobi_direct", manage_env=True) as opt:
                results = opt.solve(self.model)
                self.assertEqual(results.solver.status, SolverStatus.aborted)
                self.assertEqual(
                    results.solver.termination_condition,
                    TerminationCondition.maxIterations,
                )

    def test_nonmanaged_env(self):
        # Test that manage_env=False (default) uses the default environment

        # Set parameters on the default environment
        gp.setParam("IterationLimit", 0)
        gp.setParam("Presolve", 0)

        # Using the default env, solve times out due to parameter setting
        with SolverFactory("gurobi_direct") as opt:
            results = opt.solve(self.model)
            self.assertEqual(results.solver.status, SolverStatus.aborted)
            self.assertEqual(
                results.solver.termination_condition, TerminationCondition.maxIterations
            )


@unittest.skipIf(not gurobipy_available, "gurobipy is not available")
@unittest.skipIf(not gurobi_available, "gurobi license is not valid")
@unittest.skipIf(not single_use_license(), reason="test needs a single use license")
class GurobiSingleUseTests(GurobiBase):
    # Integration tests for Gurobi single-use licenses (useful for checking all Gurobi
    # environments were correctly freed). These tests are not run in pyomo's CI. Each
    # test in this class has an equivalent in GurobiEnvironmentTests which tests the
    # same behaviour via monkey patching.

    def test_persisted_license_failure(self):
        # Solver should allow retries to start the environment, instead of
        # persisting the same failure (default env).

        with SolverFactory("gurobi_direct") as opt:
            with gp.Env():
                # Expected to fail: there is another environment open so the
                # default env cannot be started.
                with self.assertRaises(ApplicationError):
                    opt.solve(self.model)
            # Should not raise an error, since the other environment has been freed.
            opt.solve(self.model)

    def test_persisted_license_failure_managed(self):
        # Solver should allow retries to start the environment, instead of
        # persisting the same failure (managed env).

        with SolverFactory("gurobi_direct", manage_env=True) as opt:
            with gp.Env():
                # Expected to fail: there is another environment open so the
                # default env cannot be started.
                with self.assertRaises(ApplicationError):
                    opt.solve(self.model)
            # Should not raise an error, since the other environment has been freed.
            opt.solve(self.model)

    def test_context(self):
        # Context management should close the gurobi environment.
        with SolverFactory("gurobi_direct", manage_env=True) as opt:
            opt.solve(self.model)

        # Environment closed, so another can be created
        with gp.Env():
            pass

    def test_close(self):
        # Manual close() call should close the gurobi environment.
        opt = SolverFactory("gurobi_direct", manage_env=True)
        try:
            opt.solve(self.model)
        finally:
            opt.close()

        # Environment closed, so another can be created
        with gp.Env():
            pass

    def test_multiple_solvers(self):
        # One environment per solver would break this pattern. Test that
        # global env is still used by default (manage_env=False)

        with SolverFactory("gurobi_direct") as opt1, SolverFactory(
            "gurobi_direct"
        ) as opt2:
            opt1.solve(self.model)
            opt2.solve(self.model)

    def test_multiple_models_leaky(self):
        # Make sure all models are closed explicitly by the GurobiDirect instance.

        with SolverFactory("gurobi_direct", manage_env=True) as opt:
            opt.solve(self.model)
            # Leak a model reference, then create a new model.
            # Pyomo should close the old model since it is no longed needed.
            tmp = opt._solver_model
            opt.solve(self.model)

        # Context manager properly closed all models and environments
        with gp.Env():
            pass

    def test_close_global(self):
        # If using the default environment, calling the close_global
        # classmethod closes the environment, providing any other solvers
        # have also been closed.

        opt1 = SolverFactory("gurobi_direct")
        opt2 = SolverFactory("gurobi_direct")
        try:
            opt1.solve(self.model)
            opt2.solve(self.model)
        finally:
            opt1.close()
            opt2.close_global()

        # Context closed AND close_global called
        with gp.Env():
            pass
