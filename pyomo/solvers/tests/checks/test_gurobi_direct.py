"""
Tests for working with Gurobi environments. Some require a single-use license
and are skipped if this isn't the case.
"""

import gc
from unittest.mock import patch

import pyomo.common.unittest as unittest
from pyomo.environ import SolverFactory, ConcreteModel
from pyomo.common.errors import ApplicationError

try:
    import gurobipy as gp

    gurobipy_available = True
except ImportError:
    gurobipy_available = False


def clean_up_global_state():

    # Clean up GurobiDirect's persistent error storage. Can be removed
    # once GurobiDirect is updated.
    from pyomo.solvers.plugins.solvers.gurobi_direct import GurobiDirect

    GurobiDirect._verified_license = None
    GurobiDirect._import_messages = ""

    # Best efforts to dispose any gurobipy objects from previous tests
    # which might keep the default environment active
    gc.collect()
    gp.disposeDefaultEnv()


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
    def setUp(self):
        clean_up_global_state()
        self.model = ConcreteModel()

    def tearDown(self):
        clean_up_global_state()


@unittest.skipIf(not gurobipy_available, "gurobipy is not available")
class GurobiParameterTests(GurobiBase):
    def test_set_environment_parameters(self):
        # Solver options should handle parameters which must be set before the
        # environment is started (i.e. connection params, memory limits). This
        # can only work with a managed env.

        with SolverFactory(
            "gurobi_direct",
            manage_env=True,
            options={"ComputeServer": "/url/to/server"},
        ) as opt:
            # Check that the error comes from an attempted connection, not from setting
            # the parameter after the environment is started.
            with self.assertRaisesRegex(ApplicationError, "Could not resolve host"):
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
        # Managed env: parameters set on env at solve time
        with SolverFactory(
            "gurobi_direct", options={"Method": -100}, manage_env=True
        ) as opt:
            with self.assertRaisesRegex(gp.GurobiError, "Unable to set"):
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
@unittest.skipIf(not single_use_license(), reason="test needs a single use license")
class GurobiSingleUseTests(GurobiBase):
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

        with SolverFactory("gurobi_direct") as opt1, SolverFactory("gurobi_direct") as opt2:
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
