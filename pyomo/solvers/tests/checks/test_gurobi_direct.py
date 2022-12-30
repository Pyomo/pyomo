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
