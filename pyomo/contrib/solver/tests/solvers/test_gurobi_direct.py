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

from pyomo.common.tee import capture_output
from pyomo.common import unittest

from pyomo.contrib.solver.solvers.gurobi_direct import GurobiSolverMixin, GurobiDirect

from pyomo.contrib.solver.common.base import Availability

opt = GurobiDirect()
if not opt.available():
    raise unittest.SkipTest("Gurobi is not available")


class TestGurobiMixin(unittest.TestCase):

    MODULE_PATH = "pyomo.contrib.solver.solvers.gurobi_direct"

    def setUp(self):
        # Reset shared state before each test
        GurobiSolverMixin._gurobipy_env = None
        GurobiSolverMixin._available_cache = None
        GurobiSolverMixin._version_cache = None
        GurobiSolverMixin._num_gurobipy_env_clients = 0

    class GurobiError(Exception):
        def __init__(self, msg="", errno=None):
            super().__init__(msg)
            self.errno = errno

    class Env:
        def __init__(self, *args, **kwargs):
            self.closed = False

        def setParam(self, *args, **kwargs):
            pass

        def close(self):
            self.closed = True

    class Model:
        def __init__(self, env=None, license_status="ok"):
            self.license_status = license_status
            self.Params = type("P", (), {})()
            self.Params.OutputFlag = 0
            self._disposed = False

        def addVars(self, rng):
            return None

        def optimize(self):
            if self.license_status == "ok":
                return
            if self.license_status == "too_large":
                raise TestGurobiMixin.GurobiError("Model too large", errno=10010)
            if self.license_status == "timeout":
                raise TestGurobiMixin.GurobiError("timeout waiting for license")
            if self.license_status == "no_license":
                raise TestGurobiMixin.GurobiError("no gurobi license", errno=10009)
            if self.license_status == "bad":
                raise TestGurobiMixin.GurobiError("other licensing problem")

        def dispose(self):
            self._disposed = True

    @staticmethod
    def mocked_gurobipy(license_status="ok", env_side_effect=None):
        """
        Build a fake gurobipy module.
        - license_status controls Model.optimize() behavior
        - env_side_effect (callable or Exception) controls Env() behavior
          e.g. env_side_effect=TestGurobiMixin.GurobiError("no gurobi license", errno=10009)
        """

        # Arbitrarily picking a version
        class GRB:
            VERSION_MAJOR = 12
            VERSION_MINOR = 0
            VERSION_TECHNICAL = 1

        mocker = unittest.mock.MagicMock()
        if env_side_effect is None:
            mocker.Env = unittest.mock.MagicMock(return_value=TestGurobiMixin.Env())
        else:
            if isinstance(env_side_effect, Exception):
                mocker.Env = unittest.mock.MagicMock(side_effect=env_side_effect)
            else:
                mocker.Env = unittest.mock.MagicMock(side_effect=env_side_effect)
        mocker.Model = unittest.mock.MagicMock(
            side_effect=lambda *a, **kw: TestGurobiMixin.Model(
                license_status=license_status
            )
        )
        mocker.GRB = GRB
        mocker.GurobiError = TestGurobiMixin.GurobiError
        return mocker

    def test_available_notfound(self):
        mixin = GurobiSolverMixin()
        with unittest.mock.patch.object(
            GurobiSolverMixin, "_is_gp_available", return_value=False
        ):
            self.assertEqual(mixin.available(), Availability.NotFound)

    def test_available_full_license(self):
        opt = GurobiDirect()
        mock_gp = self.mocked_gurobipy("ok")
        with (
            unittest.mock.patch.object(
                type(opt), "_is_gp_available", return_value=True
            ),
            unittest.mock.patch(f"{self.MODULE_PATH}.gurobipy", mock_gp),
        ):
            with capture_output(capture_fd=True):
                self.assertEqual(opt.available(recheck=True), Availability.FullLicense)

    def test_available_limited_license(self):
        opt = GurobiDirect()
        mock_gp = self.mocked_gurobipy("too_large")
        with (
            unittest.mock.patch.object(
                GurobiSolverMixin, "_is_gp_available", return_value=True
            ),
            unittest.mock.patch(f"{self.MODULE_PATH}.gurobipy", mock_gp),
        ):
            with capture_output(capture_fd=True):
                self.assertEqual(
                    opt.available(recheck=True), Availability.LimitedLicense
                )

    def test_available_license_error_no_license(self):
        mixin = GurobiSolverMixin()
        env_error = self.GurobiError("no gurobi license", errno=10009)
        mock_gp = self.mocked_gurobipy(license_status="ok", env_side_effect=env_error)
        with (
            unittest.mock.patch.object(
                GurobiSolverMixin, "_is_gp_available", return_value=True
            ),
            unittest.mock.patch(f"{self.MODULE_PATH}.gurobipy", mock_gp),
        ):
            with capture_output(capture_fd=True):
                self.assertEqual(
                    mixin.available(recheck=True), Availability.LicenseError
                )

    def test_available_cache_and_recheck(self):
        opt = GurobiDirect()
        # FullLicense
        mock_full = self.mocked_gurobipy("ok")
        with (
            unittest.mock.patch.object(
                GurobiSolverMixin, "_is_gp_available", return_value=True
            ),
            unittest.mock.patch(f"{self.MODULE_PATH}.gurobipy", mock_full),
        ):
            self.assertEqual(opt.available(recheck=True), Availability.FullLicense)
            # Change behavior to license error; without recheck should use cache
            env_error = self.GurobiError("no gurobi license", errno=10009)
            mock_err = self.mocked_gurobipy("ok", env_side_effect=env_error)
            with unittest.mock.patch(f"{self.MODULE_PATH}.gurobipy", mock_err):
                self.assertEqual(opt.available(), Availability.FullLicense)
                # Now recheck
                self.assertEqual(opt.available(recheck=True), Availability.LicenseError)

    def test_version_cache(self):
        mixin = GurobiSolverMixin()
        mock_gp = self.mocked_gurobipy()
        with unittest.mock.patch(f"{self.MODULE_PATH}.gurobipy", mock_gp):
            self.assertEqual(mixin.version(), (12, 0, 1))
            # Change the version, but we didn't ask for a recheck, so
            # the cached version should stay the same
            mock_gp.GRB.VERSION_MINOR = 99
            self.assertEqual(mixin.version(), (12, 0, 1))

    def test_license_acquire_release(self):
        opt = GurobiDirect()
        mock_gp = self.mocked_gurobipy()
        with unittest.mock.patch(f"{self.MODULE_PATH}.gurobipy", mock_gp):
            # Explicit acquire/release should set and clear the shared Env
            self.assertIsNone(GurobiDirect._gurobipy_env)
            opt.license.acquire()
            self.assertIsNotNone(GurobiDirect._gurobipy_env)
            opt.license.release()
            self.assertIsNone(GurobiDirect._gurobipy_env)


class TestGurobiDirectInterface(unittest.TestCase):
    def test_available_cache(self):
        opt = GurobiDirect()
        opt.available()
        self.assertIsNotNone(opt._available_cache)

    def test_version_cache(self):
        opt = GurobiDirect()
        opt.version()
        self.assertIsNotNone(opt._version_cache)
