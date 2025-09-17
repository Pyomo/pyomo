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

from pyomo.contrib.solver.solvers.gurobi_direct import (
    gurobipy_available,
    GurobiSolverMixin,
    GurobiDirect,
)
from pyomo.contrib.solver.common.availability import (
    SolverAvailability,
    LicenseAvailability,
)


class TestGurobiMixin(unittest.TestCase):

    MODULE_PATH = "pyomo.contrib.solver.solvers.gurobi_direct"

    def setUp(self):
        # Reset shared state before each test
        GurobiSolverMixin._gurobipy_env = None
        GurobiSolverMixin._license_cache = None
        GurobiSolverMixin._available_cache = None
        GurobiSolverMixin._version_cache = None
        GurobiSolverMixin._num_gurobipy_env_clients = 0

    class GurobiError(Exception):
        def __init__(self, msg="", errno=None):
            super().__init__(msg)
            self.errno = errno

    class Env:
        pass

    class Model:
        def __init__(self, env=None, license_status="ok"):
            self.license_status = license_status
            self.disposed = False

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
            self.disposed = True

    @staticmethod
    def mocked_gurobipy(license_status="ok"):
        class GRB:
            # Arbitrarily picking a version
            VERSION_MAJOR = 12
            VERSION_MINOR = 0
            VERSION_TECHNICAL = 1

            class Param:
                OutputFlag = 0

        mocker = unittest.mock.MagicMock()
        mocker.Env = unittest.mock.MagicMock(return_value=TestGurobiMixin.Env())
        mocker.Model = unittest.mock.MagicMock(
            side_effect=lambda **kw: TestGurobiMixin.Model(
                license_status=license_status
            )
        )
        mocker.GRB = GRB
        mocker.GurobiError = TestGurobiMixin.GurobiError
        return mocker

    def test_solver_available(self):
        mixin = GurobiSolverMixin()
        with unittest.mock.patch.object(GurobiSolverMixin, "_gurobipy_available", True):
            self.assertEqual(mixin.solver_available(), SolverAvailability.Available)

    def test_solver_unavailable(self):
        mixin = GurobiSolverMixin()
        with unittest.mock.patch.object(
            GurobiSolverMixin, "_gurobipy_available", False
        ):
            self.assertEqual(mixin.solver_available(), SolverAvailability.NotFound)

    def test_solver_available_recheck(self):
        mixin = GurobiSolverMixin()
        with unittest.mock.patch.object(
            GurobiSolverMixin, "_gurobipy_available", False
        ):
            self.assertEqual(mixin.solver_available(), SolverAvailability.NotFound)
        with unittest.mock.patch.object(GurobiSolverMixin, "_gurobipy_available", True):
            # Should first return the cached value
            self.assertEqual(mixin.solver_available(), SolverAvailability.NotFound)
            # Should now return the recheck value
            self.assertEqual(
                mixin.solver_available(recheck=True), SolverAvailability.Available
            )

    def test_full_license(self):
        mixin = GurobiSolverMixin()
        mock_gp = self.mocked_gurobipy("ok")
        with (
            unittest.mock.patch.object(GurobiSolverMixin, "_gurobipy_available", True),
            unittest.mock.patch(f"{self.MODULE_PATH}.gurobipy", mock_gp),
        ):
            with capture_output(capture_fd=True):
                output = mixin.license_available()
            self.assertEqual(output, LicenseAvailability.FullLicense)

    def test_limited_license(self):
        mixin = GurobiSolverMixin()
        mock_gp = self.mocked_gurobipy("too_large")
        with (
            unittest.mock.patch.object(GurobiSolverMixin, "_gurobipy_available", True),
            unittest.mock.patch(f"{self.MODULE_PATH}.gurobipy", mock_gp),
        ):
            with capture_output(capture_fd=True):
                output = mixin.license_available()
            self.assertEqual(output, LicenseAvailability.LimitedLicense)

    def test_no_license(self):
        mixin = GurobiSolverMixin()
        mock_gp = self.mocked_gurobipy("no_license")
        with (
            unittest.mock.patch.object(GurobiSolverMixin, "_gurobipy_available", True),
            unittest.mock.patch(f"{self.MODULE_PATH}.gurobipy", mock_gp),
        ):
            with capture_output(capture_fd=True):
                output = mixin.license_available()
            self.assertEqual(output, LicenseAvailability.NotAvailable)

    def test_license_timeout(self):
        mixin = GurobiSolverMixin()
        mock_gp = self.mocked_gurobipy("timeout")
        with (
            unittest.mock.patch.object(GurobiSolverMixin, "_gurobipy_available", True),
            unittest.mock.patch(f"{self.MODULE_PATH}.gurobipy", mock_gp),
        ):
            with capture_output(capture_fd=True):
                output = mixin.license_available(timeout=1)
            self.assertEqual(output, LicenseAvailability.Timeout)

    def test_license_available_recheck(self):
        mixin = GurobiSolverMixin()
        mock_gp_full = self.mocked_gurobipy("ok")
        with (
            unittest.mock.patch.object(GurobiSolverMixin, "_gurobipy_available", True),
            unittest.mock.patch(f"{self.MODULE_PATH}.gurobipy", mock_gp_full),
        ):
            with capture_output(capture_fd=True):
                output = mixin.license_available()
            self.assertEqual(output, LicenseAvailability.FullLicense)

        mock_gp_none = self.mocked_gurobipy("no_license")
        with (
            unittest.mock.patch.object(GurobiSolverMixin, "_gurobipy_available", True),
            unittest.mock.patch(f"{self.MODULE_PATH}.gurobipy", mock_gp_none),
        ):
            with capture_output(capture_fd=True):
                output = mixin.license_available()
            # Should return the cached value first because we didn't ask
            # for a recheck
            self.assertEqual(output, LicenseAvailability.FullLicense)
            with capture_output(capture_fd=True):
                output = mixin.license_available(recheck=True)
            # Should officially recheck
            self.assertEqual(output, LicenseAvailability.NotAvailable)

    def test_version(self):
        mixin = GurobiSolverMixin()
        mock_gp = self.mocked_gurobipy()
        with unittest.mock.patch(f"{self.MODULE_PATH}.gurobipy", mock_gp):
            self.assertEqual(mixin.version(), (12, 0, 1))
            # Verify that the cache works
            mock_gp.GRB.VERSION_MINOR = 99
            self.assertEqual(mixin.version(), (12, 0, 1))

    def test_acquire_license(self):
        mixin = GurobiSolverMixin()
        mock_gp = self.mocked_gurobipy()
        with unittest.mock.patch(f"{self.MODULE_PATH}.gurobipy", mock_gp):
            env = mixin.acquire_license()
            self.assertIs(env, mixin._gurobipy_env)
            self.assertIs(mixin.env(), env)

    def test_release_license(self):
        mock_env = unittest.mock.MagicMock()
        GurobiSolverMixin._gurobipy_env = mock_env
        GurobiSolverMixin._num_gurobipy_env_clients = 0

        GurobiSolverMixin.release_license()

        mock_env.close.assert_called_once()
        self.assertIsNone(GurobiSolverMixin._gurobipy_env)


@unittest.skipIf(not gurobipy_available, "The 'gurobipy' module is not available.")
class TestGurobiDirectInterface(unittest.TestCase):
    def test_solver_available_cache(self):
        opt = GurobiDirect()
        opt.solver_available()
        self.assertTrue(opt._available_cache)
        self.assertIsNotNone(opt._available_cache)

    def test_version_cache(self):
        opt = GurobiDirect()
        opt.version()
        self.assertIsNotNone(opt._version_cache[0])
        self.assertIsNotNone(opt._version_cache[1])
