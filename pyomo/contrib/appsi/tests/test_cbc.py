# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

from pyomo.common import unittest
from pyomo.contrib.appsi.solvers import Cbc


@unittest.pytest.mark.solver('cbc')
class TestCbcInterface(unittest.TestCase):
    def test_version(self):
        opt = Cbc()
        if opt.config.executable.path() is None:
            self.skipTest("The 'cbc' command is not available")

        version = opt.version()

        self.assertIsInstance(version, tuple)
        self.assertTrue(version)
        for component in version:
            self.assertTrue(isinstance(component, int) or component == float('inf'))

    def _assert_version_output(self, output, expected):
        # Replace only the executable's output; exercise the real version parser.
        # These checks do not require an installed CBC executable.
        opt = Cbc()
        with unittest.mock.patch(
            'pyomo.contrib.appsi.solvers.cbc.subprocess.run'
        ) as run:
            run.return_value.stdout = output
            self.assertEqual(opt.version(), expected)
            run.assert_called_once()

    def test_version_legacy_welcome_banner(self):
        self._assert_version_output(
            'Welcome to the CBC MILP Solver\nVersion: 2.10.10\n'
            'Build Date: Apr 19 2024\n',
            (2, 10, 10),
        )

    def test_version_banner_after_blank_lines(self):
        self._assert_version_output('\n\nVersion: 2.9.9\n', (2, 9, 9))

    def test_version_cbc_numeric_banner(self):
        self._assert_version_output(
            'CBC 2.10.12 (git:146ce89)\n  args: -stop\n', (2, 10, 12)
        )

    def test_version_cbc_version_banner(self):
        self._assert_version_output('Cbc version 2.10.12\n', (2, 10, 12))

    def test_version_numeric_devel_suffix_banner(self):
        self._assert_version_output('Version: 2.10.12-devel\n', (2, 10, 12))

    def test_version_cbc_devel_banner(self):
        self._assert_version_output(
            'CBC devel (git:146ce89)\n  args: -stop\n'
            'Total time (Wallclock seconds): 0.000225\n',
            (float('inf'),) * 3,
        )

    def test_version_devel_banner(self):
        self._assert_version_output('Version: devel\n', (float('inf'),) * 3)

    def test_version_trunk_banner(self):
        self._assert_version_output('Version: Trunk\n', (float('inf'),) * 3)

    def test_unrecognized_version_output(self):
        opt = Cbc()
        for output in (
            '',
            'Version: unknown\nTotal time (Wallclock seconds): 0.000225\n',
            'CBC unknown (git:1234567)\n  args: -stop\n',
            'Unable to load library version 1.2.3\n',
        ):
            with self.subTest(output=output):
                with unittest.mock.patch(
                    'pyomo.contrib.appsi.solvers.cbc.subprocess.run'
                ) as run:
                    run.return_value.stdout = output
                    with self.assertRaisesRegex(
                        RuntimeError, 'Could not determine CBC version'
                    ):
                        opt.version()
