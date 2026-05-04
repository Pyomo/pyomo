# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import logging
import re
import sys
import pyomo.common.unittest as unittest
from pyomo.common.download import DownloadFactory
from pyomo.common.tee import capture_output
from pyomo.common.log import LoggingIntercept

from pyomo.environ import SolverFactory
from pyomo.scripting.driver_help import help_solvers, help_transformations
from pyomo.scripting.pyomo_main import main


class skipper:
    def __init__(self, downloader):
        pass

    def __call__(self):
        pass

    def skip(self):
        return True


def ok(downloader):
    pass


def raise_exception(downloader):
    raise RuntimeError("downloader raised RuntimeError")


def raise_exit(downloader):
    sys.exit(1)


def _mock_kestrel(solvers, err=None):
    """
    Context-manager that replaces pyomo.neos.kestrel.kestrelAMPL with a
    stub whose ``solvers()`` method returns `solvers` and whose
    ``connect_error`` attribute is `err`.
    """
    import pyomo.neos.kestrel as _k

    class _MockedNEOS:
        def __init__(self, *a, **kw):
            self._solvers = solvers
            self.connect_error = err

        def solvers(self):
            return self._solvers

    return unittest.mock.patch.object(_k, "kestrelAMPL", _MockedNEOS)


class Test(unittest.TestCase):
    def test_pyomo_main_deprecation(self):
        with LoggingIntercept() as LOG:
            with unittest.pytest.raises(SystemExit) as e:
                main(args=['--solvers=glpk', 'foo.py'])
        self.assertIn("Running the 'pyomo' script with no subcommand", LOG.getvalue())

    def test_help_solvers(self):
        with capture_output() as OUT:
            help_solvers()
        OUT = OUT.getvalue()
        self.assertTrue(re.search('Pyomo Solvers and Solver Managers', OUT))
        self.assertTrue(re.search('Serial Solver', OUT))
        # Test known solvers and metasolver flags
        # ASL is a metasolver
        self.assertTrue(re.search(r'\n   \*asl ', OUT))
        # MindtPY is bundled with Pyomo so should always be available
        self.assertTrue(re.search(r'\n   \+mindtpy ', OUT))
        for solver in ('ipopt', 'cbc', 'glpk'):
            s = SolverFactory(solver)
            if s.available():
                self.assertTrue(
                    re.search(r"\n   \+%s " % solver, OUT),
                    "'   +%s' not found in help --solvers" % solver,
                )
            else:
                self.assertTrue(
                    re.search(r"\n    %s " % solver, OUT),
                    "'    %s' not found in help --solvers" % solver,
                )
        for solver in ('baron',):
            s = SolverFactory(solver)
            if s.license_is_valid():
                self.assertTrue(
                    re.search(r"\n   \+%s " % solver, OUT),
                    "'   +%s' not found in help --solvers" % solver,
                )
            elif s.available():
                self.assertTrue(
                    re.search(r"\n   \-%s " % solver, OUT),
                    "'   -%s' not found in help --solvers" % solver,
                )
            else:
                self.assertTrue(
                    re.search(r"\n    %s " % solver, OUT),
                    "'    %s' not found in help --solvers" % solver,
                )

    def test_help_solvers_neos_available(self):
        with _mock_kestrel(['ipoptAMPL', 'knitroAMPL']), capture_output() as OUT:
            help_solvers()
        txt = OUT.getvalue()
        self.assertIn("NEOS Solver Interfaces", txt)
        self.assertIn("ipopt", txt.lower())
        self.assertIn("knitro", txt.lower())
        self.assertIn("solver interfaces are available", txt)

    def test_help_solvers_neos_unavailable(self):
        """NEOS failure should show a message as to why"""
        import socket, xmlrpc.client

        cases = [
            (NotImplementedError("no ssl"), "SSL support"),
            (socket.timeout(), "timed out"),
            (socket.gaierror(-2, "not known"), "resolved"),
            (
                xmlrpc.client.ProtocolError(
                    "https://neos-server.org", 503, "Service Unavailable", {}
                ),
                "HTTP 503",
            ),
        ]

        for err, output in cases:
            with self.subTest(err=err), _mock_kestrel([], err), capture_output() as OUT:
                help_solvers()
            txt = OUT.getvalue()
            self.assertIn("NEOS Solver Interfaces", txt)
            self.assertIn("currently unavailable", txt)
            self.assertIn(output, txt)

    def test_help_transformations(self):
        with capture_output() as OUT:
            help_transformations()
        OUT = OUT.getvalue()
        self.assertTrue(re.search('Pyomo Model Transformations', OUT))
        self.assertTrue(re.search('core.relax_integer_vars', OUT))
        # test a transformation that we know is deprecated
        self.assertTrue(re.search(r'gdp.chull\s+\[DEPRECATED\]', OUT))

    def test_downloader(self):
        _orig = DownloadFactory._cls
        testers = {
            'skipper': skipper,
            'raise exception': raise_exception,
            'ok': ok,
            'raise exit': raise_exit,
        }

        try:
            DownloadFactory._cls = testers
            with (
                capture_output() as OUT,
                LoggingIntercept(module='pyomo', level=logging.INFO) as LOG,
            ):
                with self.assertRaisesRegex(ValueError, r"--retry must be >= 1"):
                    main(args=['download-extensions', '--retry', '0'])
        finally:
            DownloadFactory._cls = _orig

        self.assertEqual("", OUT.getvalue())
        self.assertEqual("", LOG.getvalue())

        try:
            DownloadFactory._cls = testers
            with (
                capture_output() as OUT,
                LoggingIntercept(module='pyomo', level=logging.INFO) as LOG,
            ):
                self.assertEqual(3, main(args=['download-extensions']))
        finally:
            DownloadFactory._cls = _orig

        self.assertEqual("", OUT.getvalue())
        self.assertEqual(
            """\
As of February 9, 2023, AMPL GSL can no longer be downloaded through \
download-extensions. Visit https://portal.ampl.com/ to download the \
AMPL GSL binaries.
RuntimeError: downloader raised RuntimeError
SystemExit: 1
Finished downloading Pyomo extensions.
The following extensions were downloaded:
    [SKIP]  skipper
    [FAIL]  raise exception
    [ OK ]  ok
    [FAIL]  raise exit
""",
            LOG.getvalue(),
        )

        try:
            DownloadFactory._cls = testers
            with (
                capture_output() as OUT,
                LoggingIntercept(module='pyomo', level=logging.INFO) as LOG,
            ):
                self.assertEqual(
                    3,
                    main(
                        args=[
                            'download-extensions',
                            '--retry',
                            '2',
                            '--retry-sleep',
                            '0',
                        ]
                    ),
                )
        finally:
            DownloadFactory._cls = _orig

        self.assertEqual("", OUT.getvalue())
        self.assertEqual(
            """\
As of February 9, 2023, AMPL GSL can no longer be downloaded through \
download-extensions. Visit https://portal.ampl.com/ to download the \
AMPL GSL binaries.
RuntimeError: downloader raised RuntimeError
Retrying download of 'raise exception' (attempt 2 of 2) in 0 seconds
RuntimeError: downloader raised RuntimeError
SystemExit: 1
Retrying download of 'raise exit' (attempt 2 of 2) in 0 seconds
SystemExit: 1
Finished downloading Pyomo extensions.
The following extensions were downloaded:
    [SKIP]  skipper
    [FAIL]  raise exception
    [ OK ]  ok
    [FAIL]  raise exit
""",
            LOG.getvalue(),
        )


if __name__ == "__main__":
    unittest.main()
