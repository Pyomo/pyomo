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
#
# Test NEOS solver interface
#
# Because the Kestrel tests require connections to the NEOS server, and
# that can take quite a while (5-20+ seconds), we will only run these
# tests as part of the nightly suite (i.e., by the CI system as part of
# PR / master tests)
#

import os
import json
import os.path
import tempfile

import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept

from pyomo.scripting.pyomo_main import main
from pyomo.scripting.util import cleanup
from pyomo.neos.kestrel import kestrelAMPL, xmlrpclib
import pyomo.neos

import pyomo.environ as pyo

from pyomo.common.fileutils import this_file_dir

currdir = this_file_dir()

neos_available = False
try:
    if kestrelAMPL().neos is not None:
        neos_available = True
except:
    pass

email_set = True
if os.environ.get('NEOS_EMAIL') is None:
    email_set = False


def _model(sense):
    # Goals of this model:
    # - linear
    # - solution has nonzero variable values (so they appear in the results)
    model = pyo.ConcreteModel()
    model.y = pyo.Var(bounds=(-10, 10), initialize=0.5)
    model.x = pyo.Var(bounds=(-5, 5), initialize=0.5)

    @model.ConstraintList()
    def c(m):
        yield m.y >= m.x - 2
        yield m.y >= -m.x
        yield m.y <= m.x
        yield m.y <= 2 - m.x

    model.obj = pyo.Objective(expr=model.y, sense=sense)
    return model


class _MockedServer:
    def __init__(self, *, ping_err=None, list_err=None, final_results=b"OK"):
        self._ping_err = ping_err
        self._list_err = list_err or (0, None)
        self._list_calls = 0
        self._final_results = final_results
        self.kill_args = None

    def ping(self):
        if self._ping_err:
            raise self._ping_err
        return "avail"

    def listSolversInCategory(self, cat):
        self._list_calls += 1
        if self._list_calls <= self._list_err[0]:
            raise self._list_err[1]
        return ["ipopt:AMPL", "cbc:AMPL", "baron:GAMS"]

    def killJob(self, job, pw):
        self.kill_args = (job, pw)
        return "killed"

    def getFinalResults(self, *_):
        return self._final_results


class TestNEOSInterface(unittest.TestCase):
    """
    This uses a mocked server to test basic functionality from kestrel;
    can run all the time, not necessary to have a real connection
    """

    def _uninit_kestrel(self):
        """Return an un-initialized kestrelAMPL"""
        return object.__new__(kestrelAMPL)

    def test_tempfile_env_set_and_unset(self):
        k = self._uninit_kestrel()

        # ampl_id unset  -> unknown
        os.environ.pop("ampl_id", None)
        self.assertTrue(kestrelAMPL.tempfile(k).endswith("atunknown.jobs"))

        # ampl_id present
        os.environ["ampl_id"] = "123"
        self.assertTrue(kestrelAMPL.tempfile(k).endswith("at123.jobs"))

    def test_kill_calls_remote(self):
        srv = _MockedServer()
        k = self._uninit_kestrel()
        k.neos = srv
        kestrelAMPL.kill(k, 42, "pw")
        self.assertEqual(srv.kill_args, (42, "pw"))

    def test_retrieve_string_and_binary(self):
        with tempfile.TemporaryDirectory() as td:
            stub = os.path.join(td, "foo")

            # string payload -> encoded
            srv = _MockedServer(final_results="text")
            k = self._uninit_kestrel()
            k.neos = srv
            kestrelAMPL.retrieve(k, stub, 1, "pw")
            with open(stub + ".sol", "rb") as fh:
                self.assertEqual(fh.read(), b"text")

            # binary payload
            payload = b"binary"
            srv = _MockedServer(final_results=xmlrpclib.Binary(payload))
            k.neos = srv
            kestrelAMPL.retrieve(k, stub, 1, "pw")
            with open(stub + ".sol", "rb") as fh:
                self.assertEqual(fh.read(), payload)

    def test_parsing_and_default(self):
        k = self._uninit_kestrel()

        # env absent
        os.environ.pop("kestrel_options", None)
        self.assertEqual(kestrelAMPL.getJobAndPassword(k), (0, ""))

        # env present
        os.environ["kestrel_options"] = "job=12 password=xyz"
        self.assertEqual(kestrelAMPL.getJobAndPassword(k), (12, "xyz"))

    def test_solvers_none_neos(self):
        k = self._uninit_kestrel()
        k.neos = None
        self.assertEqual(kestrelAMPL.getAvailableSolvers(k), [])

    def test_solvers_exception_returns_empty(self):
        srv = _MockedServer(list_err=(99, RuntimeError("boom")))
        k = self._uninit_kestrel()
        k.neos = srv
        self.assertEqual(kestrelAMPL.getAvailableSolvers(k), [])

    def test_solvers_filter_and_strip(self):
        srv = _MockedServer()
        k = self._uninit_kestrel()
        k.neos = srv
        self.assertEqual(kestrelAMPL.getAvailableSolvers(k), ["cbc", "ipopt"])


@unittest.pytest.mark.default
@unittest.pytest.mark.neos
@unittest.skipIf(not neos_available, "Cannot make connection to NEOS server")
@unittest.skipUnless(email_set, "NEOS_EMAIL not set")
class TestKestrel(unittest.TestCase):
    def test_doc(self):
        kestrel = kestrelAMPL()
        tmp = [tuple(name.split(':')) for name in kestrel.solvers()]
        amplsolvers = set(v[0].lower() for v in tmp if v[1] == 'AMPL')

        doc = pyomo.neos.doc
        dockeys = set(doc.keys())

        self.assertEqual(amplsolvers, dockeys)

        # gamssolvers = set(v[0].lower() for v in tmp if v[1]=='GAMS')
        # missing = gamssolvers - amplsolvers
        # self.assertEqual(len(missing) == 0)

    def test_connection_failed(self):
        try:
            orig_host = pyomo.neos.kestrel.NEOS.host
            pyomo.neos.kestrel.NEOS.host = 'neos-bogus-server.org'
            with LoggingIntercept() as LOG:
                kestrel = kestrelAMPL()
            self.assertIsNone(kestrel.neos)
            self.assertRegex(
                LOG.getvalue(), r"NEOS is temporarily unavailable:\n\t\(.+\)"
            )
        finally:
            pyomo.neos.kestrel.NEOS.host = orig_host

    def test_check_all_ampl_solvers(self):
        kestrel = kestrelAMPL()
        solvers = kestrel.getAvailableSolvers()
        for solver in solvers:
            name = solver.lower().replace('-', '')
            if not hasattr(RunAllNEOSSolvers, 'test_' + name):
                self.fail(f"RunAllNEOSSolvers missing test for '{solver}'")


class RunAllNEOSSolvers(object):
    def test_baron(self):
        self._run('baron')

    def test_bonmin(self):
        self._run('bonmin')

    def test_cbc(self):
        self._run('cbc')

    def test_conopt(self):
        self._run('conopt')

    def test_couenne(self):
        self._run('couenne')

    def test_cplex(self):
        self._run('cplex')

    def test_filmint(self):
        self._run('filmint')

    def test_filter(self):
        self._run('filter')

    def test_ipopt(self):
        self._run('ipopt')

    def test_knitro(self):
        self._run('knitro')

    # This solver only handles bound constrained variables
    def test_lbfgsb(self):
        self._run('l-bfgs-b', False)

    def test_lancelot(self):
        self._run('lancelot')

    def test_loqo(self):
        self._run('loqo')

    def test_minlp(self):
        self._run('minlp')

    def test_minos(self):
        self._run('minos')

    def test_minto(self):
        self._run('minto')

    def test_mosek(self):
        self._run('mosek')

    # [16 Jul 24]: Octeract is erroring.  We will disable the interface
    # (and testing) until we have time to resolve #3321
    # [20 Sep 24]: and appears to have been removed from NEOS
    # [24 Apr 25]: it appears to be there but causes timeouts
    # [29 Apr 25]: JK, it has been removed again
    # def test_octeract(self):
    #     pass
    #     self._run('octeract')

    def test_ooqp(self):
        if self.sense == pyo.maximize:
            # OOQP does not recognize maximization problems and
            # minimizes instead.
            with self.assertRaisesRegex(AssertionError, '.* != 1 within'):
                self._run('ooqp')
        else:
            self._run('ooqp')

    def test_path(self):
        # The simple tests aren't complementarity problems
        self.skipTest("The simple NEOS test is not a complementarity problem")

    def test_snopt(self):
        self._run('snopt')

    def test_raposa(self):
        self._run('raposa')

    def test_lgo(self):
        self._run('lgo')


class DirectDriver(object):
    def _run(self, opt, constrained=True):
        m = _model(self.sense)
        with pyo.SolverManagerFactory('neos') as solver_manager:
            results = solver_manager.solve(m, opt=opt)

        expected_y = {
            (pyo.minimize, True): -1,
            (pyo.maximize, True): 1,
            (pyo.minimize, False): -10,
            (pyo.maximize, False): 10,
        }[self.sense, constrained]

        self.assertEqual(results.solver[0].status, pyo.SolverStatus.ok)
        if constrained:
            # If the solver ignores constraints, x is degenerate
            self.assertAlmostEqual(pyo.value(m.x), 1, delta=1e-5)
        self.assertAlmostEqual(pyo.value(m.obj), expected_y, delta=1e-5)
        self.assertAlmostEqual(pyo.value(m.y), expected_y, delta=1e-5)


class PyomoCommandDriver(object):
    def _run(self, opt, constrained=True):
        expected_y = {
            (pyo.minimize, True): -1,
            (pyo.maximize, True): 1,
            (pyo.minimize, False): -10,
            (pyo.maximize, False): 10,
        }[self.sense, constrained]

        filename = (
            'model_min_lp.py' if self.sense == pyo.minimize else 'model_max_lp.py'
        )

        results = os.path.join(currdir, 'result.json')
        args = [
            'solve',
            os.path.join(currdir, filename),
            '--solver-manager=neos',
            '--solver=%s' % opt,
            '--logging=quiet',
            '--save-results=%s' % results,
            '--results-format=json',
            '-c',
        ]
        try:
            output = main(args)
            self.assertEqual(output.errorcode, 0)

            with open(results) as FILE:
                data = json.load(FILE)
        finally:
            cleanup()
            if os.path.exists(results):
                os.remove(results)

        self.assertEqual(data['Solver'][0]['Status'], 'ok')
        self.assertEqual(data['Solution'][1]['Status'], 'optimal')
        self.assertAlmostEqual(
            data['Solution'][1]['Objective']['obj']['Value'], expected_y, delta=1e-5
        )
        if constrained:
            # If the solver ignores constraints, x is degenerate
            self.assertAlmostEqual(
                data['Solution'][1]['Variable']['x']['Value'], 1, delta=1e-5
            )
        self.assertAlmostEqual(
            data['Solution'][1]['Variable']['y']['Value'], expected_y, delta=1e-5
        )


@unittest.pytest.mark.neos
@unittest.skipIf(not neos_available, "Cannot make connection to NEOS server")
@unittest.skipUnless(email_set, "NEOS_EMAIL not set")
class TestSolvers_direct_call_min(RunAllNEOSSolvers, DirectDriver, unittest.TestCase):
    sense = pyo.minimize


@unittest.pytest.mark.neos
@unittest.skipIf(not neos_available, "Cannot make connection to NEOS server")
@unittest.skipUnless(email_set, "NEOS_EMAIL not set")
class TestSolvers_direct_call_max(RunAllNEOSSolvers, DirectDriver, unittest.TestCase):
    sense = pyo.maximize


@unittest.pytest.mark.neos
@unittest.skipIf(not neos_available, "Cannot make connection to NEOS server")
@unittest.skipUnless(email_set, "NEOS_EMAIL not set")
class TestSolvers_pyomo_cmd_min(
    RunAllNEOSSolvers, PyomoCommandDriver, unittest.TestCase
):
    sense = pyo.minimize


@unittest.pytest.mark.default
@unittest.skipIf(not neos_available, "Cannot make connection to NEOS server")
@unittest.skipUnless(email_set, "NEOS_EMAIL not set")
class TestCBC_timeout_direct_call(DirectDriver, unittest.TestCase):
    sense = pyo.minimize

    @unittest.timeout(60, timeout_raises=unittest.SkipTest)
    def test_cbc_timeout(self):
        super()._run('cbc')


@unittest.pytest.mark.default
@unittest.skipIf(not neos_available, "Cannot make connection to NEOS server")
@unittest.skipUnless(email_set, "NEOS_EMAIL not set")
class TestCBC_timeout_pyomo_cmd(PyomoCommandDriver, unittest.TestCase):
    sense = pyo.minimize

    @unittest.timeout(60, timeout_raises=unittest.SkipTest)
    def test_cbc_timeout(self):
        super()._run('cbc')


if __name__ == "__main__":
    unittest.main()
