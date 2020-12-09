#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#
# Unit Tests for pyomo.opt.parallel (using the COLIN optimizers)
#

import os
from os.path import abspath, dirname
pyomodir = dirname(dirname(dirname(dirname(abspath(__file__)))))
pyomodir += os.sep
currdir = dirname(abspath(__file__))+os.sep

import pyutilib.th as unittest

import pyomo.opt
import pyomo.opt.blackbox
from pyomo.opt.parallel.manager import ActionManagerError
from pyomo.common.tempfiles import TempfileManager

old_tempdir = TempfileManager.tempdir


class TestProblem1(pyomo.opt.blackbox.MixedIntOptProblem):

    def __init__(self):
        pyomo.opt.blackbox.MixedIntOptProblem.__init__(self)
        self.real_lower=[0.0, -1.0, 1.0, None]
        self.real_upper=[None, 0.0, 2.0, -1.0]
        self.nreal=4

    def function_value(self, point):
        self.validate(point)
        return point.reals[0] - point.reals[1] + (point.reals[2]-1.5)**2 + (point.reals[3]+2)**4


class TestSolverManager(pyomo.opt.parallel.AsynchronousSolverManager):

    def __init__(self, **kwds):
        kwds['type'] = 'smtest_type'
        kwds['doc'] = 'TestASM Documentation'
        pyomo.opt.parallel.AsynchronousSolverManager.__init__(self,**kwds)

    def enabled(self):
        return False


class SolverManager_DelayedSerial(pyomo.opt.parallel.AsynchronousSolverManager):

    def clear(self):
        """
        Clear manager state
        """
        pyomo.opt.parallel.AsynchronousSolverManager.clear(self)
        self.delay=5
        self._ah_list = []
        self._opt = None
        self._my_results = {}
        self._ctr = 1
        self._force_error = 0

    def _perform_queue(self, ah, *args, **kwds):
        """
        Perform the queue operation.  This method returns the ActionHandle,
        and the ActionHandle status indicates whether the queue was successful.
        """
        self._opt = kwds.pop('solver', kwds.pop('opt', None))
        if self._opt is None:
            raise ActionManagerError(
                "No solver passed to %s, use keyword option 'solver'"
                % (type(self).__name__) )
        self._my_results[ah.id] = self._opt.solve(*args)
        self._ah_list.append(ah)
        return ah

    def _perform_wait_any(self):
        """
        Perform the wait_any operation.  This method returns an
        ActionHandle with the results of waiting.  If None is returned
        then the ActionManager assumes that it can call this method again.
        Note that an ActionHandle can be returned with a dummy value,
        to indicate an error.
        """
        if self._force_error == 0:
            self._ctr += 1
            if self._ctr % self.delay != 0:
                return None
            if len(self._ah_list) > 0:
                ah = self._ah_list.pop()
                ah.status = pyomo.opt.parallel.manager.ActionStatus.done
                self.results[ah.id] = self._my_results[ah.id]
                return ah
            return pyomo.opt.parallel.manager.ActionHandle(error=True, explanation="No queued evaluations available in the 'local' solver manager, which only executes solvers synchronously")
        elif self._force_error == 1:
            #
            # Wait Any returns an ActionHandle that indicates an error
            #
            return pyomo.opt.parallel.manager.ActionHandle(error=True, explanation="Forced failure")
        elif self._force_error == 2:
            #
            # Wait Any returns the correct ActionHandle, but no results are
            # available.
            #
            return self._ah_list.pop()


class Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import pyomo.environ

    def setUp(self):
        self.do_setup(False)
        TempfileManager.tempdir = currdir

    def do_setup(self,flag):
        TempfileManager.tempdir = currdir
        self.ps = pyomo.opt.SolverFactory('ps')

    def tearDown(self):
        pyomo.opt.SolverManagerFactory.unregister('smtest')
        TempfileManager.clear_tempfiles()
        TempfileManager.tempdir = old_tempdir

    def test_solve1(self):
        """ Test PatternSearch - TestProblem1 """
        problem=TestProblem1()
        self.ps.problem=problem
        self.ps.initial_point = [1.0, -0.5, 2.0, -1.0]
        self.ps.reset()
        results = self.ps.solve(logfile=currdir+"test_solve1.log")
        results.write(filename=currdir+"test_solve1.txt", times=False, format='json')
        self.assertMatchesJsonBaseline(currdir+"test_solve1.txt", currdir+"test1_ps.txt")
        if os.path.exists(currdir+"test_solve1.log"):
            os.remove(currdir+"test_solve1.log")

    def test_serial1(self):
        """ Test Serial EvalManager - TestProblem1 """
        problem=TestProblem1()
        self.ps.problem=problem
        self.ps.initial_point = [1.0, -0.5, 2.0, -1.0]
        self.ps.reset()
        mngr = pyomo.opt.parallel.SolverManagerFactory("serial")
        results = mngr.solve(opt=self.ps, logfile=currdir+"test_solve2.log")
        results.write(filename=currdir+"test_solve2.txt", times=False, format='json')
        self.assertMatchesJsonBaseline(currdir+"test_solve2.txt", currdir+"test1_ps.txt")
        if os.path.exists(currdir+"test_solve2.log"):
            os.remove(currdir+"test_solve2.log")

    def test_serial_error1(self):
        """ Test Serial SolverManager - Error with no optimizer"""
        problem=TestProblem1()
        self.ps.problem=problem
        self.ps.initial_point = [1.0, -0.5, 2.0, -1.0]
        self.ps.reset()
        mngr = pyomo.opt.parallel.SolverManagerFactory("serial")
        try:
            results = mngr.solve(logfile=currdir+"test_solve3.log")
            self.fail("Expected error")
        except pyomo.opt.parallel.manager.ActionManagerError:
            pass

    def test_serial_error2(self):
        """ Test Serial SolverManager - Error with no queue solves"""
        problem=TestProblem1()
        self.ps.problem=problem
        self.ps.initial_point = [1.0, -0.5, 2.0, -1.0]
        self.ps.reset()
        mngr = pyomo.opt.parallel.SolverManagerFactory("serial")
        results = mngr.solve(opt=self.ps, logfile=currdir+"test_solve3.log")
        if mngr.wait_any() != pyomo.opt.parallel.manager.FailedActionHandle:
            self.fail("Expected a failed action")
        if os.path.exists(currdir+"test_solve2.log"):
            os.remove(currdir+"test_solve2.log")

    def test_solver_manager_factory(self):
        """
        Testing the pyomo.opt solver factory
        """
        pyomo.opt.parallel.SolverManagerFactory.register('smtest')(TestSolverManager)
        ans = sorted(pyomo.opt.SolverManagerFactory)
        tmp = ["smtest"]
        self.assertTrue(set(tmp) <= set(ans))

    def test_solver_manager_instance(self):
        """
        Testing that we get a specific solver instance
        """
        pyomo.opt.parallel.SolverManagerFactory.register('smtest')(TestSolverManager)
        ans = pyomo.opt.SolverManagerFactory("none")
        self.assertEqual(ans, None)
        ans = pyomo.opt.SolverManagerFactory("smtest")
        self.assertEqual(type(ans), TestSolverManager)
        #ans = pyomo.opt.SolverManagerFactory("smtest", "mymock")
        #self.assertEqual(type(ans), TestSolverManager)
        #self.assertEqual(ans.name,  "mymock")

    def test_solver_manager_registration(self):
        """
        Testing methods in the solverwriter factory registration process
        """
        self.assertTrue(not 'smtest' in pyomo.opt.SolverManagerFactory)
        pyomo.opt.parallel.SolverManagerFactory.register('smtest')(TestSolverManager)
        self.assertTrue('smtest' in pyomo.opt.SolverManagerFactory)

    def test_delayed_serial1(self):
        """
        Use a solver manager that delays the evaluation of responses, and thus allows a mock testing of the wait*() methods.
        """
        problem=TestProblem1()
        self.ps.problem=problem
        self.ps.initial_point = [1.0, -0.5, 2.0, -1.0]
        self.ps.reset()
        mngr = SolverManager_DelayedSerial()
        results = mngr.solve(opt=self.ps, logfile=currdir+"test_solve4.log")
        results.write(filename=currdir+"test_solve4.txt", times=False, format='json')
        self.assertMatchesJsonBaseline(currdir+"test_solve4.txt", currdir+
"test1_ps.txt")
        if os.path.exists(currdir+"test_solve4.log"):
            os.remove(currdir+"test_solve4.log")

    def test_delayed_serial2(self):
        """
        Use a solver manager that delays the evaluation of responses, and _perform_wait_any() returns a failed action handle.
        """
        problem=TestProblem1()
        self.ps.problem=problem
        self.ps.initial_point = [1.0, -0.5, 2.0, -1.0]
        self.ps.reset()
        mngr = SolverManager_DelayedSerial()
        mngr._force_error = 1
        try:
            results = mngr.solve(opt=self.ps, logfile=currdir+"test_solve5.log")
            self.fail("Expected error")
        except pyomo.opt.parallel.manager.ActionManagerError:
            pass
        if os.path.exists(currdir+"test_solve5.log"):
            os.remove(currdir+"test_solve5.log")

    def test_delayed_serial3(self):
        """
        Use a solver manager that delays the evaluation of responses, and _perform_wait_any() returns a action handle, but no results are available.
        """
        problem=TestProblem1()
        self.ps.problem=problem
        self.ps.initial_point = [1.0, -0.5, 2.0, -1.0]
        self.ps.reset()
        mngr = SolverManager_DelayedSerial()
        mngr._force_error = 2
        try:
            results = mngr.solve(opt=self.ps, logfile=currdir+"test_solve6.log")
            self.fail("Expected error")
        except pyomo.opt.parallel.manager.ActionManagerError:
            pass
        if os.path.exists(currdir+"test_solve6.log"):
            os.remove(currdir+"test_solve6.log")

    def test_delayed_serial4(self):
        """
        Use a solver manager that delays the evaluation of responses, and verify that queue-ing multiple solves works.
        """
        problem=TestProblem1()
        self.ps.problem=problem
        self.ps.initial_point = [1.0, -0.5, 2.0, -1.0]
        self.ps.reset()
        mngr = SolverManager_DelayedSerial()
        ah_a = mngr.queue(opt=self.ps, logfile=currdir+"test_solve7a.log")
        ah_b = mngr.queue(opt=self.ps, logfile=currdir+"test_solve7b.log")
        ah_c = mngr.queue(opt=self.ps, logfile=currdir+"test_solve7c.log")

        mngr.wait_all()

        self.assertEqual(ah_c.status, pyomo.opt.parallel.manager.ActionStatus.done)
        if os.path.exists(currdir+"test_solve7a.log"):
            os.remove(currdir+"test_solve7a.log")
        if os.path.exists(currdir+"test_solve7b.log"):
            os.remove(currdir+"test_solve7b.log")
        if os.path.exists(currdir+"test_solve7c.log"):
            os.remove(currdir+"test_solve7c.log")

    def test_delayed_serial5(self):
        """
        Use a solver manager that delays the evaluation of responses, and verify that queue-ing multiple solves works.
        """
        problem=TestProblem1()
        self.ps.problem=problem
        self.ps.initial_point = [1.0, -0.5, 2.0, -1.0]
        self.ps.reset()
        mngr = SolverManager_DelayedSerial()
        ah_a = mngr.queue(opt=self.ps, logfile=currdir+"test_solve8a.log")
        ah_b = mngr.queue(opt=self.ps, logfile=currdir+"test_solve8b.log")
        ah_c = mngr.queue(opt=self.ps, logfile=currdir+"test_solve8c.log")

        mngr.wait_all(ah_b)

        self.assertEqual(ah_b.status, pyomo.opt.parallel.manager.ActionStatus.done)
        self.assertEqual(ah_a.status, pyomo.opt.parallel.manager.ActionStatus.queued)
        if os.path.exists(currdir+"test_solve8a.log"):
            os.remove(currdir+"test_solve8a.log")
        if os.path.exists(currdir+"test_solve8b.log"):
            os.remove(currdir+"test_solve8b.log")
        if os.path.exists(currdir+"test_solve8c.log"):
            os.remove(currdir+"test_solve8c.log")

    def test_delayed_serial6(self):
        """
        Use a solver manager that delays the evaluation of responses, and verify that queue-ing multiple solves works.
        """
        problem=TestProblem1()
        self.ps.problem=problem
        self.ps.initial_point = [1.0, -0.5, 2.0, -1.0]
        self.ps.reset()
        mngr = SolverManager_DelayedSerial()
        ah_a = mngr.queue(opt=self.ps, logfile=currdir+"test_solve8a.log")
        ah_b = mngr.queue(opt=self.ps, logfile=currdir+"test_solve8b.log")
        ah_c = mngr.queue(opt=self.ps, logfile=currdir+"test_solve8c.log")

        self.assertEqual( mngr.num_queued(), 3)
        mngr.wait_all( [ah_b] )

        self.assertEqual(mngr.get_status(ah_b), pyomo.opt.parallel.manager.ActionStatus.done)
        self.assertEqual(mngr.get_status(ah_a), pyomo.opt.parallel.manager.ActionStatus.queued)

        if os.path.exists(currdir+"test_solve8a.log"):
            os.remove(currdir+"test_solve8a.log")
        if os.path.exists(currdir+"test_solve8b.log"):
            os.remove(currdir+"test_solve8b.log")
        if os.path.exists(currdir+"test_solve8c.log"):
            os.remove(currdir+"test_solve8c.log")

if __name__ == "__main__":
    unittest.main()
