import threading
import pyomo.common.unittest as unittest
from pyomo.common.multithread import *
from threading import Thread
from pyomo.opt.base.solvers import check_available_solvers


class Dummy:
    """asdfg"""

    def __init__(self):
        self.number = 1
        self.rnd = threading.get_ident()

    def __str__(self):
        return 'asd'


class TestMultithreading(unittest.TestCase):
    def test_wrapper_docs(self):
        sut = MultiThreadWrapper(Dummy)
        self.assertEqual(sut.__class__.__doc__, Dummy.__doc__)

    def test_wrapper_field(self):
        sut = MultiThreadWrapper(Dummy)
        self.assertEqual(sut.number, 1)

    def test_independent_writes(self):
        sut = MultiThreadWrapper(Dummy)
        sut.number = 2

        def thread_func():
            self.assertEqual(sut.number, 1)

        t = Thread(target=thread_func)
        t.start()
        t.join()

    def test_independent_writes2(self):
        sut = MultiThreadWrapper(Dummy)

        def thread_func():
            sut.number = 2

        t = Thread(target=thread_func)
        t.start()
        t.join()
        self.assertEqual(sut.number, 1)

    def test_independent_del(self):
        sut = MultiThreadWrapper(Dummy)
        del sut.number

        def thread_func():
            self.assertEqual(sut.number, 1)

        t = Thread(target=thread_func)
        t.start()
        t.join()

    def test_independent_del2(self):
        sut = MultiThreadWrapper(Dummy)

        def thread_func():
            del sut.number

        t = Thread(target=thread_func)
        t.start()
        t.join()
        self.assertEqual(sut.number, 1)

    def test_special_methods(self):
        sut = MultiThreadWrapper(Dummy)
        self.assertTrue(set(Dummy().__dir__()).issubset(set(sut.__dir__())))
        self.assertEqual(str(sut), str(Dummy()))

    def test_main(self):
        sut = MultiThreadWrapperWithMain(Dummy)
        self.assertIs(sut.main_thread.rnd, sut.rnd)
        sut.number = 5

        def thread_func():
            self.assertEqual(sut.number, 1)
            self.assertIsNot(sut.main_thread.rnd, sut.rnd)
            del sut.number

        t = Thread(target=thread_func)
        t.start()
        t.join()
        self.assertEqual(sut.number, 5)

    @unittest.skipIf(
        len(check_available_solvers('glpk')) < 1, "glpk solver not available"
    )
    def test_solve(self):
        # Based on the minimal example in https://github.com/Pyomo/pyomo/issues/2475
        import pyomo.environ as pyo
        from pyomo.opt import SolverFactory
        from multiprocessing.dummy import Pool as ThreadPool

        model = pyo.ConcreteModel()
        model.nVars = pyo.Param(initialize=4)
        model.N = pyo.RangeSet(model.nVars)
        model.x = pyo.Var(model.N, within=pyo.Binary)
        model.obj = pyo.Objective(expr=pyo.summation(model.x))
        model.cuts = pyo.ConstraintList()

        def test(model):
            opt = SolverFactory('glpk')
            opt.solve(model)

            # Iterate, adding a cut to exclude the previously found solution
            for _ in range(5):
                expr = 0
                for j in model.x:
                    if pyo.value(model.x[j]) < 0.5:
                        expr += model.x[j]
                    else:
                        expr += 1 - model.x[j]
                model.cuts.add(expr >= 1)
                results = opt.solve(model)
            return results, [v for v in model.x]

        tp = ThreadPool(4)
        results = tp.map(test, [model.clone() for i in range(4)])
        tp.close()
        for result, _ in results:
            self.assertEqual(
                result.solver.termination_condition, pyo.TerminationCondition.optimal
            )
        results = list(results)
        for _, values in results[1:]:
            self.assertEqual(values, results[0][1])
