from random import random
import pyomo.common.unittest as unittest
from pyomo.common.multithread import *
from threading import Thread

class Dummy():
    """asdfg"""
    def __init__(self):
        self.id = 1
        self.rnd = random()
    def __str__(self):
        return 'asd'

class TestMultithreading(unittest.TestCase):
    def test_wrapper_docs(self):
        sut = MultiThreadWrapper(Dummy)
        self.assertEqual(sut.__class__.__doc__, Dummy.__doc__)
    def test_wrapper_field(self):
        sut = MultiThreadWrapper(Dummy)
        self.assertEqual(sut.id, 1)
    def test_independent_writes(self):
        sut = MultiThreadWrapper(Dummy)
        sut.id = 2
        def thread_func():
            self.assertEqual(sut.id, 1)
        t = Thread(target=thread_func)
        t.start()
        t.join()
    def test_independent_writes2(self):
        sut = MultiThreadWrapper(Dummy)
        def thread_func():
            sut.id = 2
        t = Thread(target=thread_func)
        t.start()
        t.join()
        self.assertEqual(sut.id, 1)
    def test_independent_del(self):
        sut = MultiThreadWrapper(Dummy)
        del sut.id
        def thread_func():
            self.assertEqual(sut.id, 1)
        t = Thread(target=thread_func)
        t.start()
        t.join()
    def test_independent_del2(self):
        sut = MultiThreadWrapper(Dummy)
        def thread_func():
            del sut.id
        t = Thread(target=thread_func)
        t.start()
        t.join()
        self.assertEqual(sut.id, 1)
    def test_special_methods(self):
        sut = MultiThreadWrapper(Dummy)
        self.assertTrue(set(Dummy().__dir__()).issubset(set(sut.__dir__())))
        self.assertEqual(str(sut), str(Dummy()))
    def test_main(self):
        sut = MultiThreadWrapperWithMain(Dummy)
        self.assertIs(sut.main_thread.rnd, sut.rnd)
        def thread_func():
            self.assertIsNot(sut.main_thread.rnd, sut.rnd)
        t = Thread(target=thread_func)
        t.start()
        t.join()
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
            for i in range(5):
                expr = 0
                for j in model.x:
                    if pyo.value(model.x[j]) < 0.5:
                        expr += model.x[j]
                    else:
                        expr += (1 - model.x[j])
                model.cuts.add( expr >= 1 )
                results = opt.solve(model)
                return results

        tp = ThreadPool(4)
        results = tp.map(test, [model.clone() for i in range(4)])
        tp.close()
        for result in results:
            self.assertEqual(result.solver.termination_condition, pyo.TerminationCondition.optimal)
