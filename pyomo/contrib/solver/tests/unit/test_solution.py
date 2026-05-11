# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

from pyomo.common import unittest
from pyomo.common.collections import ComponentMap
from pyomo.contrib.solver.common.solution_loader import (
    SolutionLoader,
    PersistentSolutionLoader,
    NoSolutionSolutionLoader,
)
from pyomo.contrib.solver.common.util import NoSolutionError

import pyomo.environ as pyo


class SolutionLoaderTester(SolutionLoader):
    def __init__(self):
        self._soln = 0
        self._pyomo_model = m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.c = pyo.Constraint(expr=m.x == 0)
        m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        m.b = pyo.Block()
        m.b.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        m.b.b = pyo.Block()
        m.b.b.rc = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        m.c = pyo.Block()
        m.c.rc = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    def reset(self):
        m = self._pyomo_model
        m.x.value = -1
        for i in (m.x, m.c):
            m.dual[i] = -1
            m.b.dual[i] = -1
            m.b.b.rc[i] = -1
            m.c.rc[i] = -1

    def _set_solution_id(self, solution_id):
        prev = self._soln
        self._soln = solution_id
        return prev

    def get_number_of_solutions(self):
        return 3

    def get_solution_ids(self):
        return list(range(self.get_number_of_solutions()))

    def get_vars(self, vars_to_load=None):
        return ComponentMap([(self._pyomo_model.x, self._soln)])

    def get_duals(self, cons_to_load=None):
        return {self._pyomo_model.c: 10 * self._soln}

    def get_reduced_costs(self, vars_to_load=None):
        return ComponentMap([(self._pyomo_model.x, 100 * self._soln)])


class TestSolutionLoader(unittest.TestCase):
    def test_member_list(self):
        expected_list = [
            'load_vars',
            'get_vars',
            'get_duals',
            'get_reduced_costs',
            'load_import_suffixes',
            'get_number_of_solutions',
            'get_solution_ids',
            'load_solution',
            'solution',
        ]
        method_list = [
            method for method in dir(SolutionLoader) if method.startswith('_') is False
        ]
        self.assertEqual(sorted(expected_list), sorted(method_list))

    def test_solution_loader_base(self):
        loader = SolutionLoader()
        with self.assertRaisesRegex(
            NotImplementedError,
            "SolutionLoader class failed to implement required method "
            "'get_number_of_solutions'.",
        ):
            loader.get_number_of_solutions()
        with self.assertRaisesRegex(
            NotImplementedError,
            "SolutionLoader class failed to implement required method 'get_vars'.",
        ):
            loader.get_vars()
        with self.assertRaisesRegex(
            NotImplementedError,
            "SolutionLoader class failed to implement required method 'get_duals'.",
        ):
            loader.get_duals()
        with self.assertRaisesRegex(
            NotImplementedError,
            "SolutionLoader class failed to implement required method "
            "'get_reduced_costs'.",
        ):
            loader.get_reduced_costs()

    def test_set_invalid_solutionid(self):
        # The base implementation supports solvers that only return a
        # single solution
        class MockSolutionLoader(SolutionLoader):
            def __init__(self, n):
                self.n = n
                self._pyomo_model = m = pyo.ConcreteModel()
                m.x = pyo.Var()

            def get_number_of_solutions(self):
                return self.n

            def get_vars(self, vars_to_load=None):
                return cm(self._pyomo_model.x, 1)

        loader = MockSolutionLoader(1)
        m = loader._pyomo_model
        self.assertEqual(loader.get_number_of_solutions(), 1)
        self.assertEqual(loader.get_solution_ids(), [None])
        self.assertEqual(loader.get_vars(), cm(m.x, 1))
        self.assertEqual(loader.solution(None).get_vars(), cm(m.x, 1))
        with self.assertRaisesRegex(
            ValueError, "MockSolutionLoader does not support multiple solutions"
        ):
            loader.solution(1).get_vars()


class TestPersistentSolutionLoader(unittest.TestCase):
    def test_member_list(self):
        expected_list = [
            'load_vars',
            'get_vars',
            'get_duals',
            'get_reduced_costs',
            'invalidate',
            'load_import_suffixes',
            'get_number_of_solutions',
            'get_solution_ids',
            'load_solution',
            'solution',
        ]
        method_list = [
            method
            for method in dir(PersistentSolutionLoader)
            if method.startswith('_') is False
        ]
        self.assertEqual(sorted(expected_list), sorted(method_list))

    def test_default_initialization(self):
        # Realistically, a solver object should be passed into this.
        # However, it works with a string. It'll just error loudly if you
        # try to run get_primals, etc.
        self.instance = PersistentSolutionLoader('ipopt', None)
        self.assertTrue(self.instance._valid)
        self.assertEqual(self.instance._solver, 'ipopt')

    def test_invalid(self):
        self.instance = PersistentSolutionLoader('ipopt', None)
        self.instance.invalidate()
        with self.assertRaises(RuntimeError):
            self.instance.get_vars()


def cm(*args):
    if not args:
        return ComponentMap()
    return ComponentMap([args])


class TestSolutionLoaderView(unittest.TestCase):

    def test_get_number_of_solutions(self):
        loader = SolutionLoaderTester()
        m = loader._pyomo_model

        self.assertEqual(loader.get_number_of_solutions(), 3)
        self.assertEqual(loader.solution(1).get_number_of_solutions(), 3)
        with loader.solution(2) as soln:
            self.assertEqual(soln.get_number_of_solutions(), 3)

    def test_get_solution_ids(self):
        loader = SolutionLoaderTester()
        m = loader._pyomo_model

        self.assertEqual(loader.get_solution_ids(), [0, 1, 2])
        self.assertEqual(loader.solution(1).get_solution_ids(), [0, 1, 2])
        with loader.solution(2) as soln:
            self.assertEqual(soln.get_solution_ids(), [0, 1, 2])

    def test_get_vars(self):
        loader = SolutionLoaderTester()
        m = loader._pyomo_model
        ref = ComponentMap([(m.x, -1), (m.c, -1)])

        loader.reset()
        v = loader.get_vars()
        self.assertEqual(v, cm(m.x, 0))
        self.assertEqual(m.x.value, -1)
        self.assertEqual(m.dual, ref)
        self.assertEqual(m.b.dual, ref)
        self.assertEqual(m.b.b.rc, ref)
        self.assertEqual(m.c.rc, ref)

        v = loader.solution(1).get_vars()
        self.assertEqual(v, cm(m.x, 1))
        self.assertEqual(m.x.value, -1)
        self.assertEqual(m.dual, ref)
        self.assertEqual(m.b.dual, ref)
        self.assertEqual(m.b.b.rc, ref)
        self.assertEqual(m.c.rc, ref)

        with loader.solution(2) as soln:
            v = soln.get_vars()
        self.assertEqual(v, cm(m.x, 2))
        self.assertEqual(m.x.value, -1)
        self.assertEqual(m.dual, ref)
        self.assertEqual(m.b.dual, ref)
        self.assertEqual(m.b.b.rc, ref)
        self.assertEqual(m.c.rc, ref)

    def test_get_duals(self):
        loader = SolutionLoaderTester()
        m = loader._pyomo_model
        ref = ComponentMap([(m.x, -1), (m.c, -1)])

        loader.reset()
        d = loader.get_duals()
        self.assertEqual(d, cm(m.c, 0))
        self.assertEqual(m.x.value, -1)
        self.assertEqual(m.dual, ref)
        self.assertEqual(m.b.dual, ref)
        self.assertEqual(m.b.b.rc, ref)
        self.assertEqual(m.c.rc, ref)

        d = loader.solution(1).get_duals()
        self.assertEqual(d, cm(m.c, 10))
        self.assertEqual(m.x.value, -1)
        self.assertEqual(m.dual, ref)
        self.assertEqual(m.b.dual, ref)
        self.assertEqual(m.b.b.rc, ref)
        self.assertEqual(m.c.rc, ref)

        with loader.solution(2) as soln:
            d = soln.get_duals()
        self.assertEqual(d, cm(m.c, 20))
        self.assertEqual(m.x.value, -1)
        self.assertEqual(m.dual, ref)
        self.assertEqual(m.b.dual, ref)
        self.assertEqual(m.b.b.rc, ref)
        self.assertEqual(m.c.rc, ref)

    def test_get_reduced_costs(self):
        loader = SolutionLoaderTester()
        m = loader._pyomo_model
        ref = ComponentMap([(m.x, -1), (m.c, -1)])

        loader.reset()
        rc = loader.get_reduced_costs()
        self.assertEqual(rc, cm(m.x, 0))
        self.assertEqual(m.x.value, -1)
        self.assertEqual(m.dual, ref)
        self.assertEqual(m.b.dual, ref)
        self.assertEqual(m.b.b.rc, ref)
        self.assertEqual(m.c.rc, ref)

        rc = loader.solution(1).get_reduced_costs()
        self.assertEqual(rc, cm(m.x, 100))
        self.assertEqual(m.x.value, -1)
        self.assertEqual(m.dual, ref)
        self.assertEqual(m.b.dual, ref)
        self.assertEqual(m.b.b.rc, ref)
        self.assertEqual(m.c.rc, ref)

        with loader.solution(2) as soln:
            rc = soln.get_reduced_costs()
        self.assertEqual(rc, cm(m.x, 200))
        self.assertEqual(m.x.value, -1)
        self.assertEqual(m.dual, ref)
        self.assertEqual(m.b.dual, ref)
        self.assertEqual(m.b.b.rc, ref)
        self.assertEqual(m.c.rc, ref)

    def test_load_solution(self):
        loader = SolutionLoaderTester()
        m = loader._pyomo_model

        loader.reset()
        loader.load_solution()
        self.assertEqual(m.x.value, 0)
        self.assertEqual(m.dual, cm(m.c, 0))
        self.assertEqual(m.b.dual, cm())
        self.assertEqual(m.b.b.rc, cm())
        self.assertEqual(m.c.rc, cm(m.x, 0))

        loader.reset()
        loader.solution(1).load_solution()
        self.assertEqual(m.x.value, 1)
        self.assertEqual(m.dual, cm(m.c, 10))
        self.assertEqual(m.b.dual, cm())
        self.assertEqual(m.b.b.rc, cm())
        self.assertEqual(m.c.rc, cm(m.x, 100))

        loader.reset()
        with loader.solution(2) as soln:
            loader.solution(1).load_solution()
            self.assertEqual(m.x.value, 1)
            self.assertEqual(m.dual, cm(m.c, 10))
            self.assertEqual(m.b.dual, cm())
            self.assertEqual(m.b.b.rc, cm())
            self.assertEqual(m.c.rc, cm(m.x, 100))
            soln.load_solution()
        self.assertEqual(m.x.value, 2)
        self.assertEqual(m.dual, cm(m.c, 20))
        self.assertEqual(m.b.dual, cm())
        self.assertEqual(m.b.b.rc, cm())
        self.assertEqual(m.c.rc, cm(m.x, 200))

    def test_load_import_suffixes(self):
        loader = SolutionLoaderTester()
        m = loader._pyomo_model

        loader.reset()
        loader.load_import_suffixes()
        self.assertEqual(m.x.value, -1)
        self.assertEqual(m.dual, cm(m.c, 0))
        self.assertEqual(m.b.dual, cm())
        self.assertEqual(m.b.b.rc, cm())
        self.assertEqual(m.c.rc, cm(m.x, 0))

        loader.reset()
        loader.solution(1).load_import_suffixes()
        self.assertEqual(m.x.value, -1)
        self.assertEqual(m.dual, cm(m.c, 10))
        self.assertEqual(m.b.dual, cm())
        self.assertEqual(m.b.b.rc, cm())
        self.assertEqual(m.c.rc, cm(m.x, 100))

        loader.reset()
        with loader.solution(2) as soln:
            loader.solution(1).load_import_suffixes()
            self.assertEqual(m.x.value, -1)
            self.assertEqual(m.dual, cm(m.c, 10))
            self.assertEqual(m.b.dual, cm())
            self.assertEqual(m.b.b.rc, cm())
            self.assertEqual(m.c.rc, cm(m.x, 100))
            soln.load_import_suffixes()
        self.assertEqual(m.x.value, -1)
        self.assertEqual(m.dual, cm(m.c, 20))
        self.assertEqual(m.b.dual, cm())
        self.assertEqual(m.b.b.rc, cm())
        self.assertEqual(m.c.rc, cm(m.x, 200))

    def test_load_vars(self):
        loader = SolutionLoaderTester()
        m = loader._pyomo_model
        ref = ComponentMap([(m.x, -1), (m.c, -1)])

        loader.reset()
        loader.load_vars()
        self.assertEqual(m.x.value, 0)
        self.assertEqual(m.dual, ref)
        self.assertEqual(m.b.dual, ref)
        self.assertEqual(m.b.b.rc, ref)
        self.assertEqual(m.c.rc, ref)

        loader.reset()
        loader.solution(1).load_vars()
        self.assertEqual(m.x.value, 1)
        self.assertEqual(m.dual, ref)
        self.assertEqual(m.b.dual, ref)
        self.assertEqual(m.b.b.rc, ref)
        self.assertEqual(m.c.rc, ref)

        loader.reset()
        with loader.solution(2) as soln:
            loader.solution(1).load_vars()
            self.assertEqual(m.x.value, 1)
            self.assertEqual(m.dual, ref)
            self.assertEqual(m.b.dual, ref)
            self.assertEqual(m.b.b.rc, ref)
            self.assertEqual(m.c.rc, ref)
            soln.load_vars()
        self.assertEqual(m.x.value, 2)
        self.assertEqual(m.dual, ref)
        self.assertEqual(m.b.dual, ref)
        self.assertEqual(m.b.b.rc, ref)
        self.assertEqual(m.c.rc, ref)


class TestNoSolutionSolutionLoader(unittest.TestCase):
    def test_core_API(self):
        model = pyo.ConcreteModel()
        loader = NoSolutionSolutionLoader(model, "error message")

        self.assertEqual(loader.get_number_of_solutions(), 0)
        self.assertEqual(loader.get_solution_ids(), [])
        with self.assertRaisesRegex(NoSolutionError, "^error message$"):
            loader.get_vars()
        with self.assertRaisesRegex(NoSolutionError, "^error message$"):
            loader.get_duals()
        with self.assertRaisesRegex(NoSolutionError, "^error message$"):
            loader.get_reduced_costs()
        with self.assertRaisesRegex(NoSolutionError, "^error message$"):
            loader.load_solution()
        with self.assertRaisesRegex(NoSolutionError, "^error message$"):
            loader.load_vars()
        # If there are no suffixes declared on the model, then there
        # should be no error (because there is nothing to import)
        self.assertEqual(loader.load_import_suffixes(), None)
        # non-standard suffixes are ignored
        # TODO: is this "good" behavior??
        model.my_suffix = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        self.assertEqual(loader.load_import_suffixes(), None)
        # but duals / rc will generate an exception
        model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        with self.assertRaisesRegex(NoSolutionError, "^error message$"):
            loader.load_import_suffixes()
        # but duals / rc will generate an exception
        del model.dual
        model.rc = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        with self.assertRaisesRegex(NoSolutionError, "^error message$"):
            loader.load_import_suffixes()
