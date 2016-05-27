import gc
import os
thisdir = os.path.dirname(os.path.abspath(__file__))

import pyutilib.th as unittest

from pyomo.core import (ConcreteModel,
                        Block,
                        Param,
                        Set,
                        Var,
                        Expression,
                        Constraint,
                        Objective,
                        Reals)

class ComponentPerformanceBase(object):
    @classmethod
    def _create_model(self, ctype, **kwds):
        self.model = ConcreteModel()
        self.model.x = Var()
        self.model.index = Set(initialize=sorted(range(1000000)))
        self.model.del_component('test_component')
        self.model.test_component = \
            ctype(self.model.index, **kwds)

    @classmethod
    def setUp(self):
        if self.model is None:
            self._setup()

    @classmethod
    def setUpClass(self):
        self.model = None

    @classmethod
    def tearDownClass(self):
        self.model = None

    def test_0_setup(self):
        # Needed so that the time to set up the model is not included in
        # the subsequent performance tests.  This test is named so that
        # it should appear first in the set of tests run by this test class
        pass

    def test_iteration(self):
        cnt = 0
        for cdata in self.model.component_data_objects(self.model.test_component.type()):
            cnt += 1
        self.assertTrue(cnt > 0)
        if self.model.test_component.type() in (Set, Var):
            self.assertEqual(cnt,
                             len(self.model.test_component) + 1)
        else:
            self.assertEqual(cnt,
                             len(self.model.test_component))

@unittest.category('performance')
class TestMutableParamPerformance(ComponentPerformanceBase, unittest.TestCase):
    @classmethod
    def _setup(self):
        self._create_model(Param,
                         **{'initialize_as_dense':True,
                            'initialize':1.0,
                            'mutable':True})

@unittest.category('performance')
class TestParamPerformance(ComponentPerformanceBase, unittest.TestCase):
    @classmethod
    def _setup(self):
        self._create_model(Param,
                         **{'initialize_as_dense':True,
                            'initialize':1.0,
                            'mutable':False})

@unittest.category('performance')
class TestVarPerformance(ComponentPerformanceBase, unittest.TestCase):
    @classmethod
    def _setup(self):
        self._create_model(Var, **{'initialize':1.0})

@unittest.category('performance')
class TestVarMultiDomainPerformance(ComponentPerformanceBase, unittest.TestCase):
    @classmethod
    def _setup(self):
        self._create_model(Var,
                         **{'domain': lambda m,i: Reals})

@unittest.category('performance')
class TestExpressionPerformance(ComponentPerformanceBase, unittest.TestCase):
    @classmethod
    def _setup(self):
        self._create_model(Expression,
                         **{'initialize': 1.0})

@unittest.category('performance')
class TestConstraintPerformance(ComponentPerformanceBase, unittest.TestCase):
    @classmethod
    def _setup(self):
        self._create_model(Constraint,
                         **{'rule': lambda m,i: 1 <= m.x <= 2})

@unittest.category('performance')
class TestObjectivePerformance(ComponentPerformanceBase, unittest.TestCase):
    @classmethod
    def _setup(self):
        self._create_model(Objective,
                         **{'rule': lambda m,i: m.x})

@unittest.category('performance')
class TestSetPerformance(ComponentPerformanceBase, unittest.TestCase):
    @classmethod
    def _setup(self):
        self._create_model(Set,
                         **{'initialize': [1,2,3]})

@unittest.category('performance')
class TestBlockPerformance(ComponentPerformanceBase, unittest.TestCase):
    @classmethod
    def _setup(self):
        self._create_model(Block, **{'rule': lambda b,i: b})

    def test_block_iteration(self):
        cnt = 0
        for block in self.model.block_data_objects():
            cnt += 1
        self.assertTrue(cnt > 0)
        self.assertEqual(cnt,
                         len(self.model.test_component) + 1)

if __name__ == "__main__":
    unittest.main()



