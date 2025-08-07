from pyomo.core.base.constraint import ConstraintData
from pyomo.core.base.objective import ObjectiveData
from pyomo.core.base.param import ParamData
from pyomo.core.base.sos import SOSConstraintData
from pyomo.core.base.var import VarData
import pyomo.environ as pe
from pyomo.common import unittest
from typing import List
from pyomo.contrib.observer.model_observer import Observer, ModelChangeDetector, AutoUpdateConfig
from pyomo.common.collections import ComponentMap
import logging


logger = logging.getLogger(__name__)


def make_count_dict():
    d = {'add': 0, 'remove': 0, 'update': 0, 'set': 0}
    return d


class ObserverChecker(Observer):
    def __init__(self):
        super().__init__()
        self.counts = ComponentMap()
        """
        counts is be a mapping from component (e.g., variable) to another 
        mapping from string ('add', 'remove', 'update', or 'set') to an int that 
        indicates the number of times the corresponding method has been called
        """

    def check(self, expected):
        unittest.assertStructuredAlmostEqual(
            first=expected,
            second=self.counts,
            places=7,
        )
    
    def _process(self, comps, key):
        for c in comps:
            if c not in self.counts:
                self.counts[c] = make_count_dict()
            self.counts[c][key] += 1

    def add_variables(self, variables: List[VarData]):
        for v in variables:
            assert v.is_variable_type()
        self._process(variables, 'add')

    def add_parameters(self, params: List[ParamData]):
        for p in params:
            assert p.is_parameter_type()
        self._process(params, 'add')

    def add_constraints(self, cons: List[ConstraintData]):
        for c in cons:
            assert isinstance(c, ConstraintData)
        self._process(cons, 'add')

    def add_sos_constraints(self, cons: List[SOSConstraintData]):
        for c in cons:
            assert isinstance(c, SOSConstraintData)
        self._process(cons, 'add')

    def set_objective(self, obj: ObjectiveData):
        assert obj is None or isinstance(obj, ObjectiveData)
        self._process([obj], 'set')

    def remove_constraints(self, cons: List[ConstraintData]):
        for c in cons:
            assert isinstance(c, ConstraintData)
        self._process(cons, 'remove')

    def remove_sos_constraints(self, cons: List[SOSConstraintData]):
        for c in cons:
            assert isinstance(c, SOSConstraintData)
        self._process(cons, 'remove')

    def remove_variables(self, variables: List[VarData]):
        for v in variables:
            assert v.is_variable_type()
        self._process(variables, 'remove')

    def remove_parameters(self, params: List[ParamData]):
        for p in params:
            assert p.is_parameter_type()
        self._process(params, 'remove')

    def update_variables(self, variables: List[VarData]):
        for v in variables:
            assert v.is_variable_type()
        self._process(variables, 'update')

    def update_parameters(self, params: List[ParamData]):
        for p in params:
            assert p.is_parameter_type()
        self._process(params, 'update')


class TestChangeDetector(unittest.TestCase):
    def test_objective(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.p = pe.Param(mutable=True, initialize=1)

        obs = ObserverChecker()
        detector = ModelChangeDetector([obs])

        expected = ComponentMap()
        expected[None] = make_count_dict()
        expected[None]['set'] += 1

        detector.set_instance(m)
        obs.check(expected)

        m.obj = pe.Objective(expr=m.x**2 + m.p*m.y**2)
        detector.update()
        expected[m.obj] = make_count_dict()
        expected[m.obj]['set'] += 1
        expected[m.x] = make_count_dict()
        expected[m.x]['add'] += 1
        expected[m.y] = make_count_dict()
        expected[m.y]['add'] += 1
        expected[m.p] = make_count_dict()
        expected[m.p]['add'] += 1
        obs.check(expected)
        
        m.y.setlb(0)
        detector.update()
        expected[m.y]['update'] += 1
        obs.check(expected)

        m.x.fix(2)
        detector.update()
        expected[m.x]['update'] += 1
        expected[m.obj]['set'] += 1
        obs.check(expected)

        m.x.unfix()
        detector.update()
        expected[m.x]['update'] += 1
        expected[m.obj]['set'] += 1
        obs.check(expected)

        m.p.value = 2
        detector.update()
        expected[m.p]['update'] += 1
        obs.check(expected)

        m.obj.expr = m.x**2 + m.y**2
        detector.update()
        expected[m.p]['remove'] += 1
        expected[m.obj]['set'] += 1
        obs.check(expected)

        del m.obj
        m.obj = pe.Objective(expr=m.p*m.x)
        detector.update()
        expected[m.p]['add'] += 1
        expected[m.y]['remove'] += 1
        # remember, m.obj is a different object now
        expected[m.obj] = make_count_dict()
        expected[m.obj]['set'] += 1

    def test_constraints(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.p = pe.Param(mutable=True, initialize=1)

        obs = ObserverChecker()
        detector = ModelChangeDetector([obs])

        expected = ComponentMap()
        expected[None] = make_count_dict()
        expected[None]['set'] += 1

        detector.set_instance(m)
        obs.check(expected)

        m.c1 = pe.Constraint(expr=m.y >= (m.x - m.p)**2)
        detector.update()
        expected[m.x] = make_count_dict()
        expected[m.y] = make_count_dict()
        expected[m.p] = make_count_dict()
        expected[m.x]['add'] += 1
        expected[m.y]['add'] += 1
        expected[m.p]['add'] += 1
        expected[m.c1] = make_count_dict()
        expected[m.c1]['add'] += 1
        obs.check(expected)

    def test_vars_and_params_elsewhere(self):
        pass