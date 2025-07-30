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


class ObserverChecker(Observer):
    def __init__(self):
        super().__init__()
        self.counts = ComponentMap()
        """
        counts is be a mapping from component (e.g., variable) to another 
        mapping from string ('add', 'remove', 'update', or 'value') to an int that 
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
                self.counts[c] = {'add': 0, 'remove': 0, 'update': 0, 'value': 0}
            self.counts[c][key] += 1

    def add_variables(self, variables: List[VarData]):
        self._process(variables, 'add')

    def add_parameters(self, params: List[ParamData]):
        self._process(params, 'add')

    def add_constraints(self, cons: List[ConstraintData]):
        self._process(cons, 'add')

    def add_sos_constraints(self, cons: List[SOSConstraintData]):
        self._process(cons, 'add')

    def set_objective(self, obj: ObjectiveData):
        self._process([obj], 'add')

    def remove_constraints(self, cons: List[ConstraintData]):
        self._process(cons, 'remove')

    def remove_sos_constraints(self, cons: List[SOSConstraintData]):
        self._process(cons, 'remove')

    def remove_variables(self, variables: List[VarData]):
        self._process(variables, 'remove')

    def remove_parameters(self, params: List[ParamData]):
        self._process(params, 'remove')

    def update_variables(self, variables: List[VarData]):
        self._process(variables, 'update')

    def update_parameters_and_fixed_variables(self, params: List[ParamData], variables: List[VarData]):
        self._process(params, 'value')
        self._process(variables, 'value')


class TestChangeDetector(unittest.TestCase):
    def test_basics(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()

        obs = ObserverChecker()
        detector = ModelChangeDetector([obs])

        detector.set_instance(m)

        expected = ComponentMap()

        obs.check(expected)

    def test_vars_and_params_elsewhere(self):
        pass