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

from pyomo.core.base.constraint import ConstraintData
from pyomo.core.base.objective import ObjectiveData
from pyomo.core.base.param import ParamData
from pyomo.core.base.sos import SOSConstraintData
from pyomo.core.base.var import VarData
import pyomo.environ as pyo
from pyomo.common import unittest
from typing import List
from pyomo.contrib.observer.model_observer import (
    Observer,
    ModelChangeDetector,
    AutoUpdateConfig,
)
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
        counts is a mapping from component (e.g., variable) to another 
        mapping from string ('add', 'remove', 'update', or 'set') to an int that 
        indicates the number of times the corresponding method has been called
        """

    def check(self, expected):
        unittest.assertStructuredAlmostEqual(
            first=expected, second=self.counts, places=7
        )

    def _process(self, comps, key):
        for c in comps:
            if c not in self.counts:
                self.counts[c] = make_count_dict()
            self.counts[c][key] += 1

    def pprint(self):
        for k, d in self.counts.items():
            print(f'{k}:')
            for a, v in d.items():
                print(f'  {a}: {v}')

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
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.p = pyo.Param(mutable=True, initialize=1)

        obs = ObserverChecker()
        detector = ModelChangeDetector([obs])

        expected = ComponentMap()
        expected[None] = make_count_dict()
        expected[None]['set'] += 1

        detector.set_instance(m)
        obs.check(expected)

        m.obj = pyo.Objective(expr=m.x**2 + m.p * m.y**2)
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
        m.obj = pyo.Objective(expr=m.p * m.x)
        detector.update()
        expected[m.p]['add'] += 1
        expected[m.y]['remove'] += 1
        # remember, m.obj is a different object now
        expected[m.obj] = make_count_dict()
        expected[m.obj]['set'] += 1

    def test_constraints(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.p = pyo.Param(mutable=True, initialize=1)

        obs = ObserverChecker()
        detector = ModelChangeDetector([obs])

        expected = ComponentMap()
        expected[None] = make_count_dict()
        expected[None]['set'] += 1

        detector.set_instance(m)
        obs.check(expected)

        m.obj = pyo.Objective(expr=m.y)
        m.c1 = pyo.Constraint(expr=m.y >= (m.x - m.p) ** 2)
        detector.update()
        expected[m.x] = make_count_dict()
        expected[m.y] = make_count_dict()
        expected[m.p] = make_count_dict()
        expected[m.x]['add'] += 1
        expected[m.y]['add'] += 1
        expected[m.p]['add'] += 1
        expected[m.c1] = make_count_dict()
        expected[m.c1]['add'] += 1
        expected[m.obj] = make_count_dict()
        expected[m.obj]['set'] += 1
        obs.check(expected)

        # now fix a variable and make sure the
        # constraint gets removed and added
        m.x.fix(1)
        obs.pprint()
        detector.update()
        obs.pprint()
        expected[m.c1]['remove'] += 1
        expected[m.c1]['add'] += 1
        # because x and p are only used in the
        # one constraint, they get removed when
        # the constraint is removed and then
        # added again when the constraint is added
        expected[m.x]['update'] += 1
        expected[m.x]['remove'] += 1
        expected[m.x]['add'] += 1
        expected[m.p]['remove'] += 1
        expected[m.p]['add'] += 1
        obs.check(expected)

    def test_sos(self):
        m = pyo.ConcreteModel()
        m.a = pyo.Set(initialize=[1, 2, 3], ordered=True)
        m.x = pyo.Var(m.a, within=pyo.Binary)
        m.y = pyo.Var(within=pyo.Binary)
        m.obj = pyo.Objective(expr=m.y)
        m.c1 = pyo.SOSConstraint(var=m.x, sos=1)

        obs = ObserverChecker()
        detector = ModelChangeDetector([obs])
        detector.set_instance(m)

        expected = ComponentMap()
        expected[m.obj] = make_count_dict()
        for i in m.a:
            expected[m.x[i]] = make_count_dict()
        expected[m.y] = make_count_dict()
        expected[m.c1] = make_count_dict()
        expected[m.obj]['set'] += 1
        for i in m.a:
            expected[m.x[i]]['add'] += 1
        expected[m.y]['add'] += 1
        expected[m.c1]['add'] += 1
        obs.check(expected)

        for i in m.a:
            expected[m.x[i]]['remove'] += 1
        expected[m.c1]['remove'] += 1
        del m.c1
        detector.update()
        obs.check(expected)

    def test_vars_and_params_elsewhere(self):
        m1 = pyo.ConcreteModel()
        m1.x = pyo.Var()
        m1.y = pyo.Var()
        m1.p = pyo.Param(mutable=True, initialize=1)

        m2 = pyo.ConcreteModel()

        obs = ObserverChecker()
        detector = ModelChangeDetector([obs])

        expected = ComponentMap()
        expected[None] = make_count_dict()
        expected[None]['set'] += 1

        detector.set_instance(m2)
        obs.check(expected)

        m2.obj = pyo.Objective(expr=m1.y)
        m2.c1 = pyo.Constraint(expr=m1.y >= (m1.x - m1.p) ** 2)
        detector.update()
        expected[m1.x] = make_count_dict()
        expected[m1.y] = make_count_dict()
        expected[m1.p] = make_count_dict()
        expected[m1.x]['add'] += 1
        expected[m1.y]['add'] += 1
        expected[m1.p]['add'] += 1
        expected[m2.c1] = make_count_dict()
        expected[m2.c1]['add'] += 1
        expected[m2.obj] = make_count_dict()
        expected[m2.obj]['set'] += 1
        obs.check(expected)

        # now fix a variable and make sure the
        # constraint gets removed and added
        m1.x.fix(1)
        obs.pprint()
        detector.update()
        obs.pprint()
        expected[m2.c1]['remove'] += 1
        expected[m2.c1]['add'] += 1
        # because x and p are only used in the
        # one constraint, they get removed when
        # the constraint is removed and then
        # added again when the constraint is added
        expected[m1.x]['update'] += 1
        expected[m1.x]['remove'] += 1
        expected[m1.x]['add'] += 1
        expected[m1.p]['remove'] += 1
        expected[m1.p]['add'] += 1
        obs.check(expected)

    def test_named_expression(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.p = pyo.Param(mutable=True, initialize=1)

        obs = ObserverChecker()
        detector = ModelChangeDetector([obs])

        expected = ComponentMap()
        expected[None] = make_count_dict()
        expected[None]['set'] += 1

        detector.set_instance(m)
        obs.check(expected)

        m.obj = pyo.Objective(expr=m.y)
        m.e = pyo.Expression(expr=m.x - m.p)
        m.c1 = pyo.Constraint(expr=m.y >= m.e)
        detector.update()
        expected[m.x] = make_count_dict()
        expected[m.y] = make_count_dict()
        expected[m.p] = make_count_dict()
        expected[m.x]['add'] += 1
        expected[m.y]['add'] += 1
        expected[m.p]['add'] += 1
        expected[m.c1] = make_count_dict()
        expected[m.c1]['add'] += 1
        expected[m.obj] = make_count_dict()
        expected[m.obj]['set'] += 1
        obs.check(expected)

        # now modify the named expression and make sure the
        # constraint gets removed and added
        m.e.expr = (m.x - m.p) ** 2
        detector.update()
        expected[m.c1]['remove'] += 1
        expected[m.c1]['add'] += 1
        # because x and p are only used in the
        # one constraint, they get removed when
        # the constraint is removed and then
        # added again when the constraint is added
        expected[m.x]['remove'] += 1
        expected[m.x]['add'] += 1
        expected[m.p]['remove'] += 1
        expected[m.p]['add'] += 1
        obs.check(expected)
