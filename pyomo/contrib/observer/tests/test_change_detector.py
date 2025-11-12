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

import logging
from typing import List, Mapping

import pyomo.environ as pyo
from pyomo.core.base.constraint import ConstraintData
from pyomo.core.base.objective import ObjectiveData
from pyomo.core.base.param import ParamData
from pyomo.core.base.sos import SOSConstraintData
from pyomo.core.base.var import VarData
from pyomo.common import unittest
from pyomo.contrib.observer.model_observer import (
    Observer,
    ModelChangeDetector,
    AutoUpdateConfig,
    Reason,
)
from pyomo.common.collections import DefaultComponentMap, ComponentMap
from pyomo.common.errors import PyomoException


logger = logging.getLogger(__name__)


def make_count_dict():
    d = {i: 0 for i in Reason}
    return d


class ObserverChecker(Observer):
    def __init__(self):
        super().__init__()
        self.counts = DefaultComponentMap(make_count_dict)
        """
        counts is a mapping from component (e.g., variable) to another 
        mapping from Reason to an int that indicates the number of times 
        the corresponding method has been called
        """

    def check(self, expected):
        unittest.assertStructuredAlmostEqual(
            first=expected, second=self.counts, places=7
        )

    def pprint(self):
        for k, d in self.counts.items():
            print(f'{k}:')
            for a, v in d.items():
                print(f'  {a}: {v}')

    def _update_variables(self, variables: Mapping[VarData, Reason]):
        for v, reason in variables.items():
            self.counts[v][reason] += 1

    def _update_parameters(self, params: Mapping[ParamData, Reason]):
        for p, reason in params.items():
            self.counts[p][reason] += 1

    def _update_constraints(self, cons: Mapping[ConstraintData, Reason]):
        for c, reason in cons.items():
            self.counts[c][reason] += 1

    def _update_sos_constraints(self, cons: Mapping[SOSConstraintData, Reason]):
        for c, reason in cons.items():
            self.counts[c][reason] += 1

    def _update_objectives(self, objs: Mapping[ObjectiveData, Reason]):
        for obj, reason in objs.items():
            self.counts[obj][reason] += 1


class TestChangeDetector(unittest.TestCase):
    def test_objective(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.p = pyo.Param(mutable=True, initialize=1)

        obs = ObserverChecker()
        detector = ModelChangeDetector(m, [obs])

        expected = DefaultComponentMap(make_count_dict)
        obs.check(expected)

        m.obj = pyo.Objective(expr=m.x**2 + m.p * m.y**2)
        detector.update()
        expected[m.obj][Reason.added] += 1
        expected[m.x][Reason.added] += 1
        expected[m.y][Reason.added] += 1
        expected[m.p][Reason.added] += 1
        obs.check(expected)

        m.y.setlb(0)
        detector.update()
        expected[m.y][Reason.bounds] += 1
        obs.check(expected)

        m.x.fix(2)
        detector.update()
        expected[m.x][Reason.fixed] += 1
        obs.check(expected)

        m.x.unfix()
        detector.update()
        expected[m.x][Reason.fixed] += 1
        obs.check(expected)

        m.p.value = 2
        detector.update()
        expected[m.p][Reason.value] += 1
        obs.check(expected)

        m.obj.expr = m.x**2 + m.y**2
        detector.update()
        expected[m.obj][Reason.expr] += 1
        expected[m.p][Reason.removed] += 1
        obs.check(expected)

        expected[m.obj][Reason.removed] += 1
        del m.obj
        m.obj2 = pyo.Objective(expr=m.p * m.x)
        detector.update()
        # remember, m.obj is a different object now
        expected[m.obj2][Reason.added] += 1
        expected[m.y][Reason.removed] += 1
        expected[m.p][Reason.added] += 1
        obs.check(expected)

    def test_constraints(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.p = pyo.Param(mutable=True, initialize=1)

        obs = ObserverChecker()
        detector = ModelChangeDetector(m, [obs])

        expected = DefaultComponentMap(make_count_dict)
        obs.check(expected)

        m.obj = pyo.Objective(expr=m.y)
        m.c1 = pyo.Constraint(expr=m.y >= (m.x - m.p) ** 2)
        detector.update()
        expected[m.x][Reason.added] += 1
        expected[m.y][Reason.added] += 1
        expected[m.p][Reason.added] += 1
        expected[m.c1][Reason.added] += 1
        expected[m.obj][Reason.added] += 1
        obs.check(expected)

        m.x.fix(1)
        detector.update()
        expected[m.x][Reason.fixed] += 1
        obs.check(expected)

        m.z = pyo.Var()
        m.c1.set_value(m.y == 2 * m.z)
        detector.update()
        expected[m.z][Reason.added] += 1
        expected[m.c1][Reason.expr] += 1
        expected[m.p][Reason.removed] += 1
        expected[m.x][Reason.removed] += 1
        obs.check(expected)

        expected[m.c1][Reason.removed] += 1
        del m.c1
        detector.update()
        expected[m.z][Reason.removed] += 1
        obs.check(expected)

    def test_sos(self):
        m = pyo.ConcreteModel()
        m.a = pyo.Set(initialize=[1, 2, 3], ordered=True)
        m.x = pyo.Var(m.a, within=pyo.Binary)
        m.y = pyo.Var(within=pyo.Binary)
        m.obj = pyo.Objective(expr=m.y)
        m.c1 = pyo.SOSConstraint(var=m.x, sos=1)

        obs = ObserverChecker()
        detector = ModelChangeDetector(m, [obs])

        expected = DefaultComponentMap(make_count_dict)
        expected[m.obj][Reason.added] += 1
        for i in m.a:
            expected[m.x[i]][Reason.added] += 1
        expected[m.y][Reason.added] += 1
        expected[m.c1][Reason.added] += 1
        obs.check(expected)

        detector.update()
        obs.check(expected)

        m.c1.set_items([m.x[2], m.x[1], m.x[3]], [1, 2, 3])
        detector.update()
        expected[m.c1][Reason.sos_items] += 1
        obs.check(expected)

        m.c1.set_items([m.x[2], m.x[1]], [1, 2])
        detector.update()
        expected[m.c1][Reason.sos_items] += 1
        expected[m.x[3]][Reason.removed] += 1
        obs.check(expected)

        m.c1.set_items([m.x[2], m.x[1], m.x[3]], [1, 2, 3])
        detector.update()
        expected[m.c1][Reason.sos_items] += 1
        expected[m.x[3]][Reason.added] += 1
        obs.check(expected)

        for i in m.a:
            expected[m.x[i]][Reason.removed] += 1
        expected[m.c1][Reason.removed] += 1
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
        detector = ModelChangeDetector(m2, [obs])

        expected = DefaultComponentMap(make_count_dict)
        obs.check(expected)

        m2.obj = pyo.Objective(expr=m1.y)
        m2.c1 = pyo.Constraint(expr=m1.y >= (m1.x - m1.p) ** 2)
        detector.update()
        expected[m1.x][Reason.added] += 1
        expected[m1.y][Reason.added] += 1
        expected[m1.p][Reason.added] += 1
        expected[m2.c1][Reason.added] += 1
        expected[m2.obj][Reason.added] += 1
        obs.check(expected)

        m1.x.fix(1)
        detector.update()
        expected[m1.x][Reason.fixed] += 1
        obs.check(expected)

    def test_named_expression(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.p = pyo.Param(mutable=True, initialize=1)

        obs = ObserverChecker()
        detector = ModelChangeDetector(m, [obs])

        expected = DefaultComponentMap(make_count_dict)
        obs.check(expected)

        m.obj = pyo.Objective(expr=m.y)
        m.e = pyo.Expression(expr=m.x - m.p)
        m.c1 = pyo.Constraint(expr=m.y >= m.e)
        detector.update()
        expected[m.x][Reason.added] += 1
        expected[m.y][Reason.added] += 1
        expected[m.p][Reason.added] += 1
        expected[m.c1][Reason.added] += 1
        expected[m.obj][Reason.added] += 1
        obs.check(expected)

        # now modify the named expression and make sure the
        # constraint gets removed and added
        m.e.expr = (m.x - m.p) ** 2
        detector.update()
        expected[m.c1][Reason.expr] += 1
        obs.check(expected)

    def test_update_config(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.p = pyo.Param(initialize=1, mutable=True)

        obs = ObserverChecker()
        detector = ModelChangeDetector(m, [obs])
        expected = DefaultComponentMap(make_count_dict)
        obs.check(expected)

        detector.config.check_for_new_or_removed_constraints = False
        detector.config.check_for_new_or_removed_objectives = False
        detector.config.update_constraints = False
        detector.config.update_objectives = False
        detector.config.update_vars = False
        detector.config.update_parameters = False
        detector.config.update_named_expressions = False

        m.e = pyo.Expression(expr=pyo.exp(m.x))
        m.obj = pyo.Objective(expr=m.x**2 + m.p * m.y**2)
        m.c1 = pyo.Constraint(expr=m.y >= m.e + m.p)

        detector.update()
        obs.check(expected)

        detector.config.check_for_new_or_removed_constraints = True
        detector.update()
        expected[m.x][Reason.added] += 1
        expected[m.y][Reason.added] += 1
        expected[m.p][Reason.added] += 1
        expected[m.c1][Reason.added] += 1
        obs.check(expected)

        detector.config.check_for_new_or_removed_objectives = True
        detector.update()
        expected[m.obj][Reason.added] += 1
        obs.check(expected)

        m.x.setlb(0)
        detector.update()
        obs.check(expected)

        detector.config.update_vars = True
        detector.update()
        expected[m.x][Reason.bounds] += 1
        obs.check(expected)

        m.p.value = 2
        detector.update()
        obs.check(expected)

        detector.config.update_parameters = True
        detector.update()
        expected[m.p][Reason.value] += 1
        obs.check(expected)

        m.e.expr += 1
        detector.update()
        obs.check(expected)

        detector.config.update_named_expressions = True
        detector.update()
        expected[m.c1][Reason.expr] += 1
        obs.check(expected)

        m.obj.expr += 1
        detector.update()
        obs.check(expected)

        detector.config.update_objectives = True
        detector.update()
        expected[m.obj][Reason.expr] += 1
        obs.check(expected)

        m.c1 = m.y >= m.e
        detector.update()
        obs.check(expected)

        detector.config.update_constraints = True
        detector.update()
        expected[m.c1][Reason.expr] += 1
        obs.check(expected)

    def test_param_in_bounds(self):
        m = pyo.ConcreteModel()
        m.y = pyo.Var()
        m.p = pyo.Param(mutable=True, initialize=1)

        obs = ObserverChecker()
        detector = ModelChangeDetector(m, [obs])

        expected = DefaultComponentMap(make_count_dict)
        obs.check(expected)

        m.obj = pyo.Objective(expr=m.y)
        m.y.setlb(m.p - 1)
        detector.update()
        expected[m.y][Reason.added] += 1
        expected[m.p][Reason.added] += 1
        expected[m.obj][Reason.added] += 1
        obs.check(expected)

        m.p.value = 2
        detector.update()
        expected[m.p][Reason.value] += 1
        obs.check(expected)

        m.p2 = pyo.Param(mutable=True, initialize=1)
        m.y.setub(m.p2 + 1)
        detector.update()
        expected[m.p2][Reason.added] += 1
        expected[m.y][Reason.bounds] += 1
        obs.check(expected)

    def test_incidence(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.z = pyo.Var()
        m.p1 = pyo.Param(mutable=True, initialize=1)
        m.p2 = pyo.Param(mutable=True, initialize=1)
        m.x.setlb(m.p1)

        m.e1 = pyo.Expression(expr=m.x + m.p1)
        m.e2 = pyo.Expression(expr=(m.e1**2))
        m.obj = pyo.Objective(expr=m.e2 + m.y**2)
        m.c1 = pyo.Constraint(expr=m.z + m.p2 == 0)
        m.c2 = pyo.Constraint(expr=m.x + m.p2 == 0)

        obs = ObserverChecker()
        detector = ModelChangeDetector(m, [obs])

        expected = DefaultComponentMap(make_count_dict)
        expected[m.x][Reason.added] += 1
        expected[m.y][Reason.added] += 1
        expected[m.z][Reason.added] += 1
        expected[m.p1][Reason.added] += 1
        expected[m.p2][Reason.added] += 1
        expected[m.obj][Reason.added] += 1
        expected[m.c1][Reason.added] += 1
        expected[m.c2][Reason.added] += 1
        obs.check(expected)

        self.assertEqual(detector.get_variables_impacted_by_param(m.p1), [m.x])
        self.assertEqual(detector.get_variables_impacted_by_param(m.p2), [])
        self.assertEqual(detector.get_constraints_impacted_by_param(m.p1), [])
        self.assertEqual(detector.get_constraints_impacted_by_param(m.p2), [m.c1, m.c2])
        self.assertEqual(detector.get_constraints_impacted_by_var(m.x), [m.c2])
        self.assertEqual(detector.get_constraints_impacted_by_var(m.y), [])
        self.assertEqual(detector.get_constraints_impacted_by_var(m.z), [m.c1])
        self.assertEqual(detector.get_objectives_impacted_by_param(m.p1), [m.obj])
        self.assertEqual(detector.get_objectives_impacted_by_param(m.p2), [])
        self.assertEqual(detector.get_objectives_impacted_by_var(m.x), [m.obj])
        self.assertEqual(detector.get_objectives_impacted_by_var(m.y), [m.obj])
        self.assertEqual(detector.get_objectives_impacted_by_var(m.z), [])

        m.e1.expr += m.z
        detector.update()
        expected[m.obj][Reason.expr] += 1
        obs.check(expected)

        self.assertEqual(detector.get_objectives_impacted_by_param(m.p1), [m.obj])
        self.assertEqual(detector.get_objectives_impacted_by_param(m.p2), [])
        self.assertEqual(detector.get_objectives_impacted_by_var(m.x), [m.obj])
        self.assertEqual(detector.get_objectives_impacted_by_var(m.y), [m.obj])
        self.assertEqual(detector.get_objectives_impacted_by_var(m.z), [m.obj])

    def test_manual_updates(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.p = pyo.Param(mutable=True, initialize=1)

        obs = ObserverChecker()
        detector = ModelChangeDetector(m, [obs])

        expected = DefaultComponentMap(make_count_dict)
        obs.check(expected)

        m.obj = pyo.Objective(expr=m.y)
        m.c1 = pyo.Constraint(expr=m.y >= (m.x - m.p) ** 2)
        m.c2 = pyo.Constraint(expr=m.x + m.y == 0)

        detector.add_objectives([m.obj])
        expected[m.obj][Reason.added] += 1
        expected[m.y][Reason.added] += 1
        obs.check(expected)

        detector.add_constraints([m.c1])
        expected[m.x][Reason.added] += 1
        expected[m.p][Reason.added] += 1
        expected[m.c1][Reason.added] += 1
        obs.check(expected)

        detector.add_constraints([m.c2])
        expected[m.c2][Reason.added] += 1
        obs.check(expected)

        detector.remove_constraints([m.c1])
        expected[m.c1][Reason.removed] += 1
        expected[m.p][Reason.removed] += 1
        obs.check(expected)

        detector.add_constraints([m.c1])
        expected[m.c1][Reason.added] += 1
        expected[m.p][Reason.added] += 1
        obs.check(expected)

        detector.remove_objectives([m.obj])
        expected[m.obj][Reason.removed] += 1
        obs.check(expected)

        detector.add_objectives([m.obj])
        expected[m.obj][Reason.added] += 1
        obs.check(expected)

        m.x.setlb(0)
        detector.update_variables([m.x, m.y])
        expected[m.x][Reason.bounds] += 1
        obs.check(expected)

        m.p.value = 2
        detector.update_parameters([m.p])
        expected[m.p][Reason.value] += 1
        obs.check(expected)

        m.c1.set_value(m.y >= m.x**2)
        detector.update_constraints([m.c1, m.c2])
        expected[m.p][Reason.removed] += 1
        expected[m.c1][Reason.expr] += 1
        obs.check(expected)

        m.obj.expr += m.x
        detector.update_objectives([m.obj])
        expected[m.obj][Reason.expr] += 1
        obs.check(expected)

    def test_mutable_parameters_in_sos(self):
        """
        There is logic in the ModelChangeDetector to handle
        mutable parameters in SOS constraints. However, we cannot
        currently test it because of #3769. For now, we will
        just make sure that an error is raised when attempting to
        use a mutable parameter in an SOS constraint. If #3769 is
        resolved, we will just need to update this test to make
        sure the ModelChangeDetector does the right thing.
        """
        m = pyo.ConcreteModel()
        m.a = pyo.Set(initialize=[1, 2, 3])
        m.x = pyo.Var(m.a)
        m.p = pyo.Param(m.a, mutable=True)
        m.p[1].value = 1
        m.p[2].value = 2
        m.p[3].value = 3

        with self.assertRaisesRegex(
            PyomoException, 'Cannot convert non-constant Pyomo expression .* to bool.*'
        ):
            m.c = pyo.SOSConstraint(var=m.x, sos=1, weights=m.p)
