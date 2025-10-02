#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  __________________________________________________________________________

import pyomo.environ as pyo
from pyomo.common import unittest
from pyomo.contrib.observer.component_collector import collect_components_from_expr
from pyomo.common.collections import ComponentSet


class TestComponentCollector(unittest.TestCase):
    def test_nested_named_expressions(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.z = pyo.Var()
        m.e1 = pyo.Expression(expr=m.x + m.y)
        m.e2 = pyo.Expression(expr=m.e1 + m.z)
        e = m.e2 * pyo.exp(m.e2)
        (named_exprs, vars, params, external_funcs) = collect_components_from_expr(e)
        self.assertEqual(len(named_exprs), 2)
        named_exprs = ComponentSet(named_exprs)
        self.assertIn(m.e1, named_exprs)
        self.assertIn(m.e2, named_exprs)
