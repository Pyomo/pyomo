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

from pyomo.common import unittest
import pyomo.environ as pyo
from pyomo.contrib.appsi.utils import collect_vars_and_named_exprs
from pyomo.contrib.appsi.cmodel import cmodel, cmodel_available
from typing import Callable
from pyomo.common.gsl import find_GSL


class TestCollectVarsAndNamedExpressions(unittest.TestCase):
    def basics_helper(self, collector: Callable, *args):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.z = pyo.Var()
        m.E = pyo.Expression(expr=2 * m.z + 1)
        m.y.fix(3)
        e = m.x * m.y + m.x * m.E
        named_exprs, var_list, fixed_vars, external_funcs = collector(e, *args)
        self.assertEqual([m.E], named_exprs)
        self.assertEqual([m.x, m.y, m.z], var_list)
        self.assertEqual([m.y], fixed_vars)
        self.assertEqual([], external_funcs)

    def test_basics(self):
        self.basics_helper(collect_vars_and_named_exprs)

    @unittest.skipUnless(cmodel_available, 'appsi extensions are not available')
    def test_basics_cmodel(self):
        self.basics_helper(cmodel.prep_for_repn, cmodel.PyomoExprTypes())

    def external_func_helper(self, collector: Callable, *args):
        DLL = find_GSL()
        if not DLL:
            self.skipTest('Could not find amplgsl.dll library')

        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.z = pyo.Var()
        m.hypot = pyo.ExternalFunction(library=DLL, function='gsl_hypot')
        func = m.hypot(m.x, m.x * m.y)
        m.E = pyo.Expression(expr=2 * func)
        m.y.fix(3)
        e = m.z + m.x * m.E
        named_exprs, var_list, fixed_vars, external_funcs = collector(e, *args)
        self.assertEqual([m.E], named_exprs)
        self.assertEqual([m.z, m.x, m.y], var_list)
        self.assertEqual([m.y], fixed_vars)
        self.assertEqual([func], external_funcs)

    def test_external(self):
        self.external_func_helper(collect_vars_and_named_exprs)

    @unittest.skipUnless(cmodel_available, 'appsi extensions are not available')
    def test_external_cmodel(self):
        self.basics_helper(cmodel.prep_for_repn, cmodel.PyomoExprTypes())
