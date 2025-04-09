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

import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.contrib.appsi.cmodel import cmodel_available
from pyomo.common.gsl import find_GSL


@unittest.skipUnless(cmodel_available, 'appsi extensions are not available')
class TestIpoptPersistent(unittest.TestCase):
    def test_external_function(self):
        DLL = find_GSL()
        if not DLL:
            self.skipTest('Could not find the amplgls.dll library')

        opt = pyo.SolverFactory('appsi_ipopt')
        if not opt.available(exception_flag=False):
            raise unittest.SkipTest

        m = pyo.ConcreteModel()
        m.hypot = pyo.ExternalFunction(library=DLL, function='gsl_hypot')
        m.x = pyo.Var(bounds=(-10, 10), initialize=2)
        m.y = pyo.Var(initialize=2)
        e = 2 * m.hypot(m.x, m.x * m.y)
        m.c = pyo.Constraint(expr=e == 2.82843)
        m.obj = pyo.Objective(expr=m.x)
        res = opt.solve(m)
        pyo.assert_optimal_termination(res)
        self.assertAlmostEqual(pyo.value(m.c.body) - pyo.value(m.c.lower), 0)

    def test_external_function_in_objective(self):
        DLL = find_GSL()
        if not DLL:
            self.skipTest('Could not find the amplgls.dll library')

        opt = pyo.SolverFactory('appsi_ipopt')
        if not opt.available(exception_flag=False):
            raise unittest.SkipTest

        m = pyo.ConcreteModel()
        m.hypot = pyo.ExternalFunction(library=DLL, function='gsl_hypot')
        m.x = pyo.Var(bounds=(1, 10), initialize=2)
        m.y = pyo.Var(bounds=(1, 10), initialize=2)
        e = 2 * m.hypot(m.x, m.x * m.y)
        m.obj = pyo.Objective(expr=e)
        res = opt.solve(m)
        pyo.assert_optimal_termination(res)
        self.assertAlmostEqual(m.x.value, 1)
        self.assertAlmostEqual(m.y.value, 1)
