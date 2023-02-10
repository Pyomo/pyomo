import pyomo.environ as pe
import pyomo.common.unittest as unittest
from pyomo.contrib.appsi.cmodel import cmodel_available
from pyomo.common.gsl import find_GSL


@unittest.skipUnless(cmodel_available, 'appsi extensions are not available')
class TestIpoptPersistent(unittest.TestCase):
    def test_external_function(self):
        DLL = find_GSL()
        if not DLL:
            self.skipTest('Could not find the amplgls.dll library')

        opt = pe.SolverFactory('appsi_ipopt')
        if not opt.available(exception_flag=False):
            raise unittest.SkipTest

        m = pe.ConcreteModel()
        m.hypot = pe.ExternalFunction(library=DLL, function='gsl_hypot')
        m.x = pe.Var(bounds=(-10, 10), initialize=2)
        m.y = pe.Var(initialize=2)
        e = 2 * m.hypot(m.x, m.x * m.y)
        m.c = pe.Constraint(expr=e == 2.82843)
        m.obj = pe.Objective(expr=m.x)
        res = opt.solve(m)
        pe.assert_optimal_termination(res)
        self.assertAlmostEqual(pe.value(m.c.body) - pe.value(m.c.lower), 0)

    def test_external_function_in_objective(self):
        DLL = find_GSL()
        if not DLL:
            self.skipTest('Could not find the amplgls.dll library')

        opt = pe.SolverFactory('appsi_ipopt')
        if not opt.available(exception_flag=False):
            raise unittest.SkipTest

        m = pe.ConcreteModel()
        m.hypot = pe.ExternalFunction(library=DLL, function='gsl_hypot')
        m.x = pe.Var(bounds=(1, 10), initialize=2)
        m.y = pe.Var(bounds=(1, 10), initialize=2)
        e = 2 * m.hypot(m.x, m.x * m.y)
        m.obj = pe.Objective(expr=e)
        res = opt.solve(m)
        pe.assert_optimal_termination(res)
        self.assertAlmostEqual(m.x.value, 1)
        self.assertAlmostEqual(m.y.value, 1)
