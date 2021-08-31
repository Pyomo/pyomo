import pyomo.environ as pe
import pyomo.common.unittest as unittest
try:
    from pyomo.contrib.appsi.cmodel import cmodel
except ImportError:
    raise unittest.SkipTest('appsi extensions are not available')
from pyomo.contrib.appsi.solvers import Ipopt
from pyomo.common.getGSL import find_GSL


class TestIpoptPersistent(unittest.TestCase):
    def test_external_function(self):
        DLL = find_GSL()
        if not DLL:
            self.skipTest('Could not find the amplgls.dll library')

        m = pe.ConcreteModel()
        m.hypot = pe.ExternalFunction(library=DLL, function='gsl_hypot')
        m.x = pe.Var(bounds=(-10, 10), initialize=2)
        m.y = pe.Var(initialize=2)
        e = 2 * m.hypot(m.x, m.x * m.y)
        m.c = pe.Constraint(expr=e == 2.82843)
        m.obj = pe.Objective(expr=m.x)
        opt: Ipopt = pe.SolverFactory('appsi_ipopt')
        res = opt.solve(m)
        pe.assert_optimal_termination(res)
        self.assertAlmostEqual(pe.value(m.c.body) - pe.value(m.c.lower), 0)
