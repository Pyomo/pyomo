import pyomo.environ as pe
from pyomo.common.dependencies import attempt_import
import pyomo.common.unittest as unittest
parameterized, param_available = attempt_import('parameterized')
if not param_available:
    raise unittest.SkipTest('Parameterized is not available.')
parameterized = parameterized.parameterized
try:
    from pyomo.contrib.appsi.cmodel import cmodel
except ImportError:
    raise unittest.SkipTest('appsi extensions are not available')
from pyomo.contrib.appsi.base import TerminationCondition, Results, PersistentSolver
from pyomo.contrib.appsi.solvers import Gurobi, Ipopt, Cplex, Cbc
from typing import Type
from pyomo.core.expr.numeric_expr import LinearExpression
import os
from pyomo.common.getGSL import find_GSL


all_solvers = [('gurobi', Gurobi), ('ipopt', Ipopt), ('cplex', Cplex), ('cbc', Cbc)]
mip_solvers = [('gurobi', Gurobi), ('cplex', Cplex), ('cbc', Cbc)]
nlp_solvers = [('ipopt', Ipopt)]
qcp_solvers = [('gurobi', Gurobi), ('ipopt', Ipopt), ('cplex', Cplex)]
miqcqp_solvers = [('gurobi', Gurobi), ('cplex', Cplex)]


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
