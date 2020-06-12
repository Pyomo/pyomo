import pyutilib.th as unittest
import pyomo.environ as pe
from pyomo.core.base import ConcreteModel, Var, Constraint, Objective
from pyomo.common.dependencies import attempt_import

np, numpy_available = attempt_import('numpy', 'Interior point requires numpy',
        minimum_version='1.13.0')
scipy, scipy_available = attempt_import('scipy', 'Interior point requires scipy')
mumps, mumps_available = attempt_import('mumps', 'Interior point requires mumps')
if not (numpy_available and scipy_available):
    raise unittest.SkipTest('Interior point tests require numpy and scipy')

from pyomo.contrib.pynumero.asl import AmplInterface
asl_available = AmplInterface.available()
if not asl_available:
    raise unittest.SkipTest('Regularization tests require ASL')
import pyomo.contrib.interior_point as ip
from pyomo.contrib.pynumero.linalg.ma27 import MA27Interface
ma27_available = MA27Interface.available()


def make_model_tri(n, small_val=1e-7, big_val=1e2):
    m = ConcreteModel()
    m.x = Var(range(n), initialize=0.5)

    def c_rule(m, i):
        return big_val*m.x[i-1] + small_val*m.x[i] + big_val*m.x[i+1] == 1
    
    m.c = Constraint(range(1,n-1), rule=c_rule)

    m.obj = Objective(expr=small_val*sum((m.x[i]-1)**2 for i in range(n)))

    return m

class TestReallocation(unittest.TestCase):
    def _test_ip_with_reallocation(self, linear_solver, interface):
        ip_solver = ip.InteriorPointSolver(linear_solver,
                max_reallocation_iterations=3,
                reallocation_factor=1.1,
                # The small factor is to ensure that multiple iterations of
                # reallocation are performed. The bug in the previous
                # implementation only occurred if 2+ reallocation iterations
                # were needed (max_reallocation_iterations >= 3).
                max_iter=1)
        ip_solver.set_interface(interface)

        ip_solver.solve(interface)

        return ip_solver

    def test_mumps(self):
        n = 20000
        m = make_model_tri(n, small_val=1e-7)
        interface = ip.InteriorPointInterface(m)
        linear_solver = ip.linalg.MumpsInterface()
        linear_solver.set_icntl(14, 20) # Default memory "buffer" factor: 20

        kkt = interface.evaluate_primal_dual_kkt_matrix()
        res = linear_solver.do_symbolic_factorization(kkt)
        predicted = linear_solver.get_infog(16)

        self._test_ip_with_reallocation(linear_solver, interface)
        actual = linear_solver.get_icntl(23)

        self.assertEqual(predicted, 12)
        self.assertEqual(actual, 14)


if __name__ == '__main__':
    #
    unittest.main()
