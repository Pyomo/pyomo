import pyutilib.th as unittest
import pyomo.environ as pe
from pyomo.common.dependencies import attempt_import
from pyomo.contrib.interior_point.inverse_reduced_hessian import inv_reduced_hessian_barrier

np, numpy_available = attempt_import('numpy', 'inverse_reduced_hessian numpy',
                                     minimum_version='1.13.0')
scipy, scipy_available = attempt_import('scipy', 'inverse_reduced_hessian requires scipy')
from pyomo.contrib.pynumero.extensions.asl import AmplInterface
asl_available = AmplInterface.available()
if not (numpy_available and scipy_available and asl_available):
    raise unittest.SkipTest('inverse_reduced_hessian tests require numpy, scipy, and asl')

class TestInverseReducedHessian(unittest.TestCase):
    def test_invrh_zavala_thesis(self):
        m = pe.ConcreteModel()
        m.x = pe.Var([1,2,3]) 
        m.obj = pe.Objective(expr=(m.x[1]-1)**2 + (m.x[2]-2)**2 + (m.x[3]-3)**2)
        m.c1 = pe.Constraint(expr=m.x[1] + 2*m.x[2] + 3*m.x[3]==0)

        status, invrh = inv_reduced_hessian_barrier(m, [m.x[2], m.x[3]])
        expected_invrh = np.asarray([[ 0.35714286, -0.21428571],
                                     [-0.21428571, 0.17857143]])
        np.testing.assert_array_almost_equal(invrh, expected_invrh)

